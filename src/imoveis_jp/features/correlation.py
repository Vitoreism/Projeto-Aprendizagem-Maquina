# -*- coding: utf-8 -*-
"""
matriz de correlacao e selecao de atributos sobre a matriz de features

produz tres coisas: o ranking de cada atributo contra o alvo (preco), a poda dos
atributos redundantes entre si, e o diagnostico de artefato de coleta -- atributo
que correlaciona com o portal de origem esta medindo o vocabulario do site, nao
o imovel.
"""

from __future__ import annotations

import json
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from imoveis_jp import config

ENTRADA = config.PROCESSED / "features_matrix.csv"
SAIDA_RANKING = config.PROCESSED / "correlacao_alvo.csv"
SAIDA_SELECAO = config.PROCESSED / "features_selecionadas.csv"
SAIDA_REDUNDANCIA = config.PROCESSED / "pares_redundantes.csv"
DIR_FIGURAS = config.ROOT / "docs" / "figuras"

ALVO = "preco_venda"

#: acima disso, duas features carregam praticamente a mesma informacao.
LIMIAR_REDUNDANCIA = 0.85

#: acima disso, a feature esta descrevendo o portal, nao o imovel.
LIMIAR_ARTEFATO = 0.50

TOP_HEATMAP = 30


def carregar() -> pd.DataFrame:
    if not ENTRADA.exists():
        print(
            f"[Erro] '{ENTRADA}' nao existe. Rode antes:\n"
            "       python -m imoveis_jp.features.build_features",
            flush=True,
        )
        sys.exit(1)

    df = pd.read_csv(ENTRADA, low_memory=False)
    print(f"[Info] Matriz de features: {df.shape[0]} imoveis x {df.shape[1]} colunas.", flush=True)
    return df


def preparar_alvos(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Alvo bruto, em log (preco imobiliario e assimetrico) e por m2."""
    antes = len(df)
    df = df[df[ALVO].notna() & (df[ALVO] > 0)].copy()
    print(f"[Info] {antes - len(df)} imoveis sem preco descartados; restam {len(df)}.", flush=True)

    df["log_preco"] = np.log(df[ALVO])

    area = df["area_util"].where(df["area_util"] > 0)
    df["preco_m2"] = df[ALVO] / area

    return df, [ALVO, "log_preco", "preco_m2"]


def matriz_correlacao(df: pd.DataFrame, features: List[str], alvos: List[str], metodo: str) -> pd.DataFrame:
    return df[features + alvos].corr(method=metodo, numeric_only=True)


def ranking_contra_alvo(
    df: pd.DataFrame, features: List[str], alvos: List[str]
) -> pd.DataFrame:
    """Pearson e Spearman de cada feature contra cada alvo.

    Feature binaria contra alvo continuo e correlacao ponto-bisserial, e feature
    binaria contra binaria e coeficiente phi -- ambas sao a formula de Pearson,
    entao uma matriz so cobre os tres casos.
    """
    linhas = []
    for feat in features:
        registro: Dict[str, object] = {
            "feature": feat,
            "cobertura": float(df[feat].notna().mean()),
            "media": float(df[feat].mean(skipna=True)),
        }
        for alvo in alvos:
            registro[f"pearson_{alvo}"] = df[feat].corr(df[alvo], method="pearson")
            registro[f"spearman_{alvo}"] = df[feat].corr(df[alvo], method="spearman")
        linhas.append(registro)

    ranking = pd.DataFrame(linhas)
    ranking["forca"] = ranking[f"spearman_{ALVO}"].abs()
    return ranking.sort_values("forca", ascending=False).reset_index(drop=True)


def detectar_artefatos_de_portal(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Features que na verdade separam chaves na mao de zapimoveis.

    Os dois portais publicam vocabularios de comodidade diferentes, entao uma
    comodidade que so existe num deles fica correlacionada com a origem e nao
    com o imovel. Depois da unificacao de sinonimos isso deve ser residual.
    """
    colunas_origem = [c for c in df.columns if c.startswith("origem_anuncio_")]
    if not colunas_origem:
        return pd.DataFrame(columns=["feature", "origem", "correlacao"])

    linhas = []
    for feat in features:
        if feat in colunas_origem:
            continue
        for origem in colunas_origem:
            r = df[feat].corr(df[origem])
            if pd.notna(r) and abs(r) >= LIMIAR_ARTEFATO:
                linhas.append({"feature": feat, "origem": origem, "correlacao": r})

    return pd.DataFrame(linhas).sort_values("correlacao", key=abs, ascending=False) if linhas else pd.DataFrame(
        columns=["feature", "origem", "correlacao"]
    )


def podar_redundantes(
    corr: pd.DataFrame, ranking: pd.DataFrame, features: List[str]
) -> Tuple[List[str], pd.DataFrame]:
    """De cada par com |r| alto, mantem o mais correlacionado com o alvo."""
    forca = dict(zip(ranking["feature"], ranking["forca"].fillna(0.0)))

    sub = corr.loc[features, features].abs()
    triangulo = sub.where(np.triu(np.ones(sub.shape), k=1).astype(bool))

    pares = (
        triangulo.stack()
        .reset_index()
        .set_axis(["feature_a", "feature_b", "correlacao"], axis=1)
    )
    pares = pares[pares["correlacao"] >= LIMIAR_REDUNDANCIA].sort_values(
        "correlacao", ascending=False
    )

    descartadas = set()
    decisoes = []
    for _, linha in pares.iterrows():
        a, b = linha["feature_a"], linha["feature_b"]
        if a in descartadas or b in descartadas:
            continue
        perdedora = a if forca.get(a, 0.0) < forca.get(b, 0.0) else b
        vencedora = b if perdedora == a else a
        descartadas.add(perdedora)
        decisoes.append(
            {
                "mantida": vencedora,
                "descartada": perdedora,
                "correlacao_entre_elas": linha["correlacao"],
                "forca_mantida": forca.get(vencedora, 0.0),
                "forca_descartada": forca.get(perdedora, 0.0),
            }
        )

    selecionadas = [f for f in features if f not in descartadas]
    return selecionadas, pd.DataFrame(decisoes)


def plotar_heatmap(corr: pd.DataFrame, features: List[str], nome: str, titulo: str) -> None:
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    sub = corr.loc[features, features]

    lado = max(8.0, 0.42 * len(features))
    plt.figure(figsize=(lado, lado * 0.85))
    mascara = np.triu(np.ones(sub.shape), k=1).astype(bool)

    sns.heatmap(
        sub,
        mask=mascara,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.55, "label": "correlação de Spearman"},
        annot=len(features) <= 20,
        fmt=".2f",
        annot_kws={"size": 7},
    )
    plt.title(titulo, fontsize=13, pad=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    caminho = DIR_FIGURAS / nome
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figura] {caminho}", flush=True)


def executar() -> None:
    config.ensure_dirs()
    df = carregar()
    df, alvos = preparar_alvos(df)

    ignorar = set(alvos) | {"url_anuncio"}
    features = [
        c
        for c in df.columns
        if c not in ignorar and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1
    ]
    print(f"[Info] {len(features)} features candidatas (constantes ja excluidas).", flush=True)

    ranking = ranking_contra_alvo(df, features, alvos)
    ranking.to_csv(SAIDA_RANKING, index=False, encoding="utf-8")
    print(f"[Saida] Ranking contra o alvo: {SAIDA_RANKING}", flush=True)

    corr_spearman = matriz_correlacao(df, features, alvos, "spearman")

    artefatos = detectar_artefatos_de_portal(df, features)
    if len(artefatos):
        print(
            f"\n[Alerta] {len(artefatos)} feature(s) correlacionam >= "
            f"{LIMIAR_ARTEFATO:.0%} com o portal de origem:",
            flush=True,
        )
        for _, l in artefatos.head(10).iterrows():
            print(f"          {l['feature']:38s} r={l['correlacao']:+.2f} vs {l['origem']}", flush=True)
    else:
        print("\n[OK] Nenhuma feature dominada pelo portal de origem.", flush=True)

    selecionadas, decisoes = podar_redundantes(corr_spearman, ranking, features)
    decisoes.to_csv(SAIDA_REDUNDANCIA, index=False, encoding="utf-8")
    print(
        f"\n[Poda] {len(decisoes)} par(es) com |r| >= {LIMIAR_REDUNDANCIA}; "
        f"{len(features)} -> {len(selecionadas)} features.",
        flush=True,
    )
    for _, l in decisoes.head(10).iterrows():
        print(f"        mantida '{l['mantida']}' / descartada '{l['descartada']}' (r={l['correlacao_entre_elas']:.2f})", flush=True)

    pd.DataFrame({"feature": selecionadas}).merge(
        ranking, on="feature", how="left"
    ).sort_values("forca", ascending=False).to_csv(SAIDA_SELECAO, index=False, encoding="utf-8")
    print(f"[Saida] Features selecionadas: {SAIDA_SELECAO}", flush=True)

    top = ranking[ranking["feature"].isin(selecionadas)].head(TOP_HEATMAP)["feature"].tolist()
    plotar_heatmap(
        corr_spearman,
        top + [ALVO],
        "heatmap_top30.png",
        f"Top {TOP_HEATMAP} atributos por correlação com o preço (Spearman)",
    )
    plotar_heatmap(
        corr_spearman,
        selecionadas + alvos,
        "heatmap_completo.png",
        "Matriz de correlação — atributos selecionados (Spearman)",
    )

    print("\n" + "=" * 65)
    print(f"TOP 15 ATRIBUTOS POR CORRELACAO COM '{ALVO}'")
    print("=" * 65)
    print(f"{'atributo':38s} {'spearman':>9s} {'pearson':>9s} {'presenca':>9s}")
    for _, l in ranking.head(15).iterrows():
        print(
            f"{l['feature']:38s} {l[f'spearman_{ALVO}']:+9.3f} "
            f"{l[f'pearson_{ALVO}']:+9.3f} {l['media']:9.2f}"
        )
    print("=" * 65)


if __name__ == "__main__":
    executar()
