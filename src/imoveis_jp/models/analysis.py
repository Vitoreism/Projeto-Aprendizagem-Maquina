# -*- coding: utf-8 -*-
"""
analise de residuos e importancia por permutacao

fecha o ciclo aberto pela matriz de correlacao. a correlacao diz o que anda
junto com o preco; a importancia por permutacao diz do que o modelo realmente
depende para acertar. a diferenca entre as duas listas e a parte interessante:
atributo com correlacao alta e importancia nula esta sendo explicado por outro
no lugar dele, e atributo com correlacao nula e importancia alta so funciona em
interacao -- coisa que a correlacao, por ser bivariada, nao consegue enxergar.

TUDO E MEDIDO NO CONJUNTO DE TESTE, de proposito. Importancia medida no treino
responde "do que o modelo se lembrou"; medida no teste responde "do que ele
precisa para acertar num imovel que nunca viu", que e a pergunta do projeto.
O preco disso e que o teste ja foi usado para relatar a metrica final -- por
isso nada aqui volta para dentro do modelo como selecao de feature. E leitura,
nao decisao.

Os residuos sao analisados em log e em reais ao mesmo tempo, porque as duas
escalas contam historias diferentes: o modelo minimiza erro em log, entao e la
que o residuo deve estar centrado e homogeneo, mas quem le o resultado quer
saber o erro em reais.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imoveis_jp import config
from imoveis_jp.models import dataset, train

SAIDA_RESIDUOS = config.PROCESSED / "residuos_teste.csv"
SAIDA_SEGMENTOS = config.PROCESSED / "residuos_por_segmento.csv"
SAIDA_IMPORTANCIA = config.PROCESSED / "importancia_permutacao.csv"
SAIDA_IMPORTANCIA_CODIFICADA = config.PROCESSED / "importancia_permutacao_codificada.csv"
SAIDA_RELATORIO = config.INTERIM / "relatorio_analise.json"
DIR_FIGURAS = config.ROOT / "docs" / "figuras"

#: o vencedor do train.py. a analise so faz sentido sobre o modelo que a gente
#: de fato reportaria.
MODELO = "gradient_boosting_ajustado"

#: repeticoes da permutacao. 10 ja da um desvio-padrao pequeno o bastante para
#: separar as features do topo; a lista codificada usa menos porque tem ~5x
#: mais colunas e o custo e linear nas duas dimensoes.
REPETICOES = 10
REPETICOES_CODIFICADA = 5

#: segmento com menos que isso nao entra na tabela: a mediana de 4 imoveis nao
#: sustenta conclusao nenhuma, e poluiria o ranking de pior bairro.
MINIMO_POR_SEGMENTO = 30

TOP_FIGURA = 20


def ajustar() -> Tuple:
    """Treina o modelo vencedor no treino e devolve tudo o que a analise precisa."""
    X, y, grupos = dataset.carregar()
    X_tr, X_te, y_tr, y_te, g_tr, g_te = dataset.dividir(X, y, grupos)
    if dataset.diagnostico_split(g_tr, g_te):
        raise RuntimeError("split vazou: o mesmo imovel esta no treino e no teste")

    numericas, binarias, categoricas = dataset.colunas_por_tipo(X)
    modelo = train.montar_modelos(numericas, binarias, categoricas)[MODELO]

    print(f"[Ajuste] {MODELO} em {len(X_tr)} anuncios de treino...", flush=True)
    modelo.fit(X_tr, y_tr)

    return modelo, X_tr, X_te, y_tr, y_te


def tabela_residuos(X_te: pd.DataFrame, y_te: pd.Series, previsto_log: np.ndarray) -> pd.DataFrame:
    """Um residuo por anuncio do teste, nas duas escalas, com os cortes de leitura.

    O sinal segue a convencao "erro do modelo": positivo = o modelo pediu caro
    demais. Assim o numero se le sozinho, sem consultar a formula.
    """
    real = np.exp(y_te.to_numpy())
    previsto = np.exp(previsto_log)

    tabela = pd.DataFrame(
        {
            "preco_real": real,
            "preco_previsto": previsto,
            "residuo_log": previsto_log - y_te.to_numpy(),
            "erro_reais": previsto - real,
            "erro_percentual": (previsto - real) / real * 100,
        },
        index=X_te.index,
    )
    tabela["erro_absoluto_percentual"] = tabela["erro_percentual"].abs()

    tabela["bairro"] = X_te["bairro"]
    tabela["origem_anuncio"] = X_te["origem_anuncio"]
    tabela["tipo_unidade"] = X_te["tipo_unidade"]
    tabela["area_util"] = X_te["area_util"]

    # quintis do preco REAL, nao do previsto: a pergunta e "o modelo erra mais
    # em imovel caro?", e usar o previsto para definir a faixa faria o proprio
    # erro escolher em que faixa o imovel cai.
    tabela["faixa_preco"] = _faixas_por_quantil(
        tabela["preco_real"], ["Q1 (mais barato)", "Q2", "Q3", "Q4", "Q5 (mais caro)"]
    )
    tabela["faixa_area"] = _faixas_por_quantil(
        tabela["area_util"], ["A1 (menor)", "A2", "A3", "A4 (maior)"]
    )

    # quantas das 8 continuas o anuncio deixou em branco. serve para separar
    # erro do modelo de erro por falta de informacao.
    numericas = [c for c in dataset.NUMERICAS if c in X_te.columns]
    tabela["campos_ausentes"] = X_te[numericas].isna().sum(axis=1)

    return tabela


def _faixas_por_quantil(valores: pd.Series, rotulos: List[str]) -> pd.Series:
    """qcut que nao quebra quando faltam quantis distintos.

    Com os 3.167 anuncios do teste os cortes sempre existem, mas a funcao passa
    a valer para qualquer recorte -- um bairro so, por exemplo, ou a base de
    teste de um experimento pequeno. Quando as bordas colidem, o numero de
    faixas cai e os rotulos deixam de descrever a posicao relativa, entao viram
    genericos em vez de mentir ('Q5 (mais caro)' num corte de 2 faixas).
    """
    limpos = valores.dropna()
    if limpos.empty:
        return pd.Series(pd.Categorical([None] * len(valores)), index=valores.index)

    k = len(rotulos)
    bordas = np.unique(np.quantile(limpos, np.linspace(0, 1, k + 1)))
    if len(bordas) < 2:
        unico = pd.Categorical(
            [rotulos[0]] * len(valores), categories=[rotulos[0]], ordered=True
        )
        return pd.Series(unico, index=valores.index)

    faixas = len(bordas) - 1
    usados = rotulos if faixas == k else [f"F{i + 1}/{faixas}" for i in range(faixas)]
    return pd.cut(valores, bins=bordas, labels=usados, include_lowest=True, ordered=True)


def resumo_por_segmento(tabela: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """Erro mediano e vies por categoria, so onde ha volume para sustentar."""
    agrupado = tabela.groupby(coluna, observed=True).agg(
        n=("erro_percentual", "size"),
        preco_mediano=("preco_real", "median"),
        # mediana do erro COM sinal = vies. se for longe de zero, o modelo
        # erra sistematicamente para um lado naquele segmento.
        vies_percentual=("erro_percentual", "median"),
        # mediana do erro ABSOLUTO = precisao, independente do lado.
        erro_mediano_percentual=("erro_absoluto_percentual", "median"),
        mae_reais=("erro_reais", lambda s: float(np.mean(np.abs(s)))),
    )
    agrupado = agrupado[agrupado["n"] >= MINIMO_POR_SEGMENTO]
    agrupado = agrupado.reset_index().rename(columns={coluna: "categoria"})
    agrupado.insert(0, "segmento", coluna)
    return agrupado.sort_values("erro_mediano_percentual", ascending=False)


def importancia_por_atributo(modelo, X_te: pd.DataFrame, y_te: pd.Series) -> pd.DataFrame:
    """Permuta as colunas como elas existem na matriz, antes do one-hot.

    Permutar 'bairro' inteiro (e nao cada dummy separada) responde a pergunta
    util: quanto o modelo perde se o bairro virar ruido? Se cada dummy fosse
    permutada isolada, o modelo ainda saberia o bairro pelas outras 328 colunas
    do mesmo grupo e a importancia sairia artificialmente perto de zero.
    """
    print(f"[Permutacao] {X_te.shape[1]} atributos x {REPETICOES} repeticoes...", flush=True)
    resultado = permutation_importance(
        modelo,
        X_te,
        y_te,
        scoring="neg_mean_absolute_error",
        n_repeats=REPETICOES,
        random_state=dataset.SEMENTE,
        # em serie pelo mesmo motivo do train.py: o HistGradientBoosting ja e
        # multi-thread por OpenMP.
        n_jobs=None,
    )
    return _tabela_importancia(list(X_te.columns), resultado)


def importancia_por_coluna_codificada(modelo, X_te: pd.DataFrame, y_te: pd.Series) -> pd.DataFrame:
    """Mesma medida, agora dummy a dummy, depois do ColumnTransformer.

    Serve para uma pergunta especifica: as dummies quase-constantes que o corte
    por FREQUENCIA_MAXIMA_DUMMY removia (ver build_features) valem alguma coisa?
    Aqui elas aparecem separadas e a resposta e um numero, nao um argumento.

    O regressor ja esta ajustado sobre a matriz transformada, entao basta
    transformar o teste e permutar nele -- sem reajustar nada.
    """
    preparo = modelo.named_steps["preparo"]
    regressor = modelo.named_steps["regressor"]

    Z_te = preparo.transform(X_te)
    # o ColumnTransformer prefixa com o nome do bloco ('num__', 'cat__'); tirar
    # o prefixo faz os nomes baterem com os do ranking de correlacao.
    nomes = [n.split("__", 1)[-1] for n in preparo.get_feature_names_out()]

    print(
        f"[Permutacao] {len(nomes)} colunas codificadas x {REPETICOES_CODIFICADA} repeticoes...",
        flush=True,
    )
    resultado = permutation_importance(
        regressor,
        Z_te,
        y_te,
        scoring="neg_mean_absolute_error",
        n_repeats=REPETICOES_CODIFICADA,
        random_state=dataset.SEMENTE,
        n_jobs=None,
    )
    return _tabela_importancia(nomes, resultado)


def _tabela_importancia(nomes: List[str], resultado) -> pd.DataFrame:
    """Importancia em MAE(log): quanto o erro piora quando a coluna vira ruido."""
    tabela = pd.DataFrame(
        {
            "feature": nomes,
            "importancia": resultado.importances_mean,
            "desvio": resultado.importances_std,
        }
    )
    # importancia menor que o proprio ruido da permutacao nao e distinguivel de
    # zero. a coluna e informativa, nao um filtro -- quem le decide.
    tabela["significativa"] = tabela["importancia"] > 2 * tabela["desvio"]
    return tabela.sort_values("importancia", ascending=False).reset_index(drop=True)


def confrontar_com_correlacao(importancia: pd.DataFrame) -> pd.DataFrame:
    """Junta importancia por permutacao com o |Spearman| contra o preco.

    A correlacao e bivariada e o modelo e multivariado: as duas listas nao tem
    obrigacao de coincidir, e onde elas discordam esta o que so um dos dois
    metodos enxerga.
    """
    caminho = config.PROCESSED / "correlacao_alvo.csv"
    if not caminho.exists():
        print(f"[Aviso] '{caminho.name}' nao existe; pulando o confronto.", flush=True)
        return pd.DataFrame()

    correlacao = pd.read_csv(caminho)[["feature", "forca"]].rename(
        columns={"forca": "spearman_abs"}
    )
    juntas = importancia.merge(correlacao, on="feature", how="inner")
    juntas["posto_importancia"] = juntas["importancia"].rank(ascending=False).astype(int)
    juntas["posto_correlacao"] = juntas["spearman_abs"].rank(ascending=False).astype(int)
    juntas["salto_de_posto"] = juntas["posto_correlacao"] - juntas["posto_importancia"]
    return juntas.sort_values("importancia", ascending=False).reset_index(drop=True)


def plotar_residuos(tabela: pd.DataFrame) -> None:
    """Quatro paineis: o diagnostico classico de regressao, em uma figura."""
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig, eixos = plt.subplots(2, 2, figsize=(14, 10))

    # (1) residuo contra previsto, em log. o que se procura aqui e formato: uma
    # nuvem horizontal centrada em zero significa que o modelo nao tem vies
    # sistematico ao longo da faixa de preco.
    ax = eixos[0, 0]
    ax.scatter(tabela["preco_previsto"], tabela["residuo_log"], s=6, alpha=0.2, color="#3b6ea5")
    ax.axhline(0, color="#c0392b", lw=1.2)
    faixas = pd.qcut(tabela["preco_previsto"], 20, duplicates="drop")
    mediana = tabela.groupby(faixas, observed=True).agg(
        x=("preco_previsto", "median"), y=("residuo_log", "median")
    )
    ax.plot(mediana["x"], mediana["y"], color="#c0392b", lw=2, label="mediana por vintil")
    ax.set_xscale("log")
    ax.set_xlabel("preço previsto (R$, escala log)")
    ax.set_ylabel("resíduo em log (previsto − real)")
    ax.set_title("Resíduo × valor previsto")
    ax.legend(fontsize=8)

    # (2) distribuicao do residuo. cauda simetrica e sinal de que o log
    # resolveu a assimetria do alvo.
    ax = eixos[0, 1]
    ax.hist(tabela["residuo_log"], bins=80, color="#3b6ea5", alpha=0.85)
    ax.axvline(0, color="#c0392b", lw=1.2)
    ax.axvline(
        tabela["residuo_log"].median(),
        color="#e67e22",
        lw=1.4,
        ls="--",
        label=f"mediana = {tabela['residuo_log'].median():+.3f}",
    )
    ax.set_xlabel("resíduo em log")
    ax.set_ylabel("anúncios")
    ax.set_title("Distribuição dos resíduos")
    ax.legend(fontsize=8)

    # (3) erro por faixa de preco. a pergunta que motivou a analise.
    ax = eixos[1, 0]
    # categoria vazia faria o boxplot receber array vazio e quebrar
    faixas_preco = [
        c for c in tabela["faixa_preco"].cat.categories if (tabela["faixa_preco"] == c).any()
    ]
    dados = [tabela.loc[tabela["faixa_preco"] == f, "erro_percentual"].to_numpy() for f in faixas_preco]
    ax.boxplot(dados, tick_labels=faixas_preco, showfliers=False)
    ax.axhline(0, color="#c0392b", lw=1.2)
    ax.set_ylabel("erro percentual (previsto − real)")
    ax.set_title("Erro por faixa de preço")
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)

    # (4) previsto contra real, log-log. a diagonal e o acerto perfeito; o
    # achatamento nas pontas e a regressao a media que todo modelo tem.
    ax = eixos[1, 1]
    ax.scatter(tabela["preco_real"], tabela["preco_previsto"], s=6, alpha=0.2, color="#3b6ea5")
    limites = [tabela["preco_real"].min(), tabela["preco_real"].max()]
    ax.plot(limites, limites, color="#c0392b", lw=1.2, label="acerto perfeito")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("preço real (R$)")
    ax.set_ylabel("preço previsto (R$)")
    ax.set_title("Previsto × real")
    ax.legend(fontsize=8)

    fig.suptitle(f"Diagnóstico de resíduos — {MODELO}, conjunto de teste", fontsize=14)
    fig.tight_layout()
    caminho = DIR_FIGURAS / "residuos_diagnostico.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figura] {caminho}", flush=True)


def plotar_importancia(importancia: pd.DataFrame) -> None:
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    top = importancia.head(TOP_FIGURA).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(top) + 2))
    ax.barh(
        top["feature"],
        top["importancia"],
        xerr=top["desvio"],
        color=["#3b6ea5" if s else "#b0b7bd" for s in top["significativa"]],
        error_kw={"lw": 0.8, "ecolor": "#555"},
    )
    ax.set_xlabel("aumento no MAE em log ao embaralhar o atributo")
    ax.set_title(
        f"Importância por permutação — {MODELO}, conjunto de teste\n"
        "(cinza = não distinguível de zero)",
        fontsize=12,
    )
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    caminho = DIR_FIGURAS / "importancia_permutacao.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figura] {caminho}", flush=True)


def executar() -> Dict:
    config.ensure_dirs()

    modelo, X_tr, X_te, y_tr, y_te = ajustar()
    previsto_log = modelo.predict(X_te)

    geral = train.metricas_em_reais(y_te.to_numpy(), previsto_log)
    print(
        f"[Teste] MAE=R$ {geral['mae_reais']:,.0f} | "
        f"erro mediano={geral['erro_percentual_mediano']:.1f}% | R2(log)={geral['r2_log']:.3f}",
        flush=True,
    )

    tabela = tabela_residuos(X_te, y_te, previsto_log)
    tabela.to_csv(SAIDA_RESIDUOS, index=False, encoding="utf-8")

    segmentos = pd.concat(
        [
            resumo_por_segmento(tabela, "faixa_preco"),
            resumo_por_segmento(tabela, "faixa_area"),
            resumo_por_segmento(tabela, "origem_anuncio"),
            resumo_por_segmento(tabela, "tipo_unidade"),
            resumo_por_segmento(tabela, "campos_ausentes"),
            resumo_por_segmento(tabela, "bairro"),
        ],
        ignore_index=True,
    )
    segmentos.to_csv(SAIDA_SEGMENTOS, index=False, encoding="utf-8")

    importancia = importancia_por_atributo(modelo, X_te, y_te)
    importancia.to_csv(SAIDA_IMPORTANCIA, index=False, encoding="utf-8")

    codificada = importancia_por_coluna_codificada(modelo, X_te, y_te)
    confronto = confrontar_com_correlacao(codificada)
    if len(confronto):
        codificada = confronto
    codificada.to_csv(SAIDA_IMPORTANCIA_CODIFICADA, index=False, encoding="utf-8")

    plotar_residuos(tabela)
    plotar_importancia(importancia)

    relatorio = {
        "modelo": MODELO,
        "semente": dataset.SEMENTE,
        "n_teste": int(len(X_te)),
        "metricas_teste": geral,
        "residuo_log": {
            "media": float(tabela["residuo_log"].mean()),
            "mediana": float(tabela["residuo_log"].median()),
            "desvio": float(tabela["residuo_log"].std()),
            "assimetria": float(tabela["residuo_log"].skew()),
        },
        "top_importancia": importancia.head(10).to_dict("records"),
        "piores_segmentos": segmentos.groupby("segmento").head(1).to_dict("records"),
    }
    with open(SAIDA_RELATORIO, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2, default=str)

    _imprimir_resumo(tabela, segmentos, importancia, codificada)

    print(f"\nResiduos:            {SAIDA_RESIDUOS}")
    print(f"Segmentos:           {SAIDA_SEGMENTOS}")
    print(f"Importancia:         {SAIDA_IMPORTANCIA}")
    print(f"Importancia (dummy): {SAIDA_IMPORTANCIA_CODIFICADA}")
    print(f"Relatorio:           {SAIDA_RELATORIO}")

    return relatorio


def _imprimir_resumo(
    tabela: pd.DataFrame,
    segmentos: pd.DataFrame,
    importancia: pd.DataFrame,
    codificada: pd.DataFrame,
) -> None:
    print("\n" + "=" * 78)
    print("RESIDUOS -- o residuo em log deve estar centrado em zero")
    print("=" * 78)
    print(
        f"media={tabela['residuo_log'].mean():+.4f}  "
        f"mediana={tabela['residuo_log'].median():+.4f}  "
        f"desvio={tabela['residuo_log'].std():.4f}  "
        f"assimetria={tabela['residuo_log'].skew():+.3f}"
    )

    print("\n" + "=" * 78)
    print("ERRO POR SEGMENTO (vies = mediana com sinal; + significa caro demais)")
    print("=" * 78)
    for nome, bloco in segmentos.groupby("segmento", sort=False):
        print(f"\n-- {nome}")
        print(f"{'categoria':28s} {'n':>6s} {'vies %':>9s} {'erro med %':>11s} {'MAE R$':>13s}")
        for _, l in bloco.head(8).iterrows():
            print(
                f"{str(l['categoria'])[:28]:28s} {l['n']:6d} {l['vies_percentual']:+9.1f} "
                f"{l['erro_mediano_percentual']:11.1f} {l['mae_reais']:13,.0f}"
            )

    print("\n" + "=" * 78)
    print("IMPORTANCIA POR PERMUTACAO (atributo inteiro, no teste)")
    print("=" * 78)
    print(f"{'atributo':30s} {'MAE(log)':>10s} {'desvio':>9s}  significativa")
    for _, l in importancia.head(15).iterrows():
        print(
            f"{l['feature'][:30]:30s} {l['importancia']:+10.4f} {l['desvio']:9.4f}  "
            f"{'sim' if l['significativa'] else 'nao'}"
        )

    nulas = int((~importancia["significativa"]).sum())
    print(
        f"\n{nulas} de {len(importancia)} atributos tem importancia indistinguivel de zero."
    )

    if "salto_de_posto" in codificada.columns:
        print("\n" + "=" * 78)
        print("CORRELACAO x IMPORTANCIA -- onde os dois metodos discordam")
        print("=" * 78)
        print(f"{'feature':30s} {'|spearman|':>11s} {'importancia':>12s} {'salto':>7s}")
        subida = codificada.nlargest(6, "salto_de_posto")
        descida = codificada.nsmallest(6, "salto_de_posto")
        for rotulo, bloco in (("o modelo usa mais do que a correlacao sugeria", subida),
                              ("a correlacao prometia mais do que o modelo usa", descida)):
            print(f"\n-- {rotulo}")
            for _, l in bloco.iterrows():
                print(
                    f"{l['feature'][:30]:30s} {l['spearman_abs']:11.3f} "
                    f"{l['importancia']:+12.4f} {l['salto_de_posto']:+7d}"
                )
    print("=" * 78)


if __name__ == "__main__":
    executar()
