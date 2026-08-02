# -*- coding: utf-8 -*-
"""
preenche campos estruturados ausentes a partir da descricao em texto livre (issue #3)

muitos anunciantes nao preenchem os campos estruturados do portal e jogam tudo na
descricao. o scraper le so o html estruturado, entao esses imoveis chegam com
quartos, suites, banheiros ou vagas nulos -- justamente as variaveis mais
preditivas do modelo.

a extracao e por regra, nao por llm: e deterministica, testavel, roda sem chave de
api e a precisao foi medida contra os campos que ja existem (ver validar()).

nao ha risco de vazamento: cada linha e lida isoladamente, sem nenhuma estatistica
agregada, entao pode rodar antes do split.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from imoveis_jp import config

ENTRADA = config.PROCESSED / "imoveis_joao_pessoa_global_deduplicated.csv"
SAIDA_CSV = config.INTERIM / "enriquecido_da_descricao.csv"

COLUNA_TEXTO = "descricao_completa"

#: aparecem ~900 vezes na base ("dois quartos", "tres suites").
NUMEROS_POR_EXTENSO = {
    "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3,
    "quatro": 4, "cinco": 5, "seis": 6, "sete": 7,
}

_ALTERNATIVAS = "|".join(NUMEROS_POR_EXTENSO)

#: campo -> (padrao, minimo, maximo). os limites sao os mesmos de
#: build_features.LIMITES_PLAUSIBILIDADE, repetidos aqui para nao criar
#: dependencia circular entre processing e features.
EXTRATORES: Dict[str, Tuple[str, float, float]] = {
    "quartos": (rf"(\d+|{_ALTERNATIVAS})\s*(?:quartos?|dormitorios?|dorms?)\b", 0, 10),
    "suites": (rf"(\d+|{_ALTERNATIVAS})\s*suites?\b", 0, 10),
    "banheiros": (rf"(\d+|{_ALTERNATIVAS})\s*banheiros?\b", 0, 15),
    "garagens": (rf"(\d+|{_ALTERNATIVAS})\s*vagas?\b", 0, 10),
}

#: 'area_util' fica de fora de proposito. as descricoes citam varias areas
#: (privativa, lazer, terreno, outra unidade do predio) e nenhum padrao passou
#: de 67% de acerto contra o valor estruturado, nem com contexto explicito.
#: preencher 6,5% de ausencia injetando ~35% de erro na feature mais forte do
#: modelo sai pior que manter o nulo, que o imputador ja trata.
CAMPO_NAO_EXTRAIDO = "area_util"


def normalizar(texto: Any) -> str:
    """Minusculas sem acento, espacos colapsados."""
    if not isinstance(texto, str):
        return ""
    decomposto = unicodedata.normalize("NFD", texto)
    sem_acento = "".join(c for c in decomposto if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", sem_acento.lower())


def _para_numero(bruto: str) -> Optional[float]:
    if bruto.isdigit():
        return float(bruto)
    return float(NUMEROS_POR_EXTENSO[bruto]) if bruto in NUMEROS_POR_EXTENSO else None


def extrair(texto: str, campo: str) -> Optional[float]:
    """Primeira ocorrencia do campo no texto ja normalizado, ou None."""
    padrao, minimo, maximo = EXTRATORES[campo]
    achado = re.search(padrao, texto)
    if not achado:
        return None

    valor = _para_numero(achado.group(1))
    if valor is None or not (minimo <= valor <= maximo):
        return None
    return valor


def extrair_serie(textos: pd.Series, campo: str) -> pd.Series:
    return textos.map(lambda t: extrair(t, campo))


def validar(df: pd.DataFrame) -> pd.DataFrame:
    """Mede a precisao onde o campo estruturado JA existe.

    E o que autoriza usar a extracao onde ele nao existe: sem esta medida, o
    preenchimento seria um chute com aparencia de dado.
    """
    textos = df[COLUNA_TEXTO].map(normalizar)
    linhas = []

    for campo in EXTRATORES:
        if campo not in df.columns:
            continue
        extraido = extrair_serie(textos, campo)
        conhecido = pd.to_numeric(df[campo], errors="coerce")
        comparavel = extraido.notna() & conhecido.notna()

        if comparavel.sum() == 0:
            continue

        diferenca = (extraido[comparavel] - conhecido[comparavel]).abs()
        linhas.append(
            {
                "campo": campo,
                "comparaveis": int(comparavel.sum()),
                "exato": float((diferenca == 0).mean()),
                "ate_1_de_diferenca": float((diferenca <= 1).mean()),
            }
        )

    return pd.DataFrame(linhas)


def enriquecer(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Preenche SO as celulas ausentes. Valor ja informado pelo portal manda."""
    if COLUNA_TEXTO not in df.columns:
        return df, {}

    textos = df[COLUNA_TEXTO].map(normalizar)
    preenchidos: Dict[str, int] = {}

    for campo in EXTRATORES:
        if campo not in df.columns:
            continue

        atual = pd.to_numeric(df[campo], errors="coerce")
        ausentes = atual.isna()
        if not ausentes.any():
            continue

        extraido = extrair_serie(textos[ausentes], campo)
        recuperados = extraido.notna()
        if not recuperados.any():
            continue

        atual.loc[extraido.index[recuperados]] = extraido[recuperados]
        df[campo] = atual
        preenchidos[campo] = int(recuperados.sum())

    if preenchidos:
        detalhe = ", ".join(f"{c}={n}" for c, n in sorted(preenchidos.items()))
        print(f"[Descricao] Celulas preenchidas a partir do texto: {detalhe}.", flush=True)

    return df, preenchidos


def executar(mostrar_validacao: bool = True) -> pd.DataFrame:
    if not ENTRADA.exists():
        print(f"[Erro] Base deduplicada nao encontrada: {ENTRADA}", flush=True)
        sys.exit(1)

    config.ensure_dirs()
    df = pd.read_csv(ENTRADA, low_memory=False)
    print(f"[Info] {len(df)} anuncios lidos de {ENTRADA.name}.", flush=True)

    # import adiado: features depende de processing, entao no topo do modulo
    # isso seria import circular. sem este preparo, 'suites' e 'banheiros'
    # ainda estao com os True/False da colisao de nomes e o relatorio abaixo
    # mediria a precisao sobre uma amostra minuscula e enviesada.
    from imoveis_jp.features.build_features import (
        aplicar_limites,
        converter_colunas_numericas,
        reparar_numericos,
    )

    df, _ = reparar_numericos(df)
    df = converter_colunas_numericas(df)
    df, _ = aplicar_limites(df)

    com_texto = df[COLUNA_TEXTO].map(normalizar).str.len() > 40
    print(f"[Info] {com_texto.sum()} com descricao utilizavel ({com_texto.mean():.1%}).", flush=True)

    if mostrar_validacao:
        relatorio = validar(df)
        print("\n" + "=" * 62)
        print("PRECISAO MEDIDA ONDE O CAMPO ESTRUTURADO JA EXISTE")
        print("=" * 62)
        print(f"{'campo':12s} {'comparaveis':>12s} {'exato':>9s} {'ate +-1':>9s}")
        for _, l in relatorio.iterrows():
            print(
                f"{l['campo']:12s} {l['comparaveis']:12d} "
                f"{l['exato']:8.1%} {l['ate_1_de_diferenca']:8.1%}"
            )
        print("=" * 62)
        print(f"'{CAMPO_NAO_EXTRAIDO}' nao e extraido: ver o comentario no modulo.\n")

    antes = {c: int(pd.to_numeric(df[c], errors="coerce").isna().sum()) for c in EXTRATORES}
    df, preenchidos = enriquecer(df)
    depois = {c: int(pd.to_numeric(df[c], errors="coerce").isna().sum()) for c in EXTRATORES}

    print("=" * 62)
    print("AUSENCIAS ANTES E DEPOIS")
    print("=" * 62)
    print(f"{'campo':12s} {'antes':>9s} {'depois':>9s} {'recuperado':>12s}")
    for campo in EXTRATORES:
        recuperado = antes[campo] - depois[campo]
        proporcao = recuperado / antes[campo] if antes[campo] else 0.0
        print(f"{campo:12s} {antes[campo]:9d} {depois[campo]:9d} {recuperado:8d} ({proporcao:.0%})")
    print("=" * 62)

    SAIDA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAIDA_CSV, index=False, encoding="utf-8")
    print(f"Base enriquecida salva em: {SAIDA_CSV}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preenche campos estruturados ausentes a partir da descricao (issue #3)."
    )
    parser.add_argument(
        "--sem-validacao",
        action="store_true",
        help="Pula o relatorio de precisao contra os campos ja preenchidos.",
    )
    args = parser.parse_args()
    executar(mostrar_validacao=not args.sem_validacao)


if __name__ == "__main__":
    main()
