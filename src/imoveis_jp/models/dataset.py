# -*- coding: utf-8 -*-
"""
preparacao dos dados para modelagem: alvo, features, agrupamento e split

o split e agrupado, nao aleatorio simples. a base tem o mesmo imovel anunciado
varias vezes -- 1.050 grupos somando 2.328 anuncios, o maior com 7 copias --
e um split aleatorio colocaria o mesmo apartamento no treino e no teste. a
metrica mediria memorizacao, nao generalizacao.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from imoveis_jp import config

MATRIZ = config.PROCESSED / "features_matrix.csv"

ALVO = "preco_venda"
IDENTIFICADOR = "url_anuncio"

#: fixa e versionada: sem ela o resultado nao e reproduzivel.
SEMENTE = 42

PROPORCAO_TESTE = 0.2

#: as unicas continuas da base; o resto e binaria de one-hot.
NUMERICAS = [
    "area_util",
    "area_total",
    "quartos",
    "suites",
    "banheiros",
    "garagens",
    "condominio",
    "iptu",
]

#: nominais que chegam como texto e sao codificadas dentro do Pipeline.
CATEGORICAS = ["posicao_solar", "status_construcao", "tipo_unidade", "origem_anuncio", "bairro"]

#: campos que identificam o imovel fisico, para agrupar anuncios repetidos.
ASSINATURA = ["preco_venda", "area_util", "quartos", "banheiros", "garagens"]


def assinatura_imovel(df: pd.DataFrame) -> pd.Series:
    """Identidade fisica do imovel, para o mesmo apartamento nao cair dos dois lados.

    Anuncio sem assinatura utilizavel vira grupo proprio (a url), em vez de se
    juntar a um grupo gigante de nulos.
    """
    partes = [
        df["preco_venda"].round(-3).astype("string"),
        df["area_util"].round(0).astype("string"),
        df["quartos"].astype("string"),
        df["banheiros"].astype("string"),
        df["garagens"].astype("string"),
    ]
    bruta = partes[0].str.cat(partes[1:], sep="|", na_rep="?")

    utilizavel = df["preco_venda"].notna() & df["area_util"].notna()
    return bruta.where(utilizavel, df[IDENTIFICADOR]).astype(str)


def carregar() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Devolve X, y (em log) e os grupos do split.

    O alvo vai para log porque a distribuicao e fortemente assimetrica
    (assimetria 5,92, curtose 80,4): em reais, um imovel de R$ 19,8 milhoes
    domina o erro quadratico de 16 mil observacoes.
    """
    if not MATRIZ.exists():
        raise FileNotFoundError(
            f"'{MATRIZ}' nao existe. Rode antes: python -m imoveis_jp.features.build_features"
        )

    df = pd.read_csv(MATRIZ, low_memory=False)
    df = df[df[ALVO].notna() & (df[ALVO] > 0)].copy()

    grupos = assinatura_imovel(df)
    y = pd.Series(np.log(df[ALVO].to_numpy()), index=df.index, name="log_preco")
    X = df.drop(columns=[ALVO, IDENTIFICADOR])

    return X, y, grupos


def colunas_por_tipo(X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Separa em contínuas, binárias já em 0/1 e nominais em texto.

    As nominais chegam como texto de propósito: codificá-las em build_features
    definiria o conjunto de colunas usando as linhas que virariam teste.
    """
    numericas = [c for c in NUMERICAS if c in X.columns]
    categoricas = [c for c in CATEGORICAS if c in X.columns]
    binarias = [c for c in X.columns if c not in numericas and c not in categoricas]
    return numericas, binarias, categoricas


def dividir(
    X: pd.DataFrame, y: pd.Series, grupos: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split 80/20 agrupado por imovel fisico, com semente fixa."""
    divisor = GroupShuffleSplit(n_splits=1, test_size=PROPORCAO_TESTE, random_state=SEMENTE)
    idx_treino, idx_teste = next(divisor.split(X, y, groups=grupos))

    return (
        X.iloc[idx_treino],
        X.iloc[idx_teste],
        y.iloc[idx_treino],
        y.iloc[idx_teste],
        grupos.iloc[idx_treino],
        grupos.iloc[idx_teste],
    )


def diagnostico_split(grupos_treino: pd.Series, grupos_teste: pd.Series) -> int:
    """Quantos grupos aparecem dos dois lados. Tem que ser zero."""
    return len(set(grupos_treino) & set(grupos_teste))
