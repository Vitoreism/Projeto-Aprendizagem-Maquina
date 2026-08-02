# -*- coding: utf-8 -*-
"""Testes da consolidação do one-hot e da seleção por correlação.

Vários destes travam bugs que já aconteceram na base: a colisão entre uma
comodidade e uma coluna numérica de mesmo nome, o NaN que significava "o outro
portal" e o casamento de sinônimo por substring pegando a palavra errada.
"""

import numpy as np
import pandas as pd

from imoveis_jp.features import build_features as bf
from imoveis_jp.features import correlation as corr


def test_canonizar_unifica_os_tres_geradores():
    # llm, html do chaves e html do zap descreviam a mesma piscina.
    assert bf.canonizar("piscina") == "piscina"
    assert bf.canonizar("comodidade_piscina") == "piscina"
    assert bf.canonizar("comodidade_piscina_privativa") == "piscina"


def test_canonizar_ignora_acento():
    assert bf.canonizar("comodidade_depósito") == bf.canonizar("comodidade_deposito")
    assert bf.canonizar("salão_de_festas") == bf.canonizar("comodidade_salao_de_festas")


def test_canonizar_nao_cai_nos_falsos_positivos_por_substring():
    # 'carpete' contém "pet" e 'supermercados' contém "mar".
    assert bf.canonizar("comodidade_carpete") == "carpete"
    assert bf.canonizar("supermercados") == "supermercados"


def test_colunas_binarias_nao_confunde_numerica_com_booleana():
    df = pd.DataFrame(
        {
            "piscina": [True, False, True],
            "andar": [0, 1, 0],  # 0/1 compara igual a False/True em pandas
            "quartos": [2.0, 3.0, 1.0],
        }
    )
    assert bf.colunas_binarias(df) == ["piscina"]


def test_consolidacao_nao_colide_com_coluna_numerica():
    df = pd.DataFrame(
        {
            "url_anuncio": ["a", "b"],
            "suites": [2.0, 1.0],
            "comodidade_suites": [True, False],
        }
    )
    resultado, _ = bf.consolidar_binarias(df)

    assert not resultado.columns.duplicated().any()
    assert resultado["suites"].tolist() == [2.0, 1.0]
    assert "com_suites" in resultado.columns


def test_consolidacao_funde_por_ou_e_trata_nan_como_ausencia():
    # nan aqui significa "o portal de origem não lista essa comodidade".
    df = pd.DataFrame(
        {
            "piscina": [True, False, False, False],
            "comodidade_piscina": [np.nan, True, np.nan, False],
        }
    )
    resultado, info = bf.consolidar_binarias(df)

    assert resultado["com_piscina"].tolist() == [1, 1, 0, 0]
    assert info["fundidas"]["piscina"] == ["comodidade_piscina", "piscina"]


def test_limites_anulam_valores_implausiveis():
    df = pd.DataFrame(
        {
            "preco_venda": [450_000.0, 470_000_000.0],
            "area_util": [70.0, 58_045_100.0],
            "garagens": [1.0, 155_560.0],
        }
    )
    resultado, fora = bf.aplicar_limites(df)

    assert resultado["preco_venda"].tolist()[0] == 450_000.0
    assert resultado["preco_venda"].isna().sum() == 1
    assert fora == {"preco_venda": 1, "area_util": 1, "garagens": 1}


def test_poda_mantem_a_feature_mais_ligada_ao_alvo():
    features = ["area_util", "area_total"]
    matriz = pd.DataFrame(
        [[1.0, 0.92], [0.92, 1.0]], index=features, columns=features
    )
    ranking = pd.DataFrame({"feature": features, "forca": [0.70, 0.63]})

    selecionadas, decisoes = corr.podar_redundantes(matriz, ranking, features)

    assert selecionadas == ["area_util"]
    assert decisoes.loc[0, "descartada"] == "area_total"


def test_poda_preserva_features_independentes():
    features = ["area_util", "com_piscina"]
    matriz = pd.DataFrame(
        [[1.0, 0.11], [0.11, 1.0]], index=features, columns=features
    )
    ranking = pd.DataFrame({"feature": features, "forca": [0.70, 0.18]})

    selecionadas, decisoes = corr.podar_redundantes(matriz, ranking, features)

    assert selecionadas == features
    assert decisoes.empty
