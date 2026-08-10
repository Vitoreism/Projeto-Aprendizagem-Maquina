# -*- coding: utf-8 -*-
"""Testes das duas regras que separam preco de venda de outras coisas.

Cada caso abaixo e um anuncio que existe na base, com os numeros dele. Sao dois
defeitos distintos que um corte unico confundiria:

  * o valor anunciado nao e o preco (agio de repasse)  -> descarta a linha
  * a area esta errada e o preco esta certo            -> anula a area
"""

import numpy as np
import pandas as pd
import pytest

from imoveis_jp.features import build_features as bf


def _base(linhas):
    return pd.DataFrame(linhas, columns=["preco_venda", "area_util"]).astype("float64")


# --------------------------------------------------------------------------
# regra 1: agio de repasse nao e preco de venda
# --------------------------------------------------------------------------


def test_agio_de_repasse_e_descartado():
    """R$ 22.000 por 55 m2 no Gramame e o valor das chaves, nao do imovel.

    O comprador ainda assume as parcelas do financiamento. Mantido na base, o
    anuncio ensina o modelo que 2 quartos no Gramame valem R$ 22.000 -- eram
    111 anuncios assim, e 84 deles diziam 'repasse' ou 'agio' no proprio texto.
    """
    df, info = bf.remover_precos_que_nao_sao_venda(_base([[22_000, 55]]))
    assert len(df) == 0
    assert info["descartados"] == 1


def test_venda_direta_da_caixa_sobrevive():
    """A populacao legitima mais barata da base, e ela fica logo acima do piso.

    311 anuncios de venda direta/leilao, com preco/m2 entre R$ 1.164 (p05) e
    R$ 1.899 (p75). Um piso em R$ 1.500 ou R$ 2.000 -- que a precisao contra o
    texto de repasse pareceria justificar -- apagaria metade ou todos eles.
    """
    # R$ 67.384 por 44 m2 em Paratibe = R$ 1.531/m2
    # R$ 74.147 por 47 m2 na Cuia     = R$ 1.578/m2
    df, info = bf.remover_precos_que_nao_sao_venda(_base([[67_384, 44], [74_147, 47]]))
    assert len(df) == 2
    assert info["descartados"] == 0


def test_anuncio_sem_area_nao_e_julgado():
    """Sem area nao ha preco/m2, e ausencia de prova nao e prova de defeito."""
    df, info = bf.remover_precos_que_nao_sao_venda(
        _base([[22_000, np.nan], [300_000, np.nan]])
    )
    assert len(df) == 2
    assert info["descartados"] == 0


def test_imovel_caro_nunca_e_tocado_pelo_piso():
    df, info = bf.remover_precos_que_nao_sao_venda(_base([[9_200_000, 321], [450_000, 70]]))
    assert len(df) == 2
    assert info["descartados"] == 0


# --------------------------------------------------------------------------
# regra 2: area com separador decimal perdido
# --------------------------------------------------------------------------


def test_area_com_virgula_perdida_e_anulada_e_o_preco_fica():
    """'137,20 m2' virou 1372 no scrap; o preco de R$ 1.250.000 esta correto.

    O outro caso do mesmo tipo tem o desmentido no proprio titulo: 988 m2 num
    anuncio que se chama 'Vista para o Mar em Manaira | 98m2'.
    """
    df, n = bf.corrigir_area_implausivel(_base([[1_250_000, 1372]]))
    assert n == 1
    assert pd.isna(df.loc[0, "area_util"])
    assert df.loc[0, "preco_venda"] == 1_250_000  # o preco sobrevive


def test_cobertura_gigante_de_verdade_sobrevive():
    """O contra-exemplo que proibe um teto absoluto de area.

    Coberturas de 664 m2 no Cabo Branco e 1.260 m2 no Miramar existem e custam
    R$ 11 e R$ 19,8 milhoes. Um teto em 400 ou 600 m2 mataria justamente os
    imoveis mais caros da base -- por isso a regra olha o par (area, preco/m2).
    """
    df, n = bf.corrigir_area_implausivel(_base([[11_338_626, 664], [19_800_000, 1260]]))
    assert n == 0
    assert df["area_util"].notna().all()


def test_area_pequena_com_preco_baixo_nao_vira_caso_de_area():
    """Repasse de 55 m2 e problema de PRECO; a area dele esta certa."""
    df, n = bf.corrigir_area_implausivel(_base([[22_000, 55]]))
    assert n == 0
    assert df.loc[0, "area_util"] == 55


# --------------------------------------------------------------------------
# as duas juntas, na ordem em que o pipeline as aplica
# --------------------------------------------------------------------------


def test_area_quebrada_nao_e_descartada_como_repasse():
    """A ordem importa: anular a area primeiro protege o anuncio do piso.

    R$ 550.000 por '1.280 m2' em Tambau da R$ 430/m2 -- abaixo do piso. Se o
    piso rodasse antes, a linha sairia da base levando um preco valido junto.
    Anulando a area primeiro, o anuncio perde o preco/m2 e deixa de ser
    candidato ao descarte.
    """
    df, n = bf.corrigir_area_implausivel(_base([[550_000, 1280]]))
    df, info = bf.remover_precos_que_nao_sao_venda(df)

    assert n == 1
    assert info["descartados"] == 0
    assert len(df) == 1
    assert df.iloc[0]["preco_venda"] == 550_000


def test_base_real_nao_tem_mais_preco_de_agio():
    """Varre a base inteira: nenhum preco/m2 abaixo do piso sobreviveu."""
    caminho = bf.SAIDA_CSV
    if not caminho.exists():
        pytest.skip("matriz de features ainda nao gerada")

    m = pd.read_csv(caminho, low_memory=False)
    pm = bf._preco_por_m2(m).dropna()

    assert pm.min() >= bf.PISO_PRECO_M2, f"preco/m2 minimo: R$ {pm.min():,.0f}"
