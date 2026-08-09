# -*- coding: utf-8 -*-
"""Testes da preparação para modelagem.

O que estes testes protegem é a validade da métrica: se o mesmo imóvel cair no
treino e no teste, o modelo memoriza e o número reportado deixa de significar
generalização.
"""

import numpy as np
import pandas as pd

from imoveis_jp.models import dataset


def _base_sintetica() -> pd.DataFrame:
    """Seis anúncios, três deles o mesmo apartamento repetido."""
    return pd.DataFrame(
        {
            "url_anuncio": [f"u{i}" for i in range(6)],
            "preco_venda": [500_000.0, 500_000.0, 500_000.0, 900_000.0, 300_000.0, 750_000.0],
            "area_util": [70.0, 70.0, 70.0, 120.0, 45.0, 95.0],
            "quartos": [3.0, 3.0, 3.0, 4.0, 2.0, 3.0],
            "banheiros": [2.0, 2.0, 2.0, 3.0, 1.0, 2.0],
            "garagens": [1.0, 1.0, 1.0, 2.0, 1.0, 2.0],
        }
    )


def test_assinatura_agrupa_o_mesmo_imovel():
    assinaturas = dataset.assinatura_imovel(_base_sintetica())

    assert assinaturas.iloc[0] == assinaturas.iloc[1] == assinaturas.iloc[2]
    assert assinaturas.nunique() == 4


def test_assinatura_sem_dados_utilizaveis_vira_grupo_proprio():
    # sem preço e sem área, dois anúncios não podem ser declarados o mesmo imóvel
    df = pd.DataFrame(
        {
            "url_anuncio": ["a", "b"],
            "preco_venda": [np.nan, np.nan],
            "area_util": [np.nan, np.nan],
            "quartos": [2.0, 2.0],
            "banheiros": [1.0, 1.0],
            "garagens": [1.0, 1.0],
        }
    )
    assinaturas = dataset.assinatura_imovel(df)

    assert assinaturas.tolist() == ["a", "b"]


def test_split_nao_deixa_o_mesmo_imovel_dos_dois_lados():
    X, y, grupos = dataset.carregar()
    _, _, _, _, g_tr, g_te = dataset.dividir(X, y, grupos)

    assert dataset.diagnostico_split(g_tr, g_te) == 0


def test_split_e_reproduzivel_com_a_semente():
    X, y, grupos = dataset.carregar()
    primeiro = dataset.dividir(X, y, grupos)[0].index.tolist()
    segundo = dataset.dividir(X, y, grupos)[0].index.tolist()

    assert primeiro == segundo


def test_alvo_e_identificador_ficam_fora_das_features():
    X, _, _ = dataset.carregar()

    assert dataset.ALVO not in X.columns
    assert dataset.IDENTIFICADOR not in X.columns


def test_matriz_nao_contem_features_que_vazam_o_alvo():
    X, _, _ = dataset.carregar()

    # bairro_preco_m2_medio era agregação do próprio alvo (r=0,996 com a
    # mediana de preço/m² desta base); anunciante_qtd_anuncios era contado
    # sobre treino + teste juntos
    assert "bairro_preco_m2_medio" not in X.columns
    assert "anunciante_qtd_anuncios" not in X.columns


def test_alvo_vai_para_log():
    X, y, _ = dataset.carregar()

    assert y.name == "log_preco"
    # em reais a assimetria é 5,92; em log tem que cair para perto de zero
    assert abs(pd.Series(y).skew()) < 1.0


def test_nominais_chegam_como_texto_nao_como_dummies():
    """O one-hot tem que acontecer dentro do Pipeline, não em build_features.

    Enquanto era `pd.get_dummies` sobre a base inteira, o conjunto de colunas
    era definido usando as linhas que virariam teste.
    """
    X, _, _ = dataset.carregar()
    _, _, categoricas = dataset.colunas_por_tipo(X)

    assert "bairro" in categoricas
    assert X["bairro"].dtype == object or X["bairro"].dtype == "str"
    assert not [c for c in X.columns if c.startswith("bairro_")]


def test_encoder_e_ajustado_so_no_treino():
    """Categoria que só existe no teste não pode virar coluna nova."""
    from imoveis_jp.models import train

    treino = pd.DataFrame({"bairro": ["manaira"] * 40 + ["bessa"] * 40})
    teste = pd.DataFrame({"bairro": ["manaira", "bairro_que_nunca_vimos"]})

    preparo = train.montar_preprocessador([], [], ["bairro"])
    largura_treino = preparo.fit_transform(treino).shape[1]
    transformado = preparo.transform(teste)

    assert transformado.shape[1] == largura_treino  # sem coluna nova
    assert transformado[1].sum() == 0.0  # a categoria inédita não vira 1 em nada


def test_categoria_rara_e_agrupada_contando_so_o_treino():
    # substitui o corte de bairros com <30 imóveis, que antes contava sobre a
    # base inteira em build_features
    from imoveis_jp.models import train

    treino = pd.DataFrame({"bairro": ["comum"] * 100 + ["raro"] * 5})
    preparo = train.montar_preprocessador([], [], ["bairro"])
    preparo.fit(treino)

    categorias = preparo.named_transformers_["cat"].infrequent_categories_[0]
    assert "raro" in categorias
    assert "comum" not in categorias
