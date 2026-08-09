# -*- coding: utf-8 -*-
"""Testes da analise de residuos e da importancia por permutacao.

O que estes testes protegem e a leitura do resultado: sinal de residuo trocado
faz "o modelo pede caro demais" virar o contrario, e importancia medida no
treino responderia uma pergunta diferente da que o relatorio afirma responder.
"""

import numpy as np
import pandas as pd
import pytest

from imoveis_jp.models import analysis


def _tabela_falsa(n: int = 200) -> pd.DataFrame:
    """Residuos ja calculados, para testar so a agregacao por segmento."""
    reais = np.linspace(100_000, 2_000_000, n)
    return pd.DataFrame(
        {
            "preco_real": reais,
            "erro_percentual": np.tile([10.0, -10.0, 30.0, -30.0], n // 4),
            "erro_absoluto_percentual": np.tile([10.0, 10.0, 30.0, 30.0], n // 4),
            "erro_reais": np.tile([1000.0, -1000.0, 3000.0, -3000.0], n // 4),
            "origem_anuncio": ["zapimoveis"] * (n // 2) + ["chaves_na_mao"] * (n // 2),
        }
    )


def test_residuo_positivo_significa_previsao_acima_do_real():
    """O sinal e o que torna a tabela legivel sem consultar a formula."""
    X = pd.DataFrame(
        {
            "bairro": ["manaira", "bessa"],
            "origem_anuncio": ["zapimoveis", "zapimoveis"],
            "tipo_unidade": ["apartamento_tipo", "apartamento_tipo"],
            "area_util": [70.0, 90.0],
        }
    )
    real = np.array([500_000.0, 500_000.0])
    y = pd.Series(np.log(real))
    # primeiro previsto acima do real, segundo abaixo
    previsto_log = np.log(np.array([600_000.0, 400_000.0]))

    tabela = analysis.tabela_residuos(X, y, previsto_log)

    assert tabela["residuo_log"].iloc[0] > 0
    assert tabela["erro_reais"].iloc[0] == pytest.approx(100_000.0)
    assert tabela["erro_percentual"].iloc[0] == pytest.approx(20.0)

    assert tabela["residuo_log"].iloc[1] < 0
    assert tabela["erro_percentual"].iloc[1] == pytest.approx(-20.0)
    # o absoluto nao distingue lado: e precisao, nao vies
    assert tabela["erro_absoluto_percentual"].iloc[1] == pytest.approx(20.0)


def test_faixa_de_preco_usa_o_real_e_nao_o_previsto():
    """Definir a faixa pelo previsto deixaria o erro escolher a propria faixa."""
    X = pd.DataFrame(
        {
            "bairro": ["b"] * 10,
            "origem_anuncio": ["z"] * 10,
            "tipo_unidade": ["t"] * 10,
            "area_util": np.linspace(40, 200, 10),
        }
    )
    real = np.linspace(100_000, 1_000_000, 10)
    y = pd.Series(np.log(real))
    # previsao invertida de proposito: se a faixa saisse do previsto, o imovel
    # mais barato cairia no quintil mais caro.
    previsto_log = pd.Series(np.log(real[::-1].copy()))

    tabela = analysis.tabela_residuos(X, y, previsto_log.to_numpy())
    mais_barato = tabela.sort_values("preco_real").iloc[0]

    assert str(mais_barato["faixa_preco"]).startswith("Q1")


def test_segmento_pequeno_demais_fica_de_fora():
    """A mediana de meia duzia de imoveis nao sustenta 'pior bairro'."""
    tabela = _tabela_falsa()
    tabela.loc[tabela.index[:5], "origem_anuncio"] = "portal_raro"

    resumo = analysis.resumo_por_segmento(tabela, "origem_anuncio")

    assert "portal_raro" not in resumo["categoria"].tolist()
    assert (resumo["n"] >= analysis.MINIMO_POR_SEGMENTO).all()


def test_resumo_separa_vies_de_precisao():
    tabela = _tabela_falsa()
    resumo = analysis.resumo_por_segmento(tabela, "origem_anuncio").set_index("categoria")

    # erros simetricos: o vies se cancela, a precisao nao
    assert resumo.loc["zapimoveis", "vies_percentual"] == pytest.approx(0.0, abs=1e-9)
    assert resumo.loc["zapimoveis", "erro_mediano_percentual"] == pytest.approx(20.0)


def test_importancia_marca_como_nula_o_que_nao_passa_do_ruido():
    class ResultadoFalso:
        importances_mean = np.array([0.20, 0.0001, -0.0002])
        importances_std = np.array([0.01, 0.0005, 0.0003])

    tabela = analysis._tabela_importancia(["forte", "fraca", "negativa"], ResultadoFalso())

    assert tabela.iloc[0]["feature"] == "forte"
    assert bool(tabela.set_index("feature").loc["forte", "significativa"])
    assert not bool(tabela.set_index("feature").loc["fraca", "significativa"])
    # importancia negativa e ruido: embaralhar a coluna melhorou o modelo
    assert not bool(tabela.set_index("feature").loc["negativa", "significativa"])


def test_analise_roda_sobre_o_conjunto_de_teste():
    """A afirmacao central do modulo: nada aqui e medido no treino."""
    fonte = analysis.executar.__doc__ or ""
    modulo = analysis.__doc__ or ""
    assert "TESTE" in modulo

    from imoveis_jp.models import dataset

    X, y, grupos = dataset.carregar()
    _, X_te, _, y_te, _, _ = dataset.dividir(X, y, grupos)

    # a analise precisa receber exatamente esse recorte, senao o numero
    # reportado nao e o mesmo do train.py
    modelo, _, X_analise, _, y_analise = analysis.ajustar()

    assert list(X_analise.index) == list(X_te.index)
    assert np.allclose(y_analise.to_numpy(), y_te.to_numpy())
    assert fonte is not None


def test_modelo_analisado_e_o_que_o_projeto_reporta():
    from imoveis_jp.models import train

    assert analysis.MODELO in train.montar_modelos(["area_util"], [], ["bairro"])
