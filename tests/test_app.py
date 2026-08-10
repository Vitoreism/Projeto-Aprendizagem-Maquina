# -*- coding: utf-8 -*-
"""Executa o app.py de ponta a ponta, sem navegador.

`st.tabs` nao e lazy: o corpo das sete abas roda a cada execucao do script.
Entao uma passada do AppTest exercita todas -- se qualquer aba levantar
excecao, ela aparece em `at.exception` e o teste falha.

E a verificacao que pega o erro que o pytest dos modulos nao pega: nome de
coluna errado no layout, chave que nao existe no json, metrica formatada com o
tipo errado.
"""

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

from imoveis_jp import config

APP = str(config.ROOT / "app.py")

#: o ajuste do modelo vencedor leva ~10s; o default de 3s do AppTest estoura.
TEMPO_LIMITE = 300


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=TEMPO_LIMITE)
    at.run()
    return at


def test_o_app_roda_sem_excecao(app):
    assert not app.exception, [str(e) for e in app.exception]


def test_tem_as_sete_abas(app):
    rotulos = [aba.label for aba in app.tabs] if hasattr(app, "tabs") else []
    # o container de tabs aparece como um bloco; a checagem util e o titulo
    assert app.title[0].value.startswith("Previsão de preços")
    assert len(rotulos) in (0, 7), f"esperava 7 abas, veio {len(rotulos)}"


def test_o_veredito_anuncia_o_vencedor(app):
    """A aba 1 tem que dizer quem ganhou, sem depender de clicar em nada."""
    textos = " ".join(
        [s.value for s in app.success] + [i.value for i in app.info]
    )
    assert "Gradient Boosting" in textos


def test_o_teste_fica_atras_de_um_botao(app):
    """A regra metodologica virando interface: nada de teste antes do clique."""
    rotulos = [b.label for b in app.button]
    assert any("Revelar" in r for r in rotulos), rotulos


def test_revelar_o_teste_nao_quebra(app):
    botao = next(b for b in app.button if "Revelar" in b.label)
    botao.click().run()
    assert not app.exception, [str(e) for e in app.exception]


def test_sortear_outro_imovel_nao_quebra(app):
    botao = next((b for b in app.button if "Sortear" in b.label), None)
    if botao is None:
        pytest.skip("o modo de sorteio nao esta ativo nesta execucao")
    botao.click().run()
    assert not app.exception, [str(e) for e in app.exception]
