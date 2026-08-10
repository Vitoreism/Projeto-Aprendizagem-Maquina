# -*- coding: utf-8 -*-
"""Testes do registro de candidatos.

O ponto destes testes e o momento em que eles falham. Um candidato quebrado
descoberto no dia da consolidacao custa a Fase 2 inteira; descoberto aqui,
custa dois segundos. Por isso as regras do protocolo que dao para verificar
sozinhas estao escritas como assercao, e nao como recomendacao no README.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from imoveis_jp.models import candidatos, dataset, train
from imoveis_jp.models.candidatos.base import Candidato

INSCRITOS = candidatos.descobrir()


def _base_sintetica(n=60):
    """Pequena, mas com a mesma FORMA da real: continua, binaria e nominal.

    Inclui de proposito um nulo e uma categoria rara, que sao os dois casos que
    quebram modelo sem imputacao ou sem tratamento de categoria inedita.
    """
    rng = np.random.default_rng(dataset.SEMENTE)
    X = pd.DataFrame(
        {
            "area_util": rng.uniform(40, 200, n),
            "quartos": rng.integers(1, 5, n).astype(float),
            "com_piscina": rng.integers(0, 2, n),
            "venda_direta": rng.integers(0, 2, n),
            "bairro": rng.choice(["bessa", "manaira", "gramame"], n),
        }
    )
    X.loc[0, "area_util"] = np.nan
    X.loc[1, "bairro"] = "categoria_rara"
    y = pd.Series(np.log(X["area_util"].fillna(80) * 8000 + 50_000))
    return X, y


def test_ha_candidatos_inscritos():
    assert INSCRITOS, "nenhum candidato descoberto -- o pkgutil nao esta achando os modulos"


def test_ridge_e_boosting_migraram():
    """A prova de que o formato funciona para os dois modelos que ja existiam."""
    assert "ridge" in INSCRITOS
    assert "gradient_boosting_ajustado" in INSCRITOS


@pytest.mark.parametrize("nome", sorted(INSCRITOS))
def test_hipotese_nao_e_vazia(nome):
    """A regra 'hipotese antes do resultado' virando codigo.

    Dois dos achados mais uteis do projeto foram previsoes erradas. Nenhuma
    delas apareceria se a hipotese pudesse ser escrita depois de rodar.
    """
    hipotese = INSCRITOS[nome].hipotese.strip()
    assert hipotese
    # uma frase de tres palavras nao e hipotese, e formalidade preenchida
    assert len(hipotese) >= 40, f"'{nome}': hipotese curta demais para dizer algo"


@pytest.mark.parametrize("nome", sorted(INSCRITOS))
def test_grade_tem_o_prefixo_do_pipeline(nome):
    """Sem 'regressor__' o GridSearchCV nao acha o parametro -- e nao avisa."""
    for chave in INSCRITOS[nome].grade:
        assert chave.startswith("regressor__"), f"'{nome}': chave '{chave}'"


@pytest.mark.parametrize("nome", sorted(INSCRITOS))
def test_candidato_e_reprodutivel(nome):
    """Modelo estocastico sem semente torna a comparacao irrepetivel.

    Arvore e MLP sorteiam -- inicializacao de pesos, empate de split, fatia do
    early stopping. Sem `random_state` fixo, dois devs rodando o mesmo candidato
    obtem numeros diferentes, e a Fase 2 nao teria como saber se a diferenca
    entre dois modelos e o modelo ou o sorteio.

    A verificacao e condicional porque nem todo estimador e estocastico: Ridge
    com solver fechado nao tem o parametro, e exigi-lo seria exigir o
    impossivel.
    """
    regressor = INSCRITOS[nome].regressor
    if "random_state" not in regressor.get_params():
        return
    assert regressor.get_params()["random_state"] == dataset.SEMENTE, (
        f"'{nome}': random_state={regressor.get_params()['random_state']}, "
        f"esperado dataset.SEMENTE ({dataset.SEMENTE})"
    )


@pytest.mark.parametrize("nome", sorted(INSCRITOS))
def test_escala_declarada_vem_com_motivo(nome):
    """Ligar a opcao sem dizer por que e o mesmo que nao ter decidido."""
    candidato = INSCRITOS[nome]
    if candidato.escalar_binarias:
        assert candidato.justificativa_escala.strip(), f"'{nome}'"


@pytest.mark.parametrize("nome", sorted(INSCRITOS))
def test_candidato_treina_e_preve(nome):
    """O teste que economiza o dia da consolidacao.

    Roda o candidato dentro do Pipeline real -- mesmo pre-processamento, mesma
    forma de dado. Se ele nao aceita nulo, categoria de texto ou o formato das
    colunas, quebra aqui.
    """
    candidato = INSCRITOS[nome]
    X, y = _base_sintetica()
    num, bin_, cat = dataset.colunas_por_tipo(X)

    modelo = Pipeline(
        [
            ("preparo", train.montar_preprocessador(num, bin_, cat, candidato.escalar_binarias)),
            ("regressor", candidato.regressor),
        ]
    )
    modelo.fit(X, y)
    previsto = modelo.predict(X)

    assert len(previsto) == len(y)
    assert np.isfinite(previsto).all(), f"'{nome}' previu nao-finito"


@pytest.mark.parametrize("nome", sorted(INSCRITOS))
def test_grade_e_compativel_com_o_regressor(nome):
    """Parametro que o estimador nao tem so estoura depois de horas de busca."""
    candidato = INSCRITOS[nome]
    aceitos = candidato.regressor.get_params()
    for chave in candidato.grade:
        parametro = chave.replace("regressor__", "")
        assert parametro in aceitos, (
            f"'{nome}': {type(candidato.regressor).__name__} nao aceita "
            f"'{parametro}'"
        )


def test_todos_entram_no_treino():
    """O registro tem que chegar em montar_modelos, senao nao serve para nada."""
    X, _ = _base_sintetica()
    num, bin_, cat = dataset.colunas_por_tipo(X)
    modelos = train.montar_modelos(num, bin_, cat)

    for nome in INSCRITOS:
        assert nome in modelos, f"'{nome}' foi descoberto mas nao entrou no treino"
    # as duas referencias continuam la
    assert "baseline_mediana" in modelos
    assert "gradient_boosting" in modelos


# --------------------------------------------------------------------------
# o contrato do dataclass, testado pelo que ele REJEITA
# --------------------------------------------------------------------------


def test_recusa_candidato_sem_hipotese():
    with pytest.raises(ValueError, match="sem hipotese"):
        Candidato(nome="x", dono="y", regressor=BaseEstimator(), hipotese="   ")


def test_recusa_grade_sem_prefixo():
    with pytest.raises(ValueError, match="prefixo"):
        Candidato(
            nome="x",
            dono="y",
            regressor=BaseEstimator(),
            hipotese="uma hipotese suficientemente longa para o teste passar",
            grade={"alpha": [1.0]},
        )


def test_recusa_escala_sem_justificativa():
    with pytest.raises(ValueError, match="sem justificar"):
        Candidato(
            nome="x",
            dono="y",
            regressor=BaseEstimator(),
            hipotese="uma hipotese suficientemente longa para o teste passar",
            escalar_binarias=True,
        )


# --------------------------------------------------------------------------
# escalar_binarias
# --------------------------------------------------------------------------


def test_escalar_binarias_padroniza_o_bloco_binario():
    """Com a opcao ligada, binaria e continua saem na mesma escala.

    E o problema que a opcao existe para resolver: uma binaria com p=0,1 tem
    variancia 0,09 contra 1 das continuas, e para KNN e MLP isso faz as
    continuas dominarem a distancia.
    """
    X, y = _base_sintetica(200)
    num, bin_, cat = dataset.colunas_por_tipo(X)

    sem = train.montar_preprocessador(num, bin_, cat, escalar_binarias=False)
    com = train.montar_preprocessador(num, bin_, cat, escalar_binarias=True)

    saida_sem = pd.DataFrame(sem.fit_transform(X), columns=sem.get_feature_names_out())
    saida_com = pd.DataFrame(com.fit_transform(X), columns=com.get_feature_names_out())

    binaria = "bin__com_piscina"
    assert saida_sem[binaria].isin([0, 1]).all()          # passthrough
    assert abs(saida_com[binaria].std(ddof=0) - 1.0) < 1e-6  # padronizada


def test_o_default_nao_muda_o_pre_processamento_existente():
    """escalar_binarias=False tem que ser identico ao que havia antes.

    Garante que introduzir a opcao nao mexeu nos numeros ja relatados.
    """
    X, _ = _base_sintetica(120)
    num, bin_, cat = dataset.colunas_por_tipo(X)

    p = train.montar_preprocessador(num, bin_, cat)
    bloco = dict(zip([n for n, _, _ in p.transformers], p.transformers))
    assert bloco["bin"][1] == "passthrough"
