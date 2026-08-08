# -*- coding: utf-8 -*-
"""Testes da busca de hiperparâmetros.

O que estes testes protegem é a validade metodológica da busca, não o resultado
dela: a busca precisa usar o mesmo agrupamento do split (senão o mesmo imóvel
aparece na fold de treino e na de validação) e precisa falhar alto em vez de
devolver `nan` silenciosamente.
"""

from sklearn.model_selection import GroupKFold

from imoveis_jp.models import dataset, train, tune


def _buscas():
    return tune.montar_buscas(["area_util"], ["com_piscina"], ["bairro"])


def test_busca_usa_validacao_cruzada_agrupada():
    # KFold simples deixaria o mesmo apartamento nos dois lados da fold
    for busca in _buscas().values():
        assert isinstance(busca.cv, GroupKFold)
        assert busca.cv.n_splits == train.FOLDS


def test_busca_nao_engole_falha_como_nan():
    for busca in _buscas().values():
        assert busca.error_score == "raise"


def test_busca_otimiza_o_erro_absoluto():
    # MAE e não RMSE: com o alvo em log e cauda longa, o quadrático faria a
    # busca perseguir os poucos imóveis de altíssimo padrão
    for busca in _buscas().values():
        assert busca.scoring == "neg_mean_absolute_error"


def test_grades_referem_o_passo_do_regressor_no_pipeline():
    # sem o prefixo 'regressor__' o GridSearchCV não encontra o parâmetro
    for grade in (tune.GRADE_BOOSTING, tune.GRADE_RIDGE):
        assert all(chave.startswith("regressor__") for chave in grade)


def test_grades_incluem_o_default_do_sklearn():
    # o ganho da busca só é interpretável se a configuração padrão estiver
    # dentro da grade, para servir de referência
    assert 0.1 in tune.GRADE_BOOSTING["regressor__learning_rate"]
    assert 31 in tune.GRADE_BOOSTING["regressor__max_leaf_nodes"]
    assert 20 in tune.GRADE_BOOSTING["regressor__min_samples_leaf"]
    assert 0.0 in tune.GRADE_BOOSTING["regressor__l2_regularization"]
    assert 1.0 in tune.GRADE_RIDGE["regressor__alpha"]


def test_busca_reajusta_o_melhor_modelo():
    # refit=True para que .predict() no teste use a melhor configuração
    for busca in _buscas().values():
        assert busca.refit is True


def test_semente_e_a_mesma_do_split():
    # a busca precisa cair exatamente sobre o mesmo treino do train.py
    assert dataset.SEMENTE == 42
