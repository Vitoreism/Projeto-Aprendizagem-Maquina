# -*- coding: utf-8 -*-
"""Testes da busca de hiperparâmetros.

O que estes testes protegem é a validade metodológica da busca, não o resultado
dela: a busca precisa usar o mesmo agrupamento do split (senão o mesmo imóvel
aparece na fold de treino e na de validação) e precisa falhar alto em vez de
devolver `nan` silenciosamente.
"""

from sklearn.model_selection import GroupKFold

from imoveis_jp.models import candidatos, dataset, train, tune


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


# A verificação do prefixo `regressor__` saiu daqui: as grades passaram a viver
# em `candidatos/*.py`, e o teste foi junto (`test_candidatos.py`), onde cobre
# todo candidato inscrito em vez de dois nomes fixos.


def test_grades_incluem_o_default_do_sklearn():
    """O ganho da busca só é interpretável se o default estiver dentro da grade.

    Vale para todo candidato, e não só para os dois que existiam quando este
    teste foi escrito — é `_mae_da_configuracao_padrao` que depende disso: ela
    lê os defaults do próprio sklearn e procura essa configuração nos
    resultados. Se a grade não a contém, o ganho sai `nan` no relatório sem
    ninguém entender por quê.

    A exigência vale só para eixo que está sendo *buscado*. Eixo com um valor
    único é constante declarada, não busca — e às vezes a constante é
    deliberadamente diferente do default: `max_iter=[500]` é teto e não alvo,
    porque o `early_stopping='auto'` para o modelo antes disso. Exigir o default
    ali proibiria uma decisão que está documentada e é correta.
    """
    for nome, candidato in candidatos.descobrir().items():
        padrao = type(candidato.regressor)()
        for chave, valores in candidato.grade.items():
            if len(valores) < 2:
                continue
            parametro = chave.replace("regressor__", "")
            if not hasattr(padrao, parametro):
                continue
            assert getattr(padrao, parametro) in valores, (
                f"'{nome}': o default de '{parametro}' "
                f"({getattr(padrao, parametro)}) não está na grade {valores}"
            )


def test_busca_reajusta_o_melhor_modelo():
    # refit=True para que .predict() no teste use a melhor configuração
    for busca in _buscas().values():
        assert busca.refit is True


def test_semente_e_a_mesma_do_split():
    # a busca precisa cair exatamente sobre o mesmo treino do train.py
    assert dataset.SEMENTE == 42
