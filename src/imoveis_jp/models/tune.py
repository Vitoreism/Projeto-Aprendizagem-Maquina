# -*- coding: utf-8 -*-
"""
busca de hiperparametros para os modelos de preco

o train.py compara os modelos em configuracao padrao. este modulo procura a
melhor configuracao de cada um, sempre dentro do conjunto de treino: a busca usa
o mesmo GroupKFold do train.py, entao nenhuma fold de validacao contem um imovel
que aparece na fold de treino, e o conjunto de teste continua intocado ate a
avaliacao final.

VIES DE SELECAO -- por que o score da busca nao e a metrica reportavel:

escolher a melhor de 48 configuracoes pelo score da validacao cruzada torna esse
score otimista: parte da vantagem da vencedora e sorte de particao, nao qualidade.
o jeito rigoroso de estimar isso e validacao cruzada aninhada, com um laco externo
so para medir. nao foi usada aqui porque custaria 5x o tempo para responder uma
pergunta que o conjunto de teste ja responde -- ele foi separado antes da busca e
nao participou de nenhuma decisao. entao: o score da busca serve para ESCOLHER, o
score do teste serve para RELATAR.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

from imoveis_jp import config
from imoveis_jp.models import dataset, train

SAIDA_MELHORES = config.PROCESSED / "melhores_hiperparametros.json"
SAIDA_BUSCA = config.INTERIM / "busca_hiperparametros.csv"

#: grade do gradient boosting, refinada a partir de uma primeira passada.
#:
#: a passada inicial usou max_leaf_nodes [15, 31, 63] e max_iter [200, 500], e o
#: MAE medio por valor mostrou duas coisas:
#:
#:   max_leaf_nodes  15 -> 0,2346   31 -> 0,2291   63 -> 0,2262
#:   max_iter       200 -> 0,2313  500 -> 0,2286
#:
#: max_leaf_nodes melhorava monotonicamente ATE A BORDA da grade, sinal classico
#: de que o otimo estava fora dela -- por isso a faixa foi estendida ate 255.
#: ja max_iter saturou: as duas melhores configuracoes com 200 e com 500 ficaram
#: a 0,00002 uma da outra, porque early_stopping='auto' liga sozinho acima de
#: 10.000 amostras (temos 12.820) e o modelo para antes do teto. max_iter virou
#: valor fixo, so como limite superior.
GRADE_BOOSTING: Dict[str, List[Any]] = {
    # 0,1 e o default e venceu 0,05 na primeira passada (0,2288 contra 0,2311),
    # mas a diferenca e pequena e a taxa interage com a profundidade, entao as
    # duas continuam na grade.
    "regressor__learning_rate": [0.05, 0.1],
    # teto, nao alvo: ver o comentario sobre early stopping acima.
    "regressor__max_iter": [500],
    # complexidade por arvore, o unico eixo que mexeu de verdade no resultado.
    # 31 e o default; 127 e 255 entraram porque 63 estava na borda melhorando.
    "regressor__max_leaf_nodes": [31, 63, 127, 255],
    # preco de imovel tem cauda longa: folha maior impede a arvore de isolar
    # um unico anuncio de alto padrao e decorar o valor dele.
    "regressor__min_samples_leaf": [20, 50],
    # o default e 0, ou seja, sem regularizacao L2.
    "regressor__l2_regularization": [0.0, 1.0],
}

#: varredura logaritmica ampla: com 99 features padronizadas, o alpha util pode
#: estar a ordens de magnitude do default 1,0, e passos menores nao mudariam a
#: escolha.
GRADE_RIDGE: Dict[str, List[Any]] = {
    "regressor__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
}


def montar_buscas(
    numericas: List[str], binarias: List[str], categoricas: List[str]
) -> Dict[str, GridSearchCV]:
    def com_preparo(regressor) -> Pipeline:
        return Pipeline(
            [
                ("preparo", train.montar_preprocessador(numericas, binarias, categoricas)),
                ("regressor", regressor),
            ]
        )

    def busca(regressor, grade: Dict[str, List[Any]]) -> GridSearchCV:
        return GridSearchCV(
            com_preparo(regressor),
            param_grid=grade,
            cv=GroupKFold(n_splits=train.FOLDS),
            scoring="neg_mean_absolute_error",
            # em serie pelo mesmo motivo do train.py: o HistGradientBoosting ja
            # e multi-thread por OpenMP e paralelizar por cima sobrecarrega a
            # maquina, fazendo um worker morrer e o score virar nan.
            n_jobs=None,
            error_score="raise",
            refit=True,
        )

    return {
        "ridge": busca(Ridge(random_state=dataset.SEMENTE), GRADE_RIDGE),
        "gradient_boosting": busca(
            HistGradientBoostingRegressor(random_state=dataset.SEMENTE), GRADE_BOOSTING
        ),
    }


def executar() -> pd.DataFrame:
    config.ensure_dirs()

    X, y, grupos = dataset.carregar()
    X_tr, X_te, y_tr, y_te, g_tr, g_te = dataset.dividir(X, y, grupos)
    if dataset.diagnostico_split(g_tr, g_te):
        raise RuntimeError("split vazou: o mesmo imovel esta no treino e no teste")

    print(f"[Dados] treino={len(X_tr)} teste={len(X_te)} semente={dataset.SEMENTE}", flush=True)

    numericas, binarias, categoricas = dataset.colunas_por_tipo(X)
    buscas = montar_buscas(numericas, binarias, categoricas)

    linhas = []
    melhores: Dict[str, Dict[str, Any]] = {}
    tabelas = []

    for nome, busca in buscas.items():
        total = int(np.prod([len(v) for v in busca.param_grid.values()]))
        print(f"\n[{nome}] {total} configuracoes x {train.FOLDS} folds...", flush=True)

        busca.fit(X_tr, y_tr, groups=g_tr)

        mae_padrao = _mae_da_configuracao_padrao(nome, busca)
        metricas = train.metricas_em_reais(y_te.to_numpy(), busca.predict(X_te))

        parametros = {k.replace("regressor__", ""): v for k, v in busca.best_params_.items()}
        melhores[nome] = parametros

        print(f"[{nome}] melhor CV MAE(log)={-busca.best_score_:.4f}", flush=True)
        print(f"[{nome}] parametros: {parametros}", flush=True)
        print(
            f"[{nome}] teste MAE=R$ {metricas['mae_reais']:,.0f} | "
            f"erro mediano={metricas['erro_percentual_mediano']:.1f}% | "
            f"R2(log)={metricas['r2_log']:.3f}",
            flush=True,
        )

        linhas.append(
            {
                "modelo": nome,
                "configuracoes": total,
                "cv_mae_log_padrao": mae_padrao,
                "cv_mae_log_melhor": -busca.best_score_,
                **parametros,
                **metricas,
            }
        )

        tabela = pd.DataFrame(busca.cv_results_)
        tabela.insert(0, "modelo", nome)
        tabelas.append(tabela)

    pd.concat(tabelas, ignore_index=True).to_csv(SAIDA_BUSCA, index=False, encoding="utf-8")

    with open(SAIDA_MELHORES, "w", encoding="utf-8") as f:
        json.dump(
            {
                "semente": dataset.SEMENTE,
                "folds": train.FOLDS,
                "busca": "GridSearchCV com GroupKFold sobre o treino; teste intocado",
                "melhores": melhores,
                "resultados": linhas,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    resultados = pd.DataFrame(linhas)
    print("\n" + "=" * 72)
    print("BUSCA CONCLUIDA")
    print("=" * 72)
    for l in linhas:
        ganho = l["cv_mae_log_padrao"] - l["cv_mae_log_melhor"]
        relativo = ganho / l["cv_mae_log_padrao"] if l["cv_mae_log_padrao"] else 0.0
        print(
            f"{l['modelo']:18s} CV {l['cv_mae_log_padrao']:.4f} -> {l['cv_mae_log_melhor']:.4f} "
            f"({relativo:+.1%}) | teste MAE=R$ {l['mae_reais']:,.0f}"
        )
    print("=" * 72)
    print(f"Melhores parametros: {SAIDA_MELHORES}")
    print(f"Grade completa:      {SAIDA_BUSCA}")

    return resultados


def _mae_da_configuracao_padrao(nome: str, busca: GridSearchCV) -> float:
    """MAE da configuracao default, se ela estiver na grade.

    Serve de referencia honesta para o ganho: comparar o melhor da busca com o
    numero do train.py seria comparar execucoes diferentes.
    """
    padroes = {
        "ridge": {"regressor__alpha": 1.0},
        "gradient_boosting": {
            "regressor__learning_rate": 0.1,
            "regressor__max_leaf_nodes": 31,
            "regressor__min_samples_leaf": 20,
            "regressor__l2_regularization": 0.0,
        },
    }
    alvo = padroes.get(nome, {})
    for parametros, score in zip(busca.cv_results_["params"], busca.cv_results_["mean_test_score"]):
        if all(parametros.get(k) == v for k, v in alvo.items()):
            return float(-score)
    return float("nan")


if __name__ == "__main__":
    executar()
