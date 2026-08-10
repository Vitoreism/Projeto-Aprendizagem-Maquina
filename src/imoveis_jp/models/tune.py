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
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

from imoveis_jp import config
from imoveis_jp.models import candidatos, dataset, train

SAIDA_MELHORES = config.PROCESSED / "melhores_hiperparametros.json"
SAIDA_BUSCA = config.INTERIM / "busca_hiperparametros.csv"

#: As grades migraram para `candidatos/*.py`, junto do modelo a que pertencem.
#: Elas nao sao configuracao deste modulo: sao parte da declaracao do candidato,
#: e mante-las aqui obrigaria cinco pessoas a editar este arquivo.


def montar_buscas(
    numericas: List[str], binarias: List[str], categoricas: List[str]
) -> Dict[str, GridSearchCV]:
    """Uma busca por candidato inscrito que tenha grade.

    Candidato com `grade` vazia e pulado sem reclamar: nem todo modelo tem
    hiperparametro que valha buscar, e o vazio ali significa "sem busca", nao
    "esqueci" -- quem esqueceu nao passa no teste do registro.
    """

    def com_preparo(regressor, escalar_binarias: bool) -> Pipeline:
        return Pipeline(
            [
                (
                    "preparo",
                    train.montar_preprocessador(
                        numericas, binarias, categoricas, escalar_binarias
                    ),
                ),
                ("regressor", regressor),
            ]
        )

    buscas = {}
    for nome, candidato in candidatos.descobrir().items():
        if not candidato.grade:
            print(f"[{nome}] sem grade declarada; sem busca.", flush=True)
            continue

        buscas[nome] = GridSearchCV(
            com_preparo(candidato.regressor, candidato.escalar_binarias),
            param_grid=candidato.grade,
            cv=GroupKFold(n_splits=train.FOLDS),
            scoring="neg_mean_absolute_error",
            # em serie pelo mesmo motivo do train.py: o HistGradientBoosting ja
            # e multi-thread por OpenMP e paralelizar por cima sobrecarrega a
            # maquina, fazendo um worker morrer e o score virar nan.
            n_jobs=None,
            error_score="raise",
            refit=True,
        )

    return buscas


def executar() -> pd.DataFrame:
    config.ensure_dirs()

    X, y, grupos = dataset.carregar()
    X_tr, X_te, y_tr, y_te, g_tr, g_te = dataset.dividir(X, y, grupos)
    if dataset.diagnostico_split(g_tr, g_te):
        raise RuntimeError("split vazou: o mesmo imovel esta no treino e no teste")

    print(f"[Dados] treino={len(X_tr)} teste={len(X_te)} semente={dataset.SEMENTE}", flush=True)

    numericas, binarias, categoricas = dataset.colunas_por_tipo(X)
    buscas = montar_buscas(numericas, binarias, categoricas)
    inscritos = candidatos.descobrir()

    linhas = []
    melhores: Dict[str, Dict[str, Any]] = {}
    tabelas = []

    for nome, busca in buscas.items():
        total = int(np.prod([len(v) for v in busca.param_grid.values()]))
        print(f"\n[{nome}] {total} configuracoes x {train.FOLDS} folds...", flush=True)

        busca.fit(X_tr, y_tr, groups=g_tr)

        mae_padrao = _mae_da_configuracao_padrao(inscritos[nome], busca)
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


def _mae_da_configuracao_padrao(candidato: candidatos.Candidato, busca: GridSearchCV) -> float:
    """MAE da configuracao default do estimador, se ela estiver na grade.

    Serve de referencia honesta para o ganho: comparar o melhor da busca com o
    numero do train.py seria comparar execucoes diferentes.

    Os defaults sao lidos do proprio sklearn, instanciando a classe sem
    argumento. A versao anterior mantinha um dicionario com os defaults escritos
    a mao por nome de modelo -- que quebra em silencio de duas formas: quando o
    sklearn muda um default entre versoes, e quando alguem inscreve um candidato
    novo e esquece de adiciona-lo ali. Nos dois casos a funcao devolvia nan e o
    ganho ficava em branco no relatorio sem ninguem entender por que.
    """
    padrao = type(candidato.regressor)()
    alvo = {
        chave: getattr(padrao, chave.replace("regressor__", ""))
        for chave in candidato.grade
        if hasattr(padrao, chave.replace("regressor__", ""))
    }
    if not alvo:
        return float("nan")

    for parametros, score in zip(busca.cv_results_["params"], busca.cv_results_["mean_test_score"]):
        if all(parametros.get(k) == v for k, v in alvo.items()):
            return float(-score)
    return float("nan")


if __name__ == "__main__":
    executar()
