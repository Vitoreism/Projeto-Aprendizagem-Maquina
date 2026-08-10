# -*- coding: utf-8 -*-
"""
treino e avaliacao dos modelos de preco

todo o pre-processamento que aprende dos dados (imputacao, escala) mora dentro
do Pipeline, para que o fit aconteca em cada fold da validacao cruzada e nenhuma
estatistica do fold de validacao vaze para o de treino. o conjunto de teste e
tocado uma unica vez, no fim.
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imoveis_jp import config
from imoveis_jp.models import candidatos, dataset

SAIDA_RESULTADOS = config.PROCESSED / "resultados_modelos.csv"
SAIDA_RELATORIO = config.INTERIM / "relatorio_treino.json"

FOLDS = 5


#: categoria com menos que isso vira 'infrequent' em vez de coluna propria.
#: substitui o antigo corte de bairros raros e o filtro de dummies com menos de
#: 1% -- ambos contavam sobre a base inteira, incluindo o teste. aqui a contagem
#: acontece dentro do fold de treino.
MINIMO_POR_CATEGORIA = 30


def montar_preprocessador(
    numericas: List[str],
    binarias: List[str],
    categoricas: List[str],
    escalar_binarias: bool = False,
) -> ColumnTransformer:
    """Imputacao, escala e one-hot -- todos ajustados dentro do fold.

    add_indicator preserva a informacao de ausencia: com iptu ausente em 80% dos
    anuncios, o proprio silencio do anunciante e sinal, nao ruido a ser apagado.

    O OneHotEncoder resolve de uma vez os dois vazamentos estruturais que a
    auditoria apontou: o vocabulario sai so do treino, e min_frequency agrupa as
    categorias raras contando so o treino. handle_unknown='infrequent_if_exist'
    manda categoria inedita no teste para o mesmo balde das raras, em vez de
    quebrar ou de virar uma coluna que o modelo nunca viu.

    ESCALAR_BINARIAS -- por que a opcao existe e por que o default e False.

    As continuas saem do StandardScaler com variancia 1. As binarias passam
    direto, e uma binaria com p=0,1 tem variancia 0,09. Para KNN e MLP isso faz
    as continuas dominarem a distancia e o gradiente; para arvore e Ridge e
    irrelevante (a arvore corta por limiar, e o limiar em 0/1 nao muda com a
    escala). Cada candidato declara o que precisa, em vez de existirem dois
    pre-processamentos concorrentes que ninguem sabe qual usar.

    Por que StandardScaler e nao MinMaxScaler: MinMax em dado que ja e 0/1 e
    literalmente identidade -- deixaria a binaria com variancia 0,09 e as
    continuas com 1, ou seja, nao resolveria nada. So a padronizacao iguala as
    variancias, que e o problema.

    UM DETALHE QUE VALE REGISTRAR: as colunas indicadoras de ausencia geradas
    pelo add_indicator tambem sao binarias, mas nascem DENTRO do bloco 'num' e
    ja passam pelo StandardScaler hoje, independentemente desta opcao. Ou seja,
    a base ja escalava parte das suas binarias e nao escalava a outra parte, sem
    que isso tivesse sido decidido. Com escalar_binarias=True o tratamento fica
    coerente; com False, a inconsistencia permanece -- e e inofensiva para os
    modelos que usam o default.
    """
    continuas = Pipeline(
        [
            ("imputa", SimpleImputer(strategy="median", add_indicator=True)),
            ("escala", StandardScaler()),
        ]
    )
    nominais = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=MINIMO_POR_CATEGORIA,
        sparse_output=False,
        dtype="float64",
    )
    return ColumnTransformer(
        [
            ("num", continuas, numericas),
            ("bin", StandardScaler() if escalar_binarias else "passthrough", binarias),
            ("cat", nominais, categoricas),
        ],
        remainder="drop",
    )


def montar_modelos(
    numericas: List[str], binarias: List[str], categoricas: List[str]
) -> Dict[str, Pipeline]:
    """Os modelos de referencia mais todos os candidatos inscritos.

    As duas referencias nao sao candidatas e por isso ficam aqui, fixas:

    - `baseline_mediana` e piso de sanidade, nao modelo. Nao tem hipotese a
      registrar nem grade a buscar -- ele existe para provar que a montagem
      esta correta (R2 tem que dar ~0).
    - `gradient_boosting` na configuracao padrao existe para medir o ganho do
      ajuste. Tirar do lugar apagaria a referencia do quanto a busca rendeu.

    Todo o resto vem de `candidatos.descobrir()`, cada um com o preparo que ele
    proprio declarou -- e sempre sobre as MESMAS colunas e o MESMO split, que e
    a regra 1 do protocolo.
    """

    def com_preparo(regressor, escalar_binarias: bool = False):
        return Pipeline(
            [
                (
                    "preparo",
                    montar_preprocessador(
                        numericas, binarias, categoricas, escalar_binarias
                    ),
                ),
                ("regressor", regressor),
            ]
        )

    modelos = {
        # piso absoluto: prever sempre a mediana. qualquer modelo tem que ganhar disto.
        "baseline_mediana": com_preparo(DummyRegressor(strategy="median")),
        # configuracao padrao, mantida como referencia do ganho do ajuste.
        "gradient_boosting": com_preparo(
            HistGradientBoostingRegressor(random_state=dataset.SEMENTE)
        ),
    }

    for nome, candidato in candidatos.descobrir().items():
        if nome in modelos:
            raise ValueError(
                f"O candidato '{nome}' colide com um modelo de referencia. "
                f"Escolha outro nome -- os dois iriam para a mesma linha do csv."
            )
        modelos[nome] = com_preparo(candidato.regressor, candidato.escalar_binarias)

    return modelos


def metricas_em_reais(y_log_real: np.ndarray, y_log_previsto: np.ndarray) -> Dict[str, float]:
    """O modelo treina em log; a metrica que interessa e em reais."""
    real = np.exp(y_log_real)
    previsto = np.exp(y_log_previsto)
    return {
        "mae_reais": float(mean_absolute_error(real, previsto)),
        "rmse_reais": float(root_mean_squared_error(real, previsto)),
        "mae_log": float(mean_absolute_error(y_log_real, y_log_previsto)),
        "r2_log": float(r2_score(y_log_real, y_log_previsto)),
        "erro_percentual_mediano": float(np.median(np.abs(previsto - real) / real) * 100),
    }


def executar() -> pd.DataFrame:
    config.ensure_dirs()

    X, y, grupos = dataset.carregar()
    print(f"[Dados] {len(X)} anuncios x {X.shape[1]} features.", flush=True)
    print(f"[Dados] {grupos.nunique()} imoveis fisicos distintos.", flush=True)

    X_tr, X_te, y_tr, y_te, g_tr, g_te = dataset.dividir(X, y, grupos)
    vazados = dataset.diagnostico_split(g_tr, g_te)
    print(
        f"[Split] treino={len(X_tr)} teste={len(X_te)} "
        f"(semente={dataset.SEMENTE}, agrupado por imovel).",
        flush=True,
    )
    print(f"[Split] grupos presentes nos dois lados: {vazados} (tem que ser 0).", flush=True)
    if vazados:
        raise RuntimeError("split vazou: o mesmo imovel esta no treino e no teste")

    numericas, binarias, categoricas = dataset.colunas_por_tipo(X)
    print(
        f"[Features] {len(numericas)} continuas, {len(binarias)} binarias, "
        f"{len(categoricas)} nominais (codificadas dentro do Pipeline).",
        flush=True,
    )
    modelos = montar_modelos(numericas, binarias, categoricas)

    linhas = []
    for nome, modelo in modelos.items():
        # a CV reajusta imputacao e escala dentro de cada fold.
        #
        # folds em serie de proposito: o HistGradientBoosting ja e multi-thread
        # por OpenMP, e paralelizar as folds por cima disso sobre-assina a
        # maquina -- um worker morria e o score virava nan silenciosamente.
        # error_score="raise" garante que uma falha estoure em vez de virar nan
        # no csv de resultados.
        scores = cross_val_score(
            modelo,
            X_tr,
            y_tr,
            groups=g_tr,
            cv=GroupKFold(n_splits=FOLDS),
            scoring="neg_mean_absolute_error",
            error_score="raise",
        )
        mae_cv = -scores

        if not np.isfinite(mae_cv).all():
            raise RuntimeError(f"validacao cruzada de '{nome}' produziu score nao-finito")

        modelo.fit(X_tr, y_tr)
        metricas = metricas_em_reais(y_te.to_numpy(), modelo.predict(X_te))

        linhas.append(
            {
                "modelo": nome,
                "cv_mae_log_media": mae_cv.mean(),
                "cv_mae_log_desvio": mae_cv.std(),
                **metricas,
            }
        )
        print(
            f"[{nome:18s}] CV MAE(log)={mae_cv.mean():.4f} +/- {mae_cv.std():.4f} | "
            f"teste MAE=R$ {metricas['mae_reais']:,.0f} | "
            f"erro mediano={metricas['erro_percentual_mediano']:.1f}% | "
            f"R2(log)={metricas['r2_log']:.3f}",
            flush=True,
        )

    resultados = pd.DataFrame(linhas).sort_values("cv_mae_log_media").reset_index(drop=True)
    resultados.to_csv(SAIDA_RESULTADOS, index=False, encoding="utf-8")

    with open(SAIDA_RELATORIO, "w", encoding="utf-8") as f:
        json.dump(
            {
                "semente": dataset.SEMENTE,
                "proporcao_teste": dataset.PROPORCAO_TESTE,
                "folds": FOLDS,
                "split": "GroupShuffleSplit por assinatura fisica do imovel",
                "alvo": "log(preco_venda)",
                "n_treino": int(len(X_tr)),
                "n_teste": int(len(X_te)),
                "n_features": int(X.shape[1]),
                "grupos_vazados": int(vazados),
                "resultados": linhas,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("RESULTADOS (ordenados pelo MAE da validacao cruzada no treino)")
    print("=" * 72)
    print(resultados.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("=" * 72)
    print(f"Resultados: {SAIDA_RESULTADOS}")
    print(f"Relatorio:  {SAIDA_RELATORIO}")

    return resultados


if __name__ == "__main__":
    executar()
