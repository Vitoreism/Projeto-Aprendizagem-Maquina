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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from imoveis_jp import config
from imoveis_jp.models import dataset

SAIDA_RESULTADOS = config.PROCESSED / "resultados_modelos.csv"
SAIDA_RELATORIO = config.INTERIM / "relatorio_treino.json"

FOLDS = 5


def montar_preprocessador(numericas: List[str], binarias: List[str]) -> ColumnTransformer:
    """Imputacao e escala nas continuas; as binarias ja estao em 0/1.

    add_indicator preserva a informacao de ausencia: com iptu ausente em 80% dos
    anuncios, o proprio silencio do anunciante e sinal, nao ruido a ser apagado.
    """
    continuas = Pipeline(
        [
            ("imputa", SimpleImputer(strategy="median", add_indicator=True)),
            ("escala", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [("num", continuas, numericas), ("bin", "passthrough", binarias)],
        remainder="drop",
    )


def montar_modelos(numericas: List[str], binarias: List[str]) -> Dict[str, Pipeline]:
    def com_preparo(regressor):
        return Pipeline(
            [("preparo", montar_preprocessador(numericas, binarias)), ("regressor", regressor)]
        )

    return {
        # piso absoluto: prever sempre a mediana. qualquer modelo tem que ganhar disto.
        "baseline_mediana": com_preparo(DummyRegressor(strategy="median")),
        "ridge": com_preparo(Ridge(alpha=1.0, random_state=dataset.SEMENTE)),
        "gradient_boosting": com_preparo(
            HistGradientBoostingRegressor(random_state=dataset.SEMENTE)
        ),
    }


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

    numericas, binarias = dataset.colunas_por_tipo(X)
    modelos = montar_modelos(numericas, binarias)

    linhas = []
    for nome, modelo in modelos.items():
        # a CV reajusta imputacao e escala dentro de cada fold
        scores = cross_val_score(
            modelo,
            X_tr,
            y_tr,
            groups=g_tr,
            cv=GroupKFold(n_splits=FOLDS),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        mae_cv = -scores

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
