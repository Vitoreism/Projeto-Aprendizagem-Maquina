# -*- coding: utf-8 -*-
"""Variante de pipeline com PCA, aplicada aos seis candidatos (issue #25).

Hipotese registrada ANTES de rodar: PCA piora. PCA e uma projecao LINEAR, e o
que falta para os modelos lineares (Ridge/OLS) e justamente a interacao
NAO-linear entre area e bairro -- uma rotacao linear do espaco de atributos nao
cria essa interacao, so reduz dimensao. Para os modelos baseados em arvore o
argumento e outro: eles cortam por limiar em atributos individuais, e um
componente principal (combinacao linear de dezenas de dummies de bairro) nao
tem um limiar interpretavel nem alinhado aos cortes que a arvore faria sem PCA.
E tambem destroi a interpretabilidade que da o resultado mais forte do projeto
(bairro isolado e o atributo mais importante, ver docs/modelagem.md).

Demonstrar que piora, com numero, vale mais que omitir.
"""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

from imoveis_jp import config
from imoveis_jp.models import candidatos, dataset, train
from imoveis_jp.models.decisao import CANDIDATOS_NA_DECISAO

SAIDA = config.PROCESSED / "resultados_pca.csv"

#: variancia retida -- escolha reprodutivel e unica para os seis candidatos,
#: em vez de um numero fixo de componentes que faria sentido para um modelo e
#: nao para outro.
VARIANCIA_RETIDA = 0.95


def _log(msg: str) -> None:
    print(msg, flush=True)


def montar_pipeline_pca(candidato, preprocessador) -> Pipeline:
    return Pipeline(
        [
            ("preparo", preprocessador),
            ("pca", PCA(n_components=VARIANCIA_RETIDA, random_state=dataset.SEMENTE)),
            ("regressor", candidato.regressor),
        ]
    )


def executar() -> pd.DataFrame:
    config.ensure_dirs()

    if not train.SAIDA_RESULTADOS.exists():
        raise FileNotFoundError(
            f"'{train.SAIDA_RESULTADOS}' nao existe. Rode antes: python -m imoveis_jp.models.train"
        )

    X, y, grupos = dataset.carregar()
    X_tr, X_te, y_tr, y_te, g_tr, g_te = dataset.dividir(X, y, grupos)
    numericas, binarias, categoricas = dataset.colunas_por_tipo(X)
    inscritos = candidatos.descobrir()

    linhas = []
    for indice, nome in enumerate(CANDIDATOS_NA_DECISAO, start=1):
        candidato = inscritos[nome]
        preprocessador = train.montar_preprocessador(
            numericas, binarias, categoricas, candidato.escalar_binarias
        )
        modelo = montar_pipeline_pca(candidato, preprocessador)

        _log(f"[PCA {indice}/{len(CANDIDATOS_NA_DECISAO)}] {nome}...")
        scores = cross_val_score(
            modelo,
            X_tr,
            y_tr,
            groups=g_tr,
            cv=GroupKFold(n_splits=train.FOLDS),
            scoring="neg_mean_absolute_error",
            error_score="raise",
        )
        mae_pca = -scores

        modelo.fit(X_tr, y_tr)
        n_componentes = int(modelo.named_steps["pca"].n_components_)

        _log(
            f"[PCA] {nome}: MAE(log)={mae_pca.mean():.4f} +/- {mae_pca.std():.4f} "
            f"({n_componentes} componentes p/ {VARIANCIA_RETIDA:.0%} da variancia)"
        )
        linhas.append(
            {
                "modelo": nome,
                "cv_mae_log_pca_media": float(mae_pca.mean()),
                "cv_mae_log_pca_desvio": float(mae_pca.std()),
                "n_componentes_pca": n_componentes,
            }
        )

    resultado = pd.DataFrame(linhas)
    sem_pca = pd.read_csv(train.SAIDA_RESULTADOS)[["modelo", "cv_mae_log_media"]]
    comparacao = resultado.merge(sem_pca, on="modelo", how="left")
    comparacao["diferenca"] = comparacao["cv_mae_log_pca_media"] - comparacao["cv_mae_log_media"]
    comparacao["piorou"] = comparacao["diferenca"] > 0
    comparacao = comparacao.sort_values("cv_mae_log_pca_media").reset_index(drop=True)
    comparacao.to_csv(SAIDA, index=False, encoding="utf-8")

    _log("\n" + "=" * 72)
    _log("PCA -- comparacao com o pipeline sem PCA")
    _log("=" * 72)
    _log(comparacao.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    _log(f"\nSalvo em: {SAIDA}")

    return comparacao


if __name__ == "__main__":
    executar()
