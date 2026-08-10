# -*- coding: utf-8 -*-
"""Carregamento e validacao dos artefatos que o dashboard le.

Tudo aqui e somente leitura: os numeros vem da rodada unica da issue #25, e
re-treinar pelo app contrariaria o proprio requisito da issue.

Artefato ausente nao vira traceback na cara de quem esta apresentando -- vira
uma mensagem dizendo qual comando regenera aquele arquivo. Por isso cada
artefato carrega o comando que o produz, e nao so o caminho.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import pandas as pd

from imoveis_jp import config
from imoveis_jp.models import decisao

#: fonte unica da lista de candidatos -- repetir aqui deixaria as duas versoes
#: divergirem no dia em que alguem inscrever um modelo novo.
CANDIDATOS = decisao.CANDIDATOS_NA_DECISAO

#: as duas referencias que nao competem (protocolo_comparacao.md secao 3.1).
REFERENCIAS = ["gradient_boosting", "baseline_mediana"]

TREINO = "python -m imoveis_jp.models.train"
DECISAO = "python -m imoveis_jp.models.decisao"
PCA = "python -m imoveis_jp.models.pca_variant"
TSNE = "python -m imoveis_jp.features.tsne_exploracao"
ANALISE = "python -m imoveis_jp.models.analysis"


@dataclass(frozen=True)
class Artefato:
    caminho: Path
    comando: str
    descricao: str


class ArtefatoAusente(FileNotFoundError):
    """Levantada quando o arquivo nao existe, carregando o comando que o cria."""

    def __init__(self, artefato: Artefato):
        self.artefato = artefato
        self.comando = artefato.comando
        super().__init__(
            f"'{artefato.caminho.name}' ({artefato.descricao}) ainda nao existe.\n"
            f"Gere com:  {artefato.comando}"
        )


ARTEFATOS: Dict[str, Artefato] = {
    "resultados": Artefato(
        config.PROCESSED / "resultados_modelos.csv", TREINO, "metricas de CV e teste"
    ),
    "folds": Artefato(
        config.PROCESSED / "cv_mae_por_fold.csv", TREINO, "MAE por fold, para a comparacao pareada"
    ),
    "decisao": Artefato(
        config.PROCESSED / "decisao_criterio.json", DECISAO, "veredito do criterio"
    ),
    "pca": Artefato(
        config.PROCESSED / "resultados_pca.csv", PCA, "variante PCA dos seis candidatos"
    ),
    "residuos": Artefato(
        config.PROCESSED / "residuos_teste.csv", ANALISE, "residuo de cada anuncio do teste"
    ),
    "segmentos": Artefato(
        config.PROCESSED / "residuos_por_segmento.csv", ANALISE, "erro por bairro e faixa"
    ),
    "importancia": Artefato(
        config.PROCESSED / "importancia_permutacao.csv", ANALISE, "importancia por permutacao"
    ),
    "importancia_codificada": Artefato(
        config.PROCESSED / "importancia_permutacao_codificada.csv",
        ANALISE,
        "importancia por coluna codificada, com o confronto contra a correlacao",
    ),
    "tsne": Artefato(
        config.PROCESSED / "tsne_coords.csv", TSNE, "coordenadas do t-SNE"
    ),
    "matriz": Artefato(
        config.PROCESSED / "features_matrix.csv",
        "python -m imoveis_jp.features.build_features",
        "matriz de features (usada para recuperar a url do anuncio)",
    ),
}


def ler(artefato: Artefato) -> Union[pd.DataFrame, dict]:
    """Le um artefato ja resolvido. Aceita .csv e .json."""
    if not artefato.caminho.exists():
        raise ArtefatoAusente(artefato)

    if artefato.caminho.suffix == ".json":
        with open(artefato.caminho, encoding="utf-8") as f:
            return json.load(f)
    return pd.read_csv(artefato.caminho, low_memory=False)


def carregar(nome: str) -> Union[pd.DataFrame, dict]:
    """Le pelo nome curto do registro. `KeyError` se o nome nao existe."""
    if nome not in ARTEFATOS:
        raise KeyError(
            f"artefato '{nome}' nao esta no registro. Conhecidos: {sorted(ARTEFATOS)}"
        )
    return ler(ARTEFATOS[nome])


def existe(nome: str) -> bool:
    return ARTEFATOS[nome].caminho.exists()


def apenas_candidatos(tabela: pd.DataFrame, coluna: str = "modelo") -> pd.DataFrame:
    """Filtra as duas referencias, deixando so quem competiu pela decisao."""
    return tabela[tabela[coluna].isin(CANDIDATOS)].copy()


def ordenar_por_cv(tabela: pd.DataFrame) -> pd.DataFrame:
    """Ordena pelo MAE da CV -- nunca pelo teste.

    Ordenar pelo teste sugeriria que o teste escolheu o vencedor, que e
    exatamente o que o criterio da issue #25 proibe.
    """
    coluna = "cv_mae_log_media" if "cv_mae_log_media" in tabela.columns else "mae_log"
    return tabela.sort_values(coluna).reset_index(drop=True)
