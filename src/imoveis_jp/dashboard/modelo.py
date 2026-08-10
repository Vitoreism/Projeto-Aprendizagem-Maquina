# -*- coding: utf-8 -*-
"""Ajuste do modelo vencedor, previsao de um imovel e sorteio do conjunto de teste.

POR QUE REAJUSTAR EM VEZ DE CARREGAR UM .joblib

Nada e persistido pelo train.py hoje. Salvar o estimador resolveria o custo do
boot (10,1s medidos na rodada da issue #25), mas criaria um binario no
repositorio que pode divergir em silencio do codigo que o gerou -- e ninguem
descobre olhando o arquivo. Dez segundos uma vez, sob st.cache_resource, valem
mais que um artefato de procedencia duvidosa.

POR QUE condominio E iptu VAO COMO NULO

O Pipeline treinou com SimpleImputer(add_indicator=True): a ausencia vira uma
coluna propria, e o modelo aprendeu a usa-la -- iptu falta em cerca de 80% dos
anuncios, entao o silencio do anunciante e sinal, nao buraco. Preencher com a
mediana mentiria para o modelo; mandar nulo usa exatamente o caminho em que ele
foi treinado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from imoveis_jp.dashboard import dados
from imoveis_jp.models import dataset, train

#: o vencedor do criterio da issue #25.
MODELO = "gradient_boosting_ajustado"

NUMERICAS = dataset.NUMERICAS
CATEGORICAS = dataset.CATEGORICAS

#: campos que o usuario preenche, escolhidos pela importancia por permutacao ja
#: medida (docs/comparacao_modelos.md), nao por palpite. 34 dos 76 atributos tem
#: importancia indistinguivel de zero -- expor todos seria pedir 76 respostas
#: para mudar quase nada.
CAMPOS_PRINCIPAIS = [
    "area_util",
    "bairro",
    "garagens",
    "quartos",
    "suites",
    "banheiros",
    "area_total",
    "tipo_unidade",
    "status_construcao",
    "origem_anuncio",
    "venda_direta",
]

#: as unicas comodidades com importancia acima do ruido da permutacao.
COMODIDADES_UTEIS = [
    "com_piso_ceramica",
    "com_varanda_gourmet",
    "com_elevador",
    "com_vista_ou_acesso_praia",
    "com_piso_porcelanato",
]

#: acima disso ha pouco dado (171 dos 3.087 anuncios de teste, 5,5%), e a
#: hipotese registrada do boosting ja previa que ele extrapola mal nessa faixa.
TETO_CONFIAVEL = 2_000_000.0

#: quantis do erro percentual usados na faixa de incerteza.
QUANTIL_INFERIOR = 0.10
QUANTIL_SUPERIOR = 0.90


@dataclass
class Previsao:
    central: float
    inferior: float
    superior: float
    avisos: List[str] = field(default_factory=list)


def _base() -> pd.DataFrame:
    """A matriz completa, sem o alvo -- serve de molde para a linha de entrada."""
    X, _, _ = dataset.carregar()
    return X


def ajustar() -> Pipeline:
    """Reajusta o vencedor no MESMO split da comparacao.

    Usa `train.montar_modelos`, entao o pre-processamento e identico ao que
    produziu os numeros relatados -- se alguem mexer no Pipeline, o app muda
    junto, sem uma segunda definicao para manter em dia.
    """
    X, y, grupos = dataset.carregar()
    X_tr, _, y_tr, _, _, _ = dataset.dividir(X, y, grupos)

    numericas, binarias, categoricas = dataset.colunas_por_tipo(X)
    modelo = train.montar_modelos(numericas, binarias, categoricas)[MODELO]
    modelo.fit(X_tr, y_tr)
    return modelo


def faixa_de_incerteza() -> Dict[str, float]:
    """Quantis do erro percentual observado no teste, lidos do artefato.

    Nao sao constantes escritas a mao: se a base mudar e o analysis.py rodar de
    novo, a faixa acompanha.
    """
    residuos = dados.carregar("residuos")
    return {
        "inferior": float(residuos["erro_percentual"].quantile(QUANTIL_INFERIOR)),
        "superior": float(residuos["erro_percentual"].quantile(QUANTIL_SUPERIOR)),
        "mediano_absoluto": float(residuos["erro_absoluto_percentual"].median()),
    }


def limites_de_area() -> Dict[str, float]:
    coluna = _base()["area_util"].dropna()
    return {"minimo": float(coluna.min()), "maximo": float(coluna.max())}


def opcoes_categoricas() -> Dict[str, List[str]]:
    """Valores possiveis de cada nominal, em ordem de frequencia.

    Ordenar por frequencia deixa o caso comum no topo do selectbox, em vez de
    forcar o usuario a caçar 'bessa' no meio de 66 bairros em ordem alfabetica.
    """
    X = _base()
    return {
        coluna: X[coluna].value_counts().index.tolist()
        for coluna in CATEGORICAS
        if coluna in X.columns
    }


def linha_padrao() -> pd.DataFrame:
    """Uma linha com as 76 colunas, nos defaults do spec.

    Continuas vao na mediana, EXCETO condominio e iptu, que vao nulos de
    proposito (ver o docstring do modulo). Binarias vao False. Nominais vao na
    moda -- inclusive posicao_solar, que nem e exposta na interface porque o
    anunciante quase nunca preenche.
    """
    X = _base()
    linha = {}

    for coluna in X.columns:
        if coluna in ("condominio", "iptu"):
            linha[coluna] = np.nan
        elif coluna in NUMERICAS:
            linha[coluna] = float(X[coluna].median())
        elif coluna in CATEGORICAS:
            linha[coluna] = X[coluna].mode().iloc[0]
        else:
            linha[coluna] = 0

    return pd.DataFrame([linha], columns=X.columns)


def prever(modelo: Pipeline, linha: pd.DataFrame) -> Previsao:
    """Preve em reais, com faixa de incerteza e avisos de extrapolacao.

    O modelo preve log(preco); a exponencial devolve para reais. A faixa NAO e
    um intervalo de confianca estatistico -- e a dispersao empirica do erro que
    o modelo cometeu no teste, aplicada a esta previsao. Mostrar so o numero
    central seria precisao falsa: o erro mediano do modelo e 15,6%.
    """
    central = float(np.exp(modelo.predict(linha)[0]))

    faixa = faixa_de_incerteza()
    inferior = central * (1 + faixa["inferior"] / 100)
    superior = central * (1 + faixa["superior"] / 100)

    avisos = []
    limites = limites_de_area()
    area = linha.iloc[0].get("area_util")
    if pd.notna(area) and not (limites["minimo"] <= area <= limites["maximo"]):
        avisos.append(
            f"Área de {area:.0f} m² está fora da faixa vista no treino "
            f"({limites['minimo']:.0f} a {limites['maximo']:.0f} m²). "
            f"O modelo não aprendeu essa região."
        )

    if central > TETO_CONFIAVEL:
        avisos.append(
            f"Previsão acima de R$ 2 milhões, onde há pouco dado (5,5% do teste). "
            f"A hipótese registrada do boosting já previa que ele extrapola mal aqui."
        )

    return Previsao(central=central, inferior=inferior, superior=superior, avisos=avisos)


def sortear_do_teste(semente: int = 42) -> Dict:
    """Sorteia um anuncio real do conjunto de teste, com previsao ja calculada.

    O teste e o unico lugar onde o imovel e ao mesmo tempo real e nunca visto
    pelo modelo -- e por isso a demonstracao honesta. Os numeros ja estao em
    residuos_teste.csv (gerado pelo analysis.py), entao nao ha nada a reajustar
    aqui; a url vem da matriz, pelo indice que o residuo preservou.
    """
    residuos = dados.carregar("residuos")
    matriz = dados.carregar("matriz")

    sorteado = residuos.sample(n=1, random_state=semente).iloc[0]

    # residuos_teste.csv guarda a posicao original da matriz no indice; o
    # analysis.py monta a tabela com `index=X_te.index`, e o to_csv descarta o
    # indice -- entao reencontramos o anuncio pelo par (preco, area), que e
    # praticamente unico, caindo para a url vazia se nao bater.
    candidatos = matriz[
        np.isclose(matriz["preco_venda"], sorteado["preco_real"])
        & np.isclose(matriz["area_util"], sorteado["area_util"], equal_nan=True)
    ]
    url = str(candidatos["url_anuncio"].iloc[0]) if len(candidatos) else ""

    return {
        "preco_real": float(sorteado["preco_real"]),
        "preco_previsto": float(sorteado["preco_previsto"]),
        "erro_reais": float(sorteado["erro_reais"]),
        "erro_percentual": float(sorteado["erro_percentual"]),
        "bairro": sorteado["bairro"],
        "area_util": float(sorteado["area_util"]),
        "tipo_unidade": sorteado["tipo_unidade"],
        "origem_anuncio": sorteado["origem_anuncio"],
        "faixa_preco": sorteado["faixa_preco"],
        "url_anuncio": url,
    }
