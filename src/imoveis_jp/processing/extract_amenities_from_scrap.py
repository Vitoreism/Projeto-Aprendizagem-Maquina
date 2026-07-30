# -*- coding: utf-8 -*-
"""Modulo de Processamento Deterministico e Transparente de Comodidades do Scrape.

Objetivo:
Converter as listas de texto de 'comodidades_area_privativa' e 'comodidades_area_comum'
(coletadas via HTML do portal) em colunas binarias de presenca (0 ou 1) para cada imovel.

Este codigo utiliza regras deterministicas simples (One-Hot Encoding em Python/Pandas),
sendo 100% explicavel, transparente e sem dependencia de modelos de IA (sem caixa-preta).
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from imoveis_jp import config


def normalizar_texto(texto: str) -> str:
    """Normaliza o texto removendo acentos, caracteres especiais e convertendo para minusculas.

    Exemplo:
        "Area de servico" -> "area_de_servico"
        "Portaria 24h!" -> "portaria_24h"
    """
    if not texto:
        return ""
    # Remove acentuacao usando decomposicao NFD
    texto_sem_acento = unicodedata.normalize("NFD", texto)
    texto_limpo = "".join(c for c in texto_sem_acento if unicodedata.category(c) != "Mn")
    # Converte para minusculas e substitui espacos/hifen por underline
    texto_limpo = texto_limpo.lower().strip()
    texto_limpo = re.sub(r"[^\w\s]", "", texto_limpo)  # Remove pontuacoes
    texto_limpo = re.sub(r"\s+", "_", texto_limpo)      # Substitui espacos por _
    return texto_limpo


def mapear_sinonimos(nome_comodidade: str) -> str:
    """Padroniza termos equivalentes para evitar colunas duplicadas.

    Exemplo:
        "churrasqueira_coletiva" -> "churrasqueira"
        "portaria_24h" -> "portaria_24h"
    """
    if "churrasqueira" in nome_comodidade:
        return "churrasqueira"
    if "portaria" in nome_comodidade or "seguranca" in nome_comodidade:
        return "portaria_seguranca_24h" if ("24" in nome_comodidade or "seguranca" in nome_comodidade) else "portaria"
    if "elevador" in nome_comodidade:
        return "elevador"
    if "varanda" in nome_comodidade or "sacada" in nome_comodidade:
        return "varanda_gourmet" if "gourmet" in nome_comodidade else "varanda"
    if "quadra" in nome_comodidade:
        return "quadra_esportiva"
    if "garagem" in nome_comodidade or "estacionamento" in nome_comodidade:
        return "vaga_garagem"
    return nome_comodidade


def extrair_matriz_comodidades_html(
    frequencia_minima: int = 20,
) -> Tuple[pd.DataFrame, Counter]:
    """Le os anuncios brutos em data/raw/imoveis_joao_pessoa.json e gera a matriz binaria.

    Args:
        frequencia_minima: Filtra apenas comodidades que aparecem em pelo menos N imoveis.

    Returns:
        DataFrame com colunas (url_anuncio, indicador_comodidade_1, indicador_comodidade_2, ...)
        e o contador de frequencias de cada comodidade.
    """
    config.ensure_dirs()
    input_file = config.ANUNCIOS_JSON

    if not input_file.exists():
        print(f"[Erro] Arquivo de entrada '{input_file}' nao encontrado.")
        sys.exit(1)

    print(f"[Info] Lendo anuncios brutos de: {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        imoveis = json.load(f)

    # 1. Primeira passada: Descobrir todas as comodidades unicas e suas frequencias
    contador_comodidades: Counter = Counter()
    registros_imoveis: List[Dict[str, Any]] = []

    for item in imoveis:
        url = item.get("url_anuncio")
        if not url:
            continue

        comodidades_imovel: Set[str] = set()

        # Extrai de comodidades_area_privativa
        privativa = item.get("comodidades_area_privativa")
        if privativa and isinstance(privativa, str):
            for termo in privativa.split(","):
                norm = mapear_sinonimos(normalizar_texto(termo))
                if norm:
                    comodidades_imovel.add(norm)

        # Extrai de comodidades_area_comum
        comum = item.get("comodidades_area_comum")
        if comum and isinstance(comum, str):
            for termo in comum.split(","):
                norm = mapear_sinonimos(normalizar_texto(termo))
                if norm:
                    comodidades_imovel.add(norm)

        # Atualiza o contador geral
        for com in comodidades_imovel:
            contador_comodidades[com] += 1

        registros_imoveis.append({
            "url_anuncio": url,
            "comodidades_set": comodidades_imovel
        })

    # 2. Filtrar apenas comodidades relevantes (frequencia >= frequencia_minima)
    comodidades_relevantes = [
        com for com, count in contador_comodidades.most_common()
        if count >= frequencia_minima
    ]

    print(f"[Info] Total de comodidades unicas identificadas: {len(contador_comodidades)}")
    print(f"[Info] Comodidades selecionadas (frequencia >= {frequencia_minima}): {len(comodidades_relevantes)}")

    # 3. Construir as colunas binarias (0 ou 1) para cada imovel (One-Hot Encoding)
    linhas_dataset: List[Dict[str, Any]] = []

    for reg in registros_imoveis:
        linha = {"url_anuncio": reg["url_anuncio"]}
        set_imovel = reg["comodidades_set"]

        for com in comodidades_relevantes:
            # Coluna recebe 1 se a comodidade esta presente no imovel, senao 0
            linha[f"comodidade_{com}"] = 1 if com in set_imovel else 0

        linhas_dataset.append(linha)

    df_resultado = pd.DataFrame(linhas_dataset)
    return df_resultado, contador_comodidades


def exportar_comodidades_csv() -> None:
    """Executa a extracao e salva em data/interim/amenities_scraped_normalized.csv."""
    df_comodidades, contador = extrair_matriz_comodidades_html(frequencia_minima=20)

    output_csv = config.INTERIM / "amenities_scraped_normalized.csv"
    df_comodidades.to_csv(output_csv, index=False, encoding="utf-8")

    print("\n" + "=" * 65)
    print("EXTRACAO DETERMINISTICA DE COMODIDADES CONCLUIDA COM SUCESSO!")
    print("=" * 65)
    print(f"Arquivo salvo em: {output_csv}")
    print(f"Total de imoveis processados: {len(df_comodidades):,}")
    print(f"Total de colunas binarias criadas: {len(df_comodidades.columns) - 1}")
    print("-" * 65)
    print("TOP 15 COMODIDADES MAIS FREQUENTES EXTRAIDAS DO SCRAPE:")
    for com, count in contador.most_common(15):
        print(f"  - {com:30s}: {count:,} imoveis ({(count/len(df_comodidades))*100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    exportar_comodidades_csv()
