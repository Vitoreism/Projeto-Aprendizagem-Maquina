# -*- coding: utf-8 -*-
"""
processa as comodidades brutas (area privativa e comum) extraidas do html
e converte em colunas binarias (0 ou 1) via pandas/one-hot encoding
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
    # remove acentos, pontuacao e converte espacos para underline
    if not texto:
        return ""
    texto_sem_acento = unicodedata.normalize("NFD", texto)
    texto_limpo = "".join(c for c in texto_sem_acento if unicodedata.category(c) != "Mn")
    texto_limpo = texto_limpo.lower().strip()
    texto_limpo = re.sub(r"[^\w\s]", "", texto_limpo)
    texto_limpo = re.sub(r"\s+", "_", texto_limpo)
    return texto_limpo


def mapear_sinonimos(nome_comodidade: str) -> str:
    # agrupa variações equivalentes para evitar colunas repetidas no dataset
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
    config.ensure_dirs()
    input_file = config.ANUNCIOS_JSON

    if not input_file.exists():
        print(f"[Erro] Arquivo '{input_file}' nao encontrado.")
        sys.exit(1)

    print(f"[Info] Lendo anuncios brutos de: {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        imoveis = json.load(f)

    # conta a frequencia de cada comodidade em todo o dataset
    contador_comodidades: Counter = Counter()
    registros_imoveis: List[Dict[str, Any]] = []

    for item in imoveis:
        url = item.get("url_anuncio")
        if not url:
            continue

        comodidades_imovel: Set[str] = set()

        # junta comodidades da area privativa e comum
        privativa = item.get("comodidades_area_privativa")
        if privativa and isinstance(privativa, str):
            for termo in privativa.split(","):
                norm = mapear_sinonimos(normalizar_texto(termo))
                if norm:
                    comodidades_imovel.add(norm)

        comum = item.get("comodidades_area_comum")
        if comum and isinstance(comum, str):
            for termo in comum.split(","):
                norm = mapear_sinonimos(normalizar_texto(termo))
                if norm:
                    comodidades_imovel.add(norm)

        for com in comodidades_imovel:
            contador_comodidades[com] += 1

        registros_imoveis.append({
            "url_anuncio": url,
            "comodidades_set": comodidades_imovel
        })

    # filtra apenas comodidades com frequencia relevante
    comodidades_relevantes = [
        com for com, count in contador_comodidades.most_common()
        if count >= frequencia_minima
    ]

    print(f"[Info] Total de comodidades unicas encontradas: {len(contador_comodidades)}")
    print(f"[Info] Comodidades selecionadas (aparecem em >= {frequencia_minima} imoveis): {len(comodidades_relevantes)}")

    # gera as colunas binarias 0 ou 1 para cada imovel (one-hot encoding)
    linhas_dataset: List[Dict[str, Any]] = []

    for reg in registros_imoveis:
        linha = {"url_anuncio": reg["url_anuncio"]}
        set_imovel = reg["comodidades_set"]

        for com in comodidades_relevantes:
            linha[f"comodidade_{com}"] = 1 if com in set_imovel else 0

        linhas_dataset.append(linha)

    df_resultado = pd.DataFrame(linhas_dataset)
    return df_resultado, contador_comodidades


def exportar_comodidades_csv() -> None:
    # salva o dataframe processado no interim
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
    print("TOP 15 COMODIDADES MAIS FREQUENTES:")
    for com, count in contador.most_common(15):
        print(f"  - {com:30s}: {count:,} imoveis ({(count/len(df_comodidades))*100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    exportar_comodidades_csv()
