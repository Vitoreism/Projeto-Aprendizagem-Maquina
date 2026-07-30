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
    # mapeia e padroniza todos os sinonimos encontrados nas 284 comodidades brutas
    s = nome_comodidade

    if "churrasqueira" in s:
        return "churrasqueira"
    if "piscina" in s:
        return "piscina"
    if "portaria" in s or "seguranca" in s or "guarita" in s or "camera" in s or "circuito" in s or "vigilancia" in s or "porteiro" in s or "portao" in s:
        return "portaria_seguranca_24h"
    if "elevador" in s:
        return "elevador"
    if "varanda" in s or "sacada" in s:
        return "varanda_gourmet" if "gourmet" in s else "varanda"
    if "quadra" in s or "campo" in s or "squash" in s or "tenis" in s:
        return "quadra_esportiva"
    if "garagem" in s or "estacionamento" in s or "vaga" in s:
        return "vaga_garagem"
    if "pet" in s or "animais" in s:
        return "permitido_pets"
    if "eletrico" in s or "eletrica" in s:
        return "carregador_carro_eletrico" if ("carro" in s or "veiculo" in s) else s
    if "coworking" in s or "escritorio" in s or "workspace" in s or "reuniao" in s or "conves" in s:
        return "coworking_escritorio"
    if "academia" in s or "fitness" in s or "ginastica" in s or "pilates" in s:
        return "academia"
    if "armario" in s or "planejado" in s or "projetado" in s or "embutido" in s:
        return "moveis_armarios_planejados"
    if "brinquedoteca" in s or "kids" in s:
        return "brinquedoteca_kids"
    if "playground" in s:
        return "playground"
    if "jogos" in s or "xadrez" in s:
        return "salao_de_jogos"
    if "festas" in s:
        return "salao_de_festas"
    if "gourmet" in s:
        return "espaco_gourmet"
    if "mar" in s or "praia" in s:
        return "vista_ou_acesso_praia"
    if "porcelanato" in s:
        return "piso_porcelanato"
    if "cerâmica" in s or "ceramica" in s:
        return "piso_ceramica"

    return s


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

    print(f"[Info] Total de comodidades unicas encontradas (apos sinônimos): {len(contador_comodidades)}")
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
    print("TOP 20 COMODIDADES MAIS FREQUENTES (APOS MAPEAMENTO DE SINONIMOS):")
    for com, count in contador.most_common(20):
        print(f"  - {com:35s}: {count:,} imoveis ({(count/len(df_comodidades))*100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    exportar_comodidades_csv()
