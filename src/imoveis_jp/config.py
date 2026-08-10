# -*- coding: utf-8 -*-
"""Caminhos canônicos do projeto.

Todo I/O de dados passa por aqui, para que nenhum script dependa de qual pasta
você estava quando rodou o comando (era esse o problema da estrutura antiga).

Layout::

    data/raw/        snapshots brutos do scrape (versionados)
    data/interim/    resultados intermediários de limpeza/enriquecimento (fora do git)
    data/processed/  datasets finais, versionados (CSVs prontos p/ modelagem)
"""

from __future__ import annotations

from pathlib import Path

# config.py está em <raiz>/src/imoveis_jp/, logo a raiz é 2 níveis acima.
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

#: Arquivo canônico do scrape do chavesnamao (todos os workers convergem para ele).
ANUNCIOS_JSON = RAW / "imoveis_joao_pessoa.json"

#: Arquivo canônico do scrape do zapimoveis (idem, para os workers do zap).
ANUNCIOS_ZAP_JSON = RAW / "imoveis_joao_pessoa_zap.json"

#: Arquivo intermediário de extrações via LLM (Groq).
EXTRACTIONS_JSON = INTERIM / "extractions_llm.json"


def links_cache(city: str, tipo: str, transacao: str) -> Path:
    """Cache dos links do sitemap para um escopo (cidade/tipo/transação)."""
    return RAW / f"links_{city}_{tipo}_{transacao}.json"


def ensure_dirs() -> None:
    """Cria as pastas de dados se ainda não existirem (idempotente)."""
    for d in (RAW, INTERIM, PROCESSED):
        d.mkdir(parents=True, exist_ok=True)
