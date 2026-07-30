# -*- coding: utf-8 -*-
"""Junta os arquivos-parte do scrape paralelo no arquivo canônico (dedup por URL).

Depois de rodar os workers (`scraper.py --shard i/N`), cada um deixa um
`imoveis_joao_pessoa.parteXdeN.json`. Este script funde todos eles — mais o que já
existir no canônico — em `imoveis_joao_pessoa.json`, sem duplicar (chave = url_anuncio).

Uso (de qualquer pasta, com o pacote instalado):
    python -m imoveis_jp.scraping.chaves_na_mao.merge_parts
Idempotente: pode rodar quantas vezes quiser. Não apaga as partes (segurança).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from imoveis_jp import config

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

MAIN = config.ANUNCIOS_JSON


def _load(path: Path) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def main() -> None:
    by_url: dict[str, dict] = {}
    partes = sorted(MAIN.parent.glob(f"{MAIN.stem}.parte*.json"))
    files = ([MAIN] if MAIN.exists() else []) + partes
    if not files:
        print("Nenhum arquivo encontrado para fundir.")
        return

    for f in files:
        recs = _load(f)
        novos = 0
        for o in recs:
            u = o.get("url_anuncio")
            if u and u not in by_url:
                by_url[u] = o
                novos += 1
        print(f"  {f}: {len(recs)} registros (+{novos} novos)")

    merged = list(by_url.values())
    tmp = MAIN.with_name(MAIN.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)
    os.replace(tmp, MAIN)  # escrita atômica
    print(f"\n✅ Total unificado em {MAIN}: {len(merged)} anúncios únicos.")
    print("   (as partes foram mantidas; apague-as manualmente quando quiser.)")


if __name__ == "__main__":
    main()
