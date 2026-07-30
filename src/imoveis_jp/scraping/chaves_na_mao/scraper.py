# -*- coding: utf-8 -*-
"""Scraper robusto, educado e resumível do chavesnamao (ADR-009).

Feito para rodar **a noite toda sem supervisão**:

* **Educado (não toma bloqueio):** sleep aleatório entre cada anúncio
  (``--min-sleep``/``--max-sleep``) + uma pausa longa a cada N anúncios
  (``--long-pause-every``). Bloqueia imagens/fontes/mídia no navegador para
  gastar menos banda e ir mais rápido.
* **Resumível:** os links coletados ficam em cache (``--links-cache``); os
  anúncios já salvos no JSON de saída são **pulados**. Pode parar com Ctrl+C e
  rodar de novo depois — ele continua de onde parou, sem perder nada.
* **Robusto:** cada página tem *retry* com *backoff*; falhas isoladas são
  registradas e o robô segue. Checkpoint salvo em disco a cada ``--checkpoint``.

**Escopo é parâmetro** (default = o domínio do projeto: apartamentos à venda em
João Pessoa). Ex.: ``--tipo casa`` ou ``--transacao aluguel`` para ampliar.

robots.txt do site (verificado): páginas de anúncio são permitidas; o
Content-Signal declara ``ai-train=no`` (NÃO usar para treinar/fine-tunar modelos),
``search=yes`` e ``ai-input=yes`` (RAG/grounding permitido). Este projeto só usa os
dados para grounding — compatível. Não reutilize o corpus para treino de modelos.

Uso rápido (de qualquer pasta, com o pacote instalado — ver README):
    python -m imoveis_jp.scraping.chaves_na_mao.scraper                     # tudo
    python -m imoveis_jp.scraping.chaves_na_mao.scraper --max 50            # teste
    python -m imoveis_jp.scraping.chaves_na_mao.scraper --transacao aluguel
Veja `python -m imoveis_jp.scraping.chaves_na_mao.scraper --help`.

Entrada/saída ficam em ``data/raw/`` (ver :mod:`imoveis_jp.config`).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from imoveis_jp import config
from imoveis_jp.scraping.chaves_na_mao.parser import ScraperChavesNaMao

# Console do Windows costuma ser cp1252 e quebra ao imprimir emoji/acentos.
# Força UTF-8 na saída (belt-and-suspenders: 'replace' em vez de crashar).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

SITEMAP_INDEX = "https://www.chavesnamao.com.br/sitemap.xml"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    # sem 'br': requests não decodifica brotli sem lib e o sitemap voltava comprimido
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "pt-BR,pt;q=0.9",
}


# ── coleta de links (sitemap) ──────────────────────────────────────────────────
def _fetch_xml(url: str, tries: int = 3) -> str:
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            if url.endswith(".gz"):
                return gzip.GzipFile(fileobj=io.BytesIO(r.content)).read().decode("utf-8", "ignore")
            return r.text
        except Exception as exc:  # noqa: BLE001
            if attempt == tries - 1:
                print(f"    ! falha em {url}: {exc}")
                return ""
            time.sleep(2 * (attempt + 1))
    return ""


def collect_links(city: str, tipo: str, transacao: str) -> list[str]:
    """Varre o índice de sitemaps e devolve os links de anúncios do escopo."""
    print(f"Coletando links do sitemap (city={city}, tipo={tipo}, transacao={transacao})...")
    index_xml = _fetch_xml(SITEMAP_INDEX)
    subs = re.findall(r"<loc>(.*?)</loc>", index_xml)

    if transacao == "ambos":
        wanted = ("venda", "aluguel")
    else:
        wanted = (transacao,)
    shards = [s for s in subs if any(re.search(rf"sitemap-{w}-imoveis-?\d*\.xml", s) for w in wanted)]
    print(f"  {len(shards)} shards de anúncios a varrer.")

    links: set[str] = set()
    for i, shard in enumerate(shards, 1):
        xml = _fetch_xml(shard)
        for link in re.findall(r"<loc>(.*?)</loc>", xml):
            lt = link.lower()
            if city not in lt or tipo not in lt:
                continue
            is_rent = "aluguel" in lt or "locacao" in lt
            if transacao == "venda" and (is_rent or "venda" not in lt):
                continue
            if transacao == "aluguel" and not is_rent:
                continue
            links.add(link)
        if i % 50 == 0 or i == len(shards):
            print(f"    ...{i}/{len(shards)} shards | {len(links)} links no escopo")
        time.sleep(0.3)  # gentileza também na varredura de sitemap
    result = sorted(links)
    print(f"✅ {len(result)} links únicos no escopo.\n")
    return result


# ── util de I/O resumível ───────────────────────────────────────────────────────
def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_json_list(path: Path, data: list) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)  # escrita atômica: não corrompe se cair no meio


def _fmt_eta(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


# ── carga de página robusta (Playwright) ─────────────────────────────────────────
def _block_heavy(route) -> None:
    if route.request.resource_type in ("image", "media", "font"):
        route.abort()
    else:
        route.continue_()


def _load_html(page, url: str, tries: int = 3) -> str | None:
    for attempt in range(tries):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            try:
                page.wait_for_load_state("networkidle", timeout=12000)
            except Exception:  # noqa: BLE001
                pass
            try:
                page.wait_for_selector("h1", timeout=8000)
            except Exception:  # noqa: BLE001
                pass
            return page.content()
        except Exception as exc:  # noqa: BLE001
            wait = 5 * (attempt + 1)
            print(f"    ! tentativa {attempt + 1}/{tries} falhou ({exc}); aguardando {wait}s")
            time.sleep(wait)
    return None


# ── loop principal ────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    links_cache = (
        Path(args.links_cache) if args.links_cache
        else config.links_cache(args.city, args.tipo, args.transacao)
    )

    if args.refresh_links or not links_cache.exists():
        links = collect_links(args.city, args.tipo, args.transacao)
        _save_json_list(links_cache, links)
    else:
        links = _load_json_list(links_cache)
        print(f"Links carregados do cache '{links_cache}': {len(links)}\n")

    if args.max > 0:
        links = links[: args.max]

    # ── paralelismo por sharding ──────────────────────────────────────────────
    # Cada worker (--shard i/N) pega uma fatia DISJUNTA (links[i::N]) e escreve no
    # SEU arquivo, evitando contenção. Todos pulam o que já está no arquivo canônico.
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    canon = config.ANUNCIOS_JSON
    out = Path(args.out) if args.out else canon
    if shard_n > 1 and out == canon:
        out = canon.with_name(f"{canon.stem}.parte{shard_i + 1}de{shard_n}.json")

    # conjunto "já feito" = arquivo deste worker + arquivos indicados em --skip-from
    # + (em modo shard) o arquivo canônico, para não repetir o que já foi coletado.
    skip_files = [Path(p) for p in args.skip_from]
    if shard_n > 1 and canon.exists() and canon != out:
        skip_files.append(canon)

    dados = _load_json_list(out)
    seen = {d.get("url_anuncio") for d in dados}
    for sf in skip_files:
        seen |= {d.get("url_anuncio") for d in _load_json_list(sf)}

    pendentes = [u for u in links if u not in seen]
    if shard_n > 1:
        pendentes = pendentes[shard_i::shard_n]  # fatia disjunta deste worker

    tag = f"[shard {shard_i + 1}/{shard_n}] " if shard_n > 1 else ""
    print(f"{tag}saída: {out}")
    print(f"{tag}total no escopo: {len(links)} | já feito (todos os arquivos): {len(seen)} "
          f"| a fazer neste worker: {len(pendentes)}\n")

    if args.dry_run:
        print("--dry-run: planejamento apenas, nada foi baixado.")
        return
    if not pendentes:
        print("Nada a fazer — nada pendente para este worker. 🎉")
        return

    parser = ScraperChavesNaMao()  # reutiliza só os métodos de extração (seletores)
    feitos = 0
    t0 = time.time()

    with sync_playwright() as p:
        navegador = p.chromium.launch(headless=not args.headful)
        page = navegador.new_page()
        page.route("**/*", _block_heavy)
        try:
            for i, url in enumerate(pendentes, 1):
                html = _load_html(page, url)
                if html:
                    parser.soup = BeautifulSoup(html, "html.parser")
                    parser.url_atual = url
                    registro = parser.extrair_dados_do_anuncio()
                    if registro:
                        registro["descricao_completa"] = parser.extrair_descricao_anuncio()
                        dados.append(registro)
                        feitos += 1
                else:
                    print(f"    x desistindo de {url} (falhou {3}x)")

                if feitos and feitos % args.checkpoint == 0:
                    _save_json_list(out, dados)
                    rate = (time.time() - t0) / i
                    eta = rate * (len(pendentes) - i)
                    print(f"  💾 checkpoint: {len(dados)} salvos | {i}/{len(pendentes)} | ETA {_fmt_eta(eta)}")

                # gentileza: sleep entre anúncios + pausa longa periódica
                time.sleep(random.uniform(args.min_sleep, args.max_sleep))
                if args.long_pause_every and i % args.long_pause_every == 0:
                    print(f"  ⏸️  pausa longa de {args.long_pause}s (anti-bloqueio) após {i} páginas")
                    time.sleep(args.long_pause)
        except KeyboardInterrupt:
            print("\n⚠️  interrompido pelo usuário — salvando o progresso...")
        finally:
            _save_json_list(out, dados)  # dados garantidos ANTES de fechar o browser
            try:
                navegador.close()
            except Exception:  # noqa: BLE001
                # Ctrl+C pode derrubar o driver do Playwright antes deste close;
                # o save acima já persistiu tudo, então o erro é inofensivo.
                pass

    print(f"\n🎉 Fim. Novos nesta rodada: {feitos} | total no arquivo '{out}': {len(dados)}")
    print("   (para retomar/complementar, é só rodar de novo — ele pula o que já está salvo.)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="pb-joao-pessoa", help="slug da cidade na URL (default: pb-joao-pessoa)")
    ap.add_argument("--tipo", default="apartamento", help="tipo de imóvel na URL (default: apartamento)")
    ap.add_argument("--transacao", default="venda", choices=["venda", "aluguel", "ambos"])
    ap.add_argument("--out", default="", help="JSON de saída, resumível (default: data/raw/imoveis_joao_pessoa.json)")
    ap.add_argument("--links-cache", default="", help="cache dos links (default: data/raw/links_<escopo>.json)")
    ap.add_argument("--max", type=int, default=0, help="limita a N anúncios (0 = todos; use p/ testar)")
    ap.add_argument("--min-sleep", type=float, default=3.0, help="sleep mínimo entre anúncios, s (default 3)")
    ap.add_argument("--max-sleep", type=float, default=7.0, help="sleep máximo entre anúncios, s (default 7)")
    ap.add_argument("--long-pause-every", type=int, default=100, help="pausa longa a cada N páginas (0 desliga)")
    ap.add_argument("--long-pause", type=float, default=30.0, help="duração da pausa longa, s (default 30)")
    ap.add_argument("--checkpoint", type=int, default=25, help="salva o JSON a cada N anúncios novos (default 25)")
    ap.add_argument("--refresh-links", action="store_true", help="recoleta os links do sitemap (ignora o cache)")
    ap.add_argument("--headful", action="store_true", help="mostra o navegador (debug)")
    # paralelismo
    ap.add_argument("--shard", default="0/1", metavar="i/N",
                    help="paraleliza: este é o worker i de N (0-indexado). Ex.: --shard 0/3")
    ap.add_argument("--skip-from", action="append", default=[], metavar="ARQUIVO",
                    help="também pula URLs já presentes neste arquivo (pode repetir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="mostra o planejamento (fatia, saída, pendentes) e sai sem baixar")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
