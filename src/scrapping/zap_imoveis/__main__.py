import argparse
import sys
import os
import json
import glob

try:
    from .scraper import ScraperEngine
    from .storage import StorageManager
    from .config import DIR_ATUAL, ARQUIVO_SAIDA
except ImportError:
    from scraper import ScraperEngine
    from storage import StorageManager
    from config import DIR_ATUAL, ARQUIVO_SAIDA


def merge_worker_files():
    """Junta todos os arquivos JSON gerados por trabalhadores ou faixas de partições na base principal."""
    all_files = glob.glob(os.path.join(DIR_ATUAL, "imoveis_joao_pessoa_zap_*.json"))
    worker_files = sorted([f for f in all_files if f != ARQUIVO_SAIDA])
    if not worker_files:
        print("[!] Nenhum arquivo de trabalhador/partição (imoveis_joao_pessoa_zap_*.json) encontrado para fusão.")
        return

    print("=" * 65)
    print(f"[*] FUSÃO DE TRABALHADORES E PARTIÇÕES: {len(worker_files)} arquivos encontrados")
    print("=" * 65)
    for wf in worker_files:
        print(f"    - {os.path.basename(wf)}")

    main_storage = StorageManager(file_path=ARQUIVO_SAIDA)
    total_added = 0

    for wf in worker_files:
        try:
            with open(wf, 'r', encoding='utf-8') as f:
                records = json.load(f)
            if records:
                added = main_storage.save_batch(records)
                total_added += added
                print(f"   [+] Unificados {len(records)} registros de '{os.path.basename(wf)}' -> {added} novos únicos gravados.")
        except Exception as e:
            print(f"   [!] Erro ao ler '{os.path.basename(wf)}': {e}")

    print("\n" + "=" * 65)
    print(f"[FUSÃO CONCLUÍDA] Base principal '{os.path.basename(ARQUIVO_SAIDA)}': {main_storage.get_total_collected()} imóveis únicos no total.")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Scraper Zap Imóveis - Execução individual ou paralela por trabalhadores (Workers)."
    )
    parser.add_argument("-w", "--worker", type=int, default=1, help="ID do trabalhador atual (ex: 1, 2, 3). Padrão: 1")
    parser.add_argument("-t", "--total-workers", type=int, default=1, help="Número total de trabalhadores paralelos. Padrão: 1")
    parser.add_argument("--partition-range", type=int, nargs=2, metavar=("START", "END"), help="Faixa específica de partições (ex: --partition-range 1 135)")
    parser.add_argument("--strategy", choices=["block", "interleaved"], default="interleaved", help="Estratégia de divisão: 'block' (blocos sequenciais) ou 'interleaved' (alternado intercalado - padrão).")
    parser.add_argument("--merge", action="store_true", help="Funde todos os arquivos de saída dos trabalhadores no arquivo JSON principal com desduplicação.")

    args = parser.parse_args()

    if args.merge:
        merge_worker_files()
        return

    if args.worker < 1 or args.worker > args.total_workers:
        print(f"[!] Erro: --worker ({args.worker}) deve estar entre 1 e --total-workers ({args.total_workers}).")
        sys.exit(1)

    p_range = tuple(args.partition_range) if args.partition_range else None

    if args.total_workers > 1 or p_range:
        print(f"\n=======================================================")
        range_str = f" Faixa #{p_range[0]}-#{p_range[1]} |" if p_range else ""
        print(f" INICIANDO TRABALHADOR PARALELO [{args.worker}/{args.total_workers}] |{range_str} Modo: {args.strategy.upper()}")
        print(f"=======================================================\n")

    engine = ScraperEngine(
        worker_id=args.worker,
        total_workers=args.total_workers,
        partition_range=p_range,
        strategy=args.strategy
    )
    engine.run()


if __name__ == "__main__":
    main()
