import time
import random
from typing import List
from playwright.sync_api import sync_playwright

try:
    from .config import (
        USER_AGENT, EXTRA_HEADERS, TAMANHO_LOTE,
        PAUSA_ENTRE_IMOVEIS_SEC, ARQUIVO_SAIDA
    )
    from .storage import StorageManager
    from .url_builder import UrlStrategyBuilder
    from .collector import LinkCollector
    from .extractor import PropertyExtractor
    from .tracker import ProgressTracker
    from .controller import ExecutionController
    from .logger import LoggerManager
except ImportError:
    from config import (
        USER_AGENT, EXTRA_HEADERS, TAMANHO_LOTE,
        PAUSA_ENTRE_IMOVEIS_SEC, ARQUIVO_SAIDA
    )
    from storage import StorageManager
    from url_builder import UrlStrategyBuilder
    from collector import LinkCollector
    from extractor import PropertyExtractor
    from tracker import ProgressTracker
    from controller import ExecutionController
    from logger import LoggerManager


class ScraperEngine:
    """
    Orquestrador Principal da Raspagem (Engine).
    Aplica os princípios SOLID (SRP, DIP) agregando os módulos especializados
    com rastreabilidade e observabilidade completa via LoggerManager.
    """
    def __init__(
        self,
        storage: StorageManager = None,
        url_builder: UrlStrategyBuilder = None,
        tracker: ProgressTracker = None,
        controller: ExecutionController = None,
        logger: LoggerManager = None
    ):
        self.storage = storage or StorageManager()
        self.url_builder = url_builder or UrlStrategyBuilder()
        self.tracker = tracker or ProgressTracker()
        self.controller = controller or ExecutionController()
        self.logger = logger or LoggerManager()
        self.extractor = PropertyExtractor()

        # Configura o callback para salvamento seguro caso haja Ctrl+C
        self.controller.on_stop_callback = self._on_emergency_stop
        self.current_batch: List[dict] = []

    def _on_emergency_stop(self):
        """Callback de parada de emergência para salvar o lote pendente."""
        if self.current_batch:
            saved = self.storage.save_batch(self.current_batch)
            self.logger.log_checkpoint(saved, self.storage.get_total_collected())
            self.current_batch.clear()

    def run(self):
        """Executa a pipeline completa de raspagem por partições."""
        self.logger.log_phase("INICIALIZAÇÃO", "Carregando estado do scraper e base existente")
        self.logger.log(f"Arquivo de saída: {self.storage.file_path}")
        self.logger.log(f"Imóveis pré-existentes na base: {self.storage.get_total_collected()}")

        partitions = self.url_builder.build_partitions()
        self.logger.log(f"154 partições de busca geradas (Bairros x Faixas de Preço).")

        self.logger.log_phase("EXECUÇÃO", "Iniciando raspagem particionada das URLs e dados")
        self.tracker.start()

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=['--disable-blink-features=AutomationControlled', '--no-sandbox', '--disable-setuid-sandbox']
            )
            context = browser.new_context(
                user_agent=USER_AGENT,
                locale='pt-BR',
                viewport={'width': 1366, 'height': 768},
                extra_http_headers=EXTRA_HEADERS
            )
            page = context.new_page()
            collector = LinkCollector(page)

            total_processed = 0

            for p_idx, partition in enumerate(partitions, 1):
                if self.controller.should_stop():
                    self.logger.log("\n[!] Encerrando loop de partições (Parada solicitada).", level="WARN")
                    break

                self.controller.check_pause()

                self.logger.log_partition_start(p_idx, len(partitions), partition.label)

                total_pages = collector.detect_partition_pages(partition.url)
                print(f"   Páginas detectadas: ~{total_pages}")
                print("   [+] Coletando links das páginas:", end=" ", flush=True)

                partition_links = []
                for page_num in range(1, total_pages + 1):
                    if self.controller.should_stop():
                        break
                    self.controller.check_pause()

                    links = collector.collect_links_from_page(partition.url, page_num)
                    new_links = [l for l in links if not self.storage.is_already_collected(l)]
                    partition_links.extend(new_links)

                    print(f"P{page_num}(+{len(new_links)})", end=" ", flush=True)

                    if not links:
                        print(" [fim da partição]")
                        break

                print()
                unique_partition_links = list(dict.fromkeys(partition_links))
                self.logger.log_partition_result(partition.label, total_pages, len(unique_partition_links), len(partition_links))
                print(f"   [+] {len(unique_partition_links)} novos links para extrair nesta partição.")

                # Extrai dados dos imóveis da partição
                for l_idx, link in enumerate(unique_partition_links, 1):
                    if self.controller.should_stop():
                        break
                    self.controller.check_pause()

                    total_processed += 1
                    error_desc = None
                    try:
                        time.sleep(random.uniform(*PAUSA_ENTRE_IMOVEIS_SEC))
                        page.goto(link, wait_until="domcontentloaded", timeout=40000)
                        time.sleep(1.2)
                        html_content = page.content()
                        
                        dados = self.extractor.extract(html_content, link)
                        if dados:
                            self.current_batch.append(dados)
                            info_msg = dados.get('titulo', 'Sem título')[:65]
                            self.logger.log_extraction(l_idx, len(unique_partition_links), info_msg, True)
                        else:
                            info_msg = "[-] Falha ao parsear anúncio"
                            self.logger.log_extraction(l_idx, len(unique_partition_links), link, False, "Parse HTML nulo")

                    except Exception as e:
                        info_msg = f"[-] Erro de navegação: {e}"
                        self.logger.log_extraction(l_idx, len(unique_partition_links), link, False, str(e))

                    # Atualiza o rastreador e relógio ETA
                    self.tracker.update(
                        processed=total_processed,
                        saved=self.storage.get_total_collected(),
                        total_estimated=len(unique_partition_links) * len(partitions),
                        current_partition=partition.label
                    )
                    self.tracker.print_progress_bar(f"[{l_idx}/{len(unique_partition_links)}] {info_msg}")

                    # Salva em lote
                    if len(self.current_batch) >= TAMANHO_LOTE:
                        saved_count = self.storage.save_batch(self.current_batch)
                        self.logger.log_checkpoint(saved_count, self.storage.get_total_collected())
                        self.current_batch.clear()

            # Salva qualquer lote remanescente ao final
            if self.current_batch:
                saved_count = self.storage.save_batch(self.current_batch)
                self.logger.log_checkpoint(saved_count, self.storage.get_total_collected())
                self.current_batch.clear()

            browser.close()

        self.logger.log_phase("FINALIZAÇÃO", "Gerando relatórios e encerrando sessão")
        self.logger.generate_final_report()
