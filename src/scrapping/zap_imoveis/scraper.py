import os
import time
import random
from typing import List, Tuple
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page

try:
    from .config import (
        USER_AGENT, EXTRA_HEADERS, TAMANHO_LOTE,
        PAUSA_ENTRE_IMOVEIS_SEC, ARQUIVO_SAIDA,
        INTERVALO_PAUSA_LONGA_IMOVEIS, PAUSA_LONGA_DURACAO_SEC,
        STORAGE_STATE_FILE, CONTEXT_RECYCLE_EVERY, MAX_RETRIES_PER_URL
    )
    from .storage import StorageManager
    from .url_builder import UrlStrategyBuilder
    from .collector import LinkCollector
    from .extractor import PropertyExtractor
    from .tracker import ProgressTracker
    from .controller import ExecutionController
    from .logger import LoggerManager
    from .verifier import PageIntegrityVerifier, PageStatus
    from .rate_limiter import AdaptiveRateLimiter
except ImportError:
    from config import (
        USER_AGENT, EXTRA_HEADERS, TAMANHO_LOTE,
        PAUSA_ENTRE_IMOVEIS_SEC, ARQUIVO_SAIDA,
        INTERVALO_PAUSA_LONGA_IMOVEIS, PAUSA_LONGA_DURACAO_SEC,
        STORAGE_STATE_FILE, CONTEXT_RECYCLE_EVERY, MAX_RETRIES_PER_URL
    )
    from storage import StorageManager
    from url_builder import UrlStrategyBuilder
    from collector import LinkCollector
    from extractor import PropertyExtractor
    from tracker import ProgressTracker
    from controller import ExecutionController
    from logger import LoggerManager
    from verifier import PageIntegrityVerifier, PageStatus
    from rate_limiter import AdaptiveRateLimiter


class ScraperEngine:
    """
    Orquestrador Principal da Raspagem (Engine).
    Aplica os princípios SOLID (SRP, DIP) agregando os módulos especializados
    com resiliência enterprise (persistencia de sessão, reciclagem de contexto,
    rate limiter adaptativo e circuit breaker com retentativas).
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
        self.rate_limiter = AdaptiveRateLimiter()

        self.active_context: BrowserContext = None
        self.current_batch: List[dict] = []

        # Configura o callback para salvamento seguro caso haja Ctrl+C
        self.controller.on_stop_callback = self._on_emergency_stop

    def _on_emergency_stop(self):
        """Callback de parada de emergência para salvar o lote pendente e persistir a sessão."""
        if self.active_context:
            self._save_session_state(self.active_context)
        if self.current_batch:
            saved = self.storage.save_batch(self.current_batch)
            self.logger.log_checkpoint(saved, self.storage.get_total_collected())
            self.current_batch.clear()

    def _create_browser_context(self, browser: Browser) -> Tuple[BrowserContext, Page]:
        """Cria um novo contexto Playwright com persistência de sessão e evasão stealth."""
        context_kwargs = {
            'user_agent': USER_AGENT,
            'locale': 'pt-BR',
            'viewport': {'width': 1366, 'height': 768},
            'extra_http_headers': EXTRA_HEADERS
        }
        
        if os.path.exists(STORAGE_STATE_FILE):
            try:
                context_kwargs['storage_state'] = STORAGE_STATE_FILE
            except Exception:
                pass

        context = browser.new_context(**context_kwargs)
        page = context.new_page()
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        self.active_context = context
        return context, page

    def _save_session_state(self, context: BrowserContext) -> None:
        """Persiste os cookies e localStorage da sessão ativa no disco."""
        try:
            if context:
                context.storage_state(path=STORAGE_STATE_FILE)
        except Exception as e:
            self.logger.log(f"Aviso ao salvar sessão: {e}", level="WARN", print_console=False)

    def _recycle_context(
        self,
        browser: Browser,
        current_context: BrowserContext,
        collector: LinkCollector,
        reason: str = "Rotina Periódica"
    ) -> Tuple[BrowserContext, Page]:
        """Recicla o contexto Playwright renovando memória e sessões estragadas."""
        self._save_session_state(current_context)
        try:
            current_context.close()
        except Exception:
            pass

        new_context, new_page = self._create_browser_context(browser)
        collector.set_page(new_page)
        self.logger.log_context_recycle(reason)
        return new_context, new_page

    def run(self):
        """Executa a pipeline completa de raspagem por partições com resiliência enterprise."""
        self.logger.log_phase("INICIALIZAÇÃO", "Carregando estado do scraper, sessão e base existente")
        self.logger.log(f"Arquivo de saída: {self.storage.file_path}")
        self.logger.log(f"Imóveis pré-existentes na base: {self.storage.get_total_collected()}")

        partitions = self.url_builder.build_partitions()
        self.logger.log(f"154 partições de busca geradas (Bairros x Faixas de Preço).")

        self.logger.log_phase("EXECUÇÃO", "Iniciando raspagem particionada das URLs e dados com resiliência")
        self.tracker.start()

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=['--disable-blink-features=AutomationControlled', '--no-sandbox', '--disable-setuid-sandbox']
            )

            context, page = self._create_browser_context(browser)
            collector = LinkCollector(page)

            total_processed = 0
            requests_since_recycle = 0

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

                # Extrai dados dos imóveis da partição com Fila de Retentativas e Circuit Breaker
                for l_idx, link in enumerate(unique_partition_links, 1):
                    if self.controller.should_stop():
                        break
                    self.controller.check_pause()

                    requests_since_recycle += 1
                    if requests_since_recycle >= CONTEXT_RECYCLE_EVERY:
                        context, page = self._recycle_context(
                            browser, context, collector, f"Ciclo periódico ({requests_since_recycle} reqs)"
                        )
                        requests_since_recycle = 0

                    total_processed += 1
                    info_msg = ""

                    # Loop de Retentativa com Circuit Breaker para a URL
                    for attempt in range(1, MAX_RETRIES_PER_URL + 2):
                        if self.controller.should_stop():
                            break

                        # 1. Delay adaptativo com monitoramento de taxa de erro
                        self._apply_human_delay(total_processed)

                        try:
                            response = page.goto(link, wait_until="domcontentloaded", timeout=40000)
                            
                            # 2. Simula interação humana (scroll de leitura)
                            self._simulate_human_interaction(page)

                            status_code = response.status if response else 0
                            page_title = page.title() or ""
                            html_content = page.content() if response else ""

                            # 3. Verificação de Integridade da Página (Anti-Bot / Soft-Ban / Layout)
                            p_status, msg_reason = PageIntegrityVerifier.verify(status_code, page_title, html_content)

                            if p_status in (PageStatus.CLOUDFLARE_CHALLENGE, PageStatus.SOFT_BAN):
                                self.rate_limiter.record_result(False)
                                self.logger.log_resilience_event("Desafio Anti-Bot", f"URL: {link} -> {msg_reason}")

                                if attempt <= MAX_RETRIES_PER_URL:
                                    cooldown = 20.0 * attempt
                                    print(f"\n   [🛡 CIRCUIT BREAKER] Bloqueio detectado. Cooldown de {cooldown:.0f}s + Reciclagem de contexto (Tentativa {attempt}/{MAX_RETRIES_PER_URL})...")
                                    time.sleep(cooldown)
                                    context, page = self._recycle_context(
                                        browser, context, collector, f"Mitigação de Bloqueio em {link}"
                                    )
                                    requests_since_recycle = 0
                                    continue
                                else:
                                    info_msg = f"[-] {msg_reason} (Tentativas esgotadas)"
                                    self.logger.log_extraction(l_idx, len(unique_partition_links), link, False, msg_reason)
                                    break

                            if p_status == PageStatus.NOT_FOUND:
                                self.rate_limiter.record_result(True)
                                info_msg = "[-] Imóvel indisponível (HTTP 404)"
                                self.logger.log_extraction(l_idx, len(unique_partition_links), link, False, "HTTP 404")
                                break

                            # 4. Extração dos dados do imóvel
                            dados = self.extractor.extract(html_content, link)
                            if dados:
                                self.current_batch.append(dados)
                                self.rate_limiter.record_result(True)
                                
                                if attempt > 1:
                                    self.logger.log_resilience_event("Retentativa Sucesso", f"Sucesso na tentativa {attempt} para {link}")
                                
                                info_msg = (dados.get('titulo') or 'Sem título')[:65]
                                self.logger.log_extraction(l_idx, len(unique_partition_links), info_msg, True)
                            else:
                                self.rate_limiter.record_result(False)
                                info_msg = "[-] Parse HTML nulo"
                                self.logger.log_extraction(l_idx, len(unique_partition_links), link, False, "Parse HTML nulo")
                            
                            break  # Sucesso ou processamento concluído sem necessidade de retentar

                        except Exception as e:
                            self.rate_limiter.record_result(False)
                            if attempt <= MAX_RETRIES_PER_URL:
                                print(f"\n   [!] Erro de navegação ({e}). Retentando em 5s ({attempt}/{MAX_RETRIES_PER_URL})...")
                                time.sleep(5.0)
                                continue
                            else:
                                info_msg = f"[-] Erro de navegação: {e}"
                                self.logger.log_extraction(l_idx, len(unique_partition_links), link, False, str(e))
                                break

                    # Atualiza o rastreador e relógio ETA
                    self.tracker.update(
                        processed=total_processed,
                        saved=self.storage.get_total_collected(),
                        total_estimated=len(unique_partition_links) * len(partitions),
                        current_partition=partition.label
                    )
                    
                    status_prefix = " [DESACELERADO]" if self.rate_limiter.is_throttled() else ""
                    self.tracker.print_progress_bar(f"[{l_idx}/{len(unique_partition_links)}]{status_prefix} {info_msg}")

                    # Salva em lote
                    if len(self.current_batch) >= TAMANHO_LOTE:
                        saved_count = self.storage.save_batch(self.current_batch)
                        self._save_session_state(context)
                        self.logger.log_checkpoint(saved_count, self.storage.get_total_collected())
                        self.current_batch.clear()

            # Salva lote remanescente e sessão final
            self._save_session_state(context)
            if self.current_batch:
                saved_count = self.storage.save_batch(self.current_batch)
                self.logger.log_checkpoint(saved_count, self.storage.get_total_collected())
                self.current_batch.clear()

            browser.close()

        self.logger.log_phase("FINALIZAÇÃO", "Gerando relatórios e encerrando sessão")
        self.logger.generate_final_report()

    def _apply_human_delay(self, index: int) -> None:
        """Aplica pausas adaptativas com base na saúde da janela deslizante e pausas de leitura humanas."""
        # Utiliza o AdaptiveRateLimiter para calcular o delay dinâmico
        self.rate_limiter.wait(PAUSA_ENTRE_IMOVEIS_SEC)

        # Pausa longa de leitura a cada N imóveis
        break_trigger = random.randint(*INTERVALO_PAUSA_LONGA_IMOVEIS)
        if index > 0 and index % break_trigger == 0:
            long_pause = random.uniform(*PAUSA_LONGA_DURACAO_SEC)
            print(f"\n   [☕ PAUSA DE LEITURA] Simulando navegação humana por {long_pause:.1f}s...")
            time.sleep(long_pause)

    def _simulate_human_interaction(self, page) -> None:
        """Simula comportamento de navegação humana (scroll suave de página e pequenas pausas)."""
        try:
            scroll_amount = random.randint(250, 550)
            page.evaluate(f"window.scrollBy(0, {scroll_amount})")
            time.sleep(random.uniform(0.5, 1.1))
        except Exception:
            pass


if __name__ == "__main__":
    engine = ScraperEngine()
    engine.run()
