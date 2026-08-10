import os
import sys
import time
import signal
from typing import Callable, Optional
from .config import PAUSE_FLAG_FILE


class ExecutionController:
    """
    Controlador de Execução do Scraper.
    Gerencia flags de pausa/retomada via arquivo `pause.flag` e interceptação graciosa de interrupção (Ctrl+C).
    """
    def __init__(self, pause_file: str = PAUSE_FLAG_FILE):
        self.pause_file = pause_file
        self.stop_requested: bool = False
        self.on_stop_callback: Optional[Callable] = None
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Registra interceptador para a tecla Ctrl+C (SIGINT)."""
        def handle_sigint(signum, frame):
            print("\n\n[!] Interrupção solicitada pelo usuário (Ctrl+C)!")
            print("[+] Finalizando lote atual e salvando estado de forma segura antes de sair...")
            self.stop_requested = True
            if self.on_stop_callback:
                try:
                    self.on_stop_callback()
                except Exception as e:
                    print(f"Erro no callback de encerramento: {e}")

        try:
            signal.signal(signal.SIGINT, handle_sigint)
        except (ValueError, OSError):
            pass

    def is_paused(self) -> bool:
        """Verifica se a flag de pausa existe no sistema de arquivos."""
        return os.path.exists(self.pause_file)

    def check_pause(self) -> None:
        """Se o scraper estiver pausado, congela a execução até que a flag seja removida ou o stop seja chamado."""
        if not self.is_paused():
            return

        print(f"\n[PAUSA DETECTADA] O arquivo '{os.path.basename(self.pause_file)}' foi encontrado.")
        print("   O scraper está em PAUSA. Remova o arquivo para continuar ou pressione Ctrl+C para encerrar.")

        while self.is_paused() and not self.stop_requested:
            time.sleep(2)

        if not self.stop_requested:
            print("[RETOMANDO] Arquivo de pausa removido. Continuando execução...\n")

    def should_stop(self) -> bool:
        """Retorna se o cancelamento/parada foi solicitado."""
        return self.stop_requested

    def pause(self) -> None:
        """Cria o arquivo de pausa."""
        with open(self.pause_file, 'w', encoding='utf-8') as f:
            f.write("paused")
        print(f"[PAUSA] Execução pausada. Arquivo '{self.pause_file}' criado.")

    def resume(self) -> None:
        """Remove o arquivo de pausa."""
        if os.path.exists(self.pause_file):
            os.remove(self.pause_file)
            print("[RETOMAR] Execução retomada.")
