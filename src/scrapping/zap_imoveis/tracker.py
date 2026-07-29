import time
import math
from typing import Optional


class ProgressTracker:
    """
    Rastreia e exibe no terminal métricas em tempo real:
    - Imóveis processados / Salvos
    - Tempo decorrido
    - Taxa de velocidade (imóveis/minuto)
    - Tempo Restante Estimado (ETA Clock)
    """
    def __init__(self):
        self.start_time: Optional[float] = None
        self.processed_items: int = 0
        self.saved_items: int = 0
        self.total_estimated: int = 0
        self.current_partition: str = ""

    def start(self) -> None:
        """Inicia o cronômetro do rastreador."""
        self.start_time = time.time()

    def update(self, processed: int, saved: int, total_estimated: int = 0, current_partition: str = "") -> None:
        """Atualiza a contagem dos itens e estimativas."""
        self.processed_items = processed
        self.saved_items = saved
        if total_estimated > 0:
            self.total_estimated = total_estimated
        if current_partition:
            self.current_partition = current_partition

    def get_elapsed_seconds(self) -> float:
        """Retorna o tempo decorrido em segundos."""
        if not self.start_time:
            return 0.0
        return time.time() - self.start_time

    def get_items_per_minute(self) -> float:
        """Calcula a velocidade atual de processamento por minuto."""
        elapsed = self.get_elapsed_seconds()
        if elapsed < 1.0 or self.processed_items == 0:
            return 0.0
        return (self.processed_items / elapsed) * 60.0

    def get_eta_formatted(self) -> str:
        """Calcula o ETA (Tempo Restante Estimado) formatado HH:MM:SS."""
        rate = self.get_items_per_minute()
        if rate <= 0 or self.total_estimated <= self.processed_items:
            return "--:--:--"

        remaining_items = self.total_estimated - self.processed_items
        eta_minutes = remaining_items / rate
        eta_seconds = int(eta_minutes * 60)

        hours = eta_seconds // 3600
        minutes = (eta_seconds % 3600) // 60
        seconds = eta_seconds % 60

        if hours > 0:
            return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
        return f"{minutes:02d}m {seconds:02d}s"

    def get_elapsed_formatted(self) -> str:
        """Retorna o tempo decorrido formatado HH:MM:SS."""
        elapsed = int(self.get_elapsed_seconds())
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        if hours > 0:
            return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
        return f"{minutes:02d}m {seconds:02d}s"

    def print_progress_bar(self, current_item_info: str = "") -> None:
        """Imprime a linha de status visual do terminal com o relógio de progresso e ETA."""
        elapsed = self.get_elapsed_formatted()
        eta = self.get_eta_formatted()
        rate = self.get_items_per_minute()
        total_str = str(self.total_estimated) if self.total_estimated > 0 else "?"
        pct = f"{(self.processed_items / self.total_estimated * 100):.1f}%" if self.total_estimated > 0 else "?%"

        status_msg = (
            f"[TEMPO: {elapsed} | ETA: {eta}] "
            f"Progresso: {self.processed_items}/{total_str} ({pct}) | "
            f"Taxa: {rate:.1f} imoveis/min | Salvos: {self.saved_items}"
        )
        if self.current_partition:
            status_msg += f" | Particao: {self.current_partition}"
        
        print(f"\n{status_msg}")
        if current_item_info:
            print(f"   -> {current_item_info[:80]}")
