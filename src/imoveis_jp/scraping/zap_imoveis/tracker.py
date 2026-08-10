import time
from typing import Optional
from collections import deque


class ProgressTracker:
    """
    Rastreia e exibe no terminal métricas de progresso com ETA duplo (Global e Partição):
    - ETA Global: Baseado no tempo médio por partição (janela deslizante)
    - ETA Partição: Baseado na velocidade recente dos últimos N imóveis
    - Taxa de velocidade em imóveis por minuto
    """
    def __init__(self):
        self.start_time: Optional[float] = None
        self.total_partitions: int = 0
        self.current_partition_idx: int = 0
        self.current_partition_label: str = ""
        self.current_partition_total_links: int = 0
        self.current_partition_processed: int = 0
        self.total_processed_items: int = 0
        self.saved_items: int = 0

        # Janelas deslizantes para cálculo de velocidade
        self.partition_times: deque = deque(maxlen=10)
        self.item_timestamps: deque = deque(maxlen=30)
        self.last_partition_start_time: Optional[float] = None

    def start(self, total_partitions: int = 0) -> None:
        """Inicia o cronômetro do rastreador."""
        self.start_time = time.time()
        self.total_partitions = total_partitions

    def start_partition(self, p_idx: int, total_partitions: int, label: str, total_links: int) -> None:
        """Notifica o início de uma nova partição."""
        self.current_partition_idx = p_idx
        self.total_partitions = total_partitions
        self.current_partition_label = label
        self.current_partition_total_links = total_links
        self.current_partition_processed = 0
        self.last_partition_start_time = time.time()

    def finish_partition(self) -> None:
        """Notifica a conclusão de uma partição."""
        if self.last_partition_start_time:
            duration = time.time() - self.last_partition_start_time
            self.partition_times.append(duration)

    def record_item_processed(self, saved_items: int) -> None:
        """Registra o processamento de um imóvel individual."""
        self.total_processed_items += 1
        self.current_partition_processed += 1
        self.saved_items = saved_items
        self.item_timestamps.append(time.time())

    def get_elapsed_seconds(self) -> float:
        """Retorna o tempo decorrido total em segundos."""
        if not self.start_time:
            return 0.0
        return time.time() - self.start_time

    def get_items_per_minute(self) -> float:
        """Calcula a velocidade de processamento por minuto usando a janela dos últimos imóveis."""
        if len(self.item_timestamps) >= 2:
            time_window = self.item_timestamps[-1] - self.item_timestamps[0]
            items_window = len(self.item_timestamps) - 1
            if time_window > 0:
                return (items_window / time_window) * 60.0

        elapsed = self.get_elapsed_seconds()
        if elapsed < 1.0 or self.total_processed_items == 0:
            return 0.0
        return (self.total_processed_items / elapsed) * 60.0

    def get_partition_eta(self) -> str:
        """Calcula o ETA para término da partição atual."""
        if self.current_partition_total_links == 0 or self.current_partition_processed >= self.current_partition_total_links:
            return "00m 00s"

        remaining = self.current_partition_total_links - self.current_partition_processed
        rate = self.get_items_per_minute()
        if rate <= 0:
            return "--:--:--"

        eta_seconds = int((remaining / rate) * 60)
        return self._format_seconds(eta_seconds)

    def get_global_eta(self) -> str:
        """Calcula o ETA global baseado no tempo médio por partição."""
        if self.total_partitions == 0 or self.current_partition_idx >= self.total_partitions:
            return "00h 00m 00s"

        remaining_partitions = self.total_partitions - self.current_partition_idx

        if len(self.partition_times) > 0:
            avg_partition_sec = sum(self.partition_times) / len(self.partition_times)
        else:
            elapsed = self.get_elapsed_seconds()
            if self.current_partition_idx > 1 and elapsed > 0:
                avg_partition_sec = elapsed / (self.current_partition_idx - 1)
            else:
                avg_partition_sec = 180.0

        rate = self.get_items_per_minute()
        current_part_remaining_sec = 0.0
        if rate > 0 and self.current_partition_total_links > self.current_partition_processed:
            current_part_remaining_sec = ((self.current_partition_total_links - self.current_partition_processed) / rate) * 60.0

        total_eta_seconds = int(current_part_remaining_sec + remaining_partitions * avg_partition_sec)
        return self._format_seconds(total_eta_seconds)

    def get_elapsed_formatted(self) -> str:
        """Retorna o tempo decorrido total formatado."""
        return self._format_seconds(int(self.get_elapsed_seconds()))

    @staticmethod
    def _format_seconds(seconds: int) -> str:
        if seconds < 0:
            seconds = 0
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours > 0:
            return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
        return f"{minutes:02d}m {secs:02d}s"

    def print_progress_bar(self, current_item_info: str = "") -> None:
        """Imprime a linha de status visual do terminal com o relógio de progresso e ETA duplo."""
        elapsed = self.get_elapsed_formatted()
        global_eta = self.get_global_eta()
        part_eta = self.get_partition_eta()
        rate = self.get_items_per_minute()

        part_pct = f"{(self.current_partition_processed / self.current_partition_total_links * 100):.1f}%" if self.current_partition_total_links > 0 else "?%"

        status_msg = (
            f"[DECORRIDO: {elapsed} | ETA Global: {global_eta} | Partição {self.current_partition_idx}/{self.total_partitions}: ETA {part_eta}] "
            f"Progresso Partição: {self.current_partition_processed}/{self.current_partition_total_links} ({part_pct}) | "
            f"Taxa: {rate:.1f} imoveis/min | Salvos Base: {self.saved_items}"
        )
        print(f"\n{status_msg}")
        if current_item_info:
            print(f"   -> {current_item_info[:85]}")

