import random
import time
from collections import deque
from typing import Tuple

try:
    from .config import SLIDING_WINDOW_SIZE, BLOCK_RATE_THRESHOLD
except ImportError:
    from config import SLIDING_WINDOW_SIZE, BLOCK_RATE_THRESHOLD


class AdaptiveRateLimiter:
    """
    Controlador Adaptativo de Cadência (Sliding Window Rate Limiter).
    Monitora o histórico recente de requisições e ajusta dinamicamente os delays
    para evitar surtos de bloqueio anti-bot.
    """
    def __init__(
        self,
        window_size: int = SLIDING_WINDOW_SIZE,
        threshold: float = BLOCK_RATE_THRESHOLD
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.history = deque(maxlen=window_size)

    def record_result(self, success: bool) -> None:
        """Registra o resultado (Sucesso=True, Bloqueio/Erro=False) da última requisição."""
        self.history.append(1 if success else 0)

    @property
    def failure_rate(self) -> float:
        """Retorna a taxa atual de erros/bloqueios na janela deslizante (0.0 a 1.0)."""
        if not self.history:
            return 0.0
        failures = self.history.count(0)
        return failures / len(self.history)

    def is_throttled(self) -> bool:
        """Indica se o scraper está operando sob regime de mitigação de bloqueio."""
        return len(self.history) >= 5 and self.failure_rate > self.threshold

    def calculate_delay(self, base_range: Tuple[float, float]) -> float:
        """
        Calcula o tempo de espera ideal com base no status de saúde da janela deslizante.
        
        Args:
            base_range: Faixa (min_sec, max_sec) configurada padrão.
            
        Returns:
            float: Segundos a aguardar antes da próxima requisição.
        """
        min_sec, max_sec = base_range
        rate = self.failure_rate

        if rate <= self.threshold:
            # Cadência normal com jitter gaussiano leve
            base_sleep = random.uniform(min_sec, max_sec)
            jitter = random.gauss(0, 0.3)
            return max(min_sec, base_sleep + jitter)
        else:
            # Cadência desacelerada adaptativa (Multiplicador proporcional à taxa de falha)
            penalty_factor = 1.0 + (rate * 3.5)  # Ex: 25% falhas -> 1.87x delay
            min_sec_penalized = min_sec * penalty_factor
            max_sec_penalized = max_sec * penalty_factor
            
            actual_sleep = random.uniform(min_sec_penalized, max_sec_penalized)
            return round(actual_sleep, 2)

    def wait(self, base_range: Tuple[float, float]) -> float:
        """Aplica o delay diretamente via time.sleep e retorna os segundos dormidos."""
        sleep_dur = self.calculate_delay(base_range)
        time.sleep(sleep_dur)
        return sleep_dur
