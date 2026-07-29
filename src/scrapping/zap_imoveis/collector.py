import re
import time
import random
from typing import List
from bs4 import BeautifulSoup
from playwright.sync_api import Page

try:
    from .config import PAUSA_ENTRE_PAGINAS_SEC
except ImportError:
    from config import PAUSA_ENTRE_PAGINAS_SEC


class LinkCollector:
    """
    Coletor de Links de Anúncios.
    Responsável pela navegação paginada em uma partição de busca e extração das URLs dos imóveis.
    """
    def __init__(self, page: Page):
        self.page = page

    def set_page(self, page: Page) -> None:
        """Atualiza a referência da página ativa após reciclagem de contexto."""
        self.page = page

    def detect_partition_pages(self, partition_url: str) -> int:
        """Detecta o total de páginas disponíveis em uma partição de busca."""
        try:
            url_p1 = self._append_page_param(partition_url, 1)
            self.page.goto(url_p1, wait_until="domcontentloaded", timeout=45000)
            time.sleep(1.8)
            html = self.page.content()
            soup = BeautifulSoup(html, 'html.parser')

            for tag in soup.find_all(['h1', 'h2', 'p', 'span', 'div']):
                texto = tag.get_text(strip=True)
                match = re.search(r'([\d\.]+)\s*(apartamentos|im[oó]veis)', texto, re.IGNORECASE)
                if match:
                    total = int(match.group(1).replace('.', ''))
                    paginas = min((total // 24) + 1, 100)
                    return max(paginas, 1)

            nums = [int(m.group(1)) for a in soup.find_all('a', href=True) if (m := re.search(r'pagina=(\d+)', a.get('href') or ''))]
            if nums:
                return min(max(nums), 100)
        except Exception as e:
            print(f"   [!] Erro ao detectar páginas da partição: {e}")

        return 1

    def collect_links_from_page(self, partition_url: str, page_number: int) -> List[str]:
        """Extrai todos os links de imóveis de uma página específica da partição."""
        url = self._append_page_param(partition_url, page_number)
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=45000)
            time.sleep(random.uniform(*PAUSA_ENTRE_PAGINAS_SEC))
            html = self.page.content()
            soup = BeautifulSoup(html, 'html.parser')

            links = []
            for a in soup.find_all('a', href=True):
                href = a.get('href')
                if href and '/imovel/' in href and 'joao-pessoa' in href.lower():
                    url_completa = href if href.startswith('http') else f"https://www.zapimoveis.com.br{href}"
                    links.append(url_completa)

            return list(dict.fromkeys(links))
        except Exception as e:
            print(f"   [-] Erro ao ler página {page_number}: {e}")
            return []

    @staticmethod
    def _append_page_param(base_url: str, page_num: int) -> str:
        """Adiciona ou substitui o parâmetro `pagina` na URL."""
        if "pagina=" in base_url:
            return re.sub(r'pagina=\d+', f'pagina={page_num}', base_url)
        sep = "&" if "?" in base_url else "?"
        return f"{base_url}{sep}pagina={page_num}"
