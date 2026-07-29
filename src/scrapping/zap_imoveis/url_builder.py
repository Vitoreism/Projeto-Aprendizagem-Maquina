from typing import List, Dict, Any

try:
    from .config import BASE_LISTAGEM_URL, BAIRROS_JOAO_PESSOA, FAIXAS_PRECO
except ImportError:
    from config import BASE_LISTAGEM_URL, BAIRROS_JOAO_PESSOA, FAIXAS_PRECO


class SearchPartition:
    """Representa uma sub-partição de busca (ex: Bairro X + Faixa de Preço Y)."""
    def __init__(self, label: str, url: str):
        self.label = label
        self.url = url

    def __repr__(self):
        return f"<SearchPartition label='{self.label}' url='{self.url}'>"


class UrlStrategyBuilder:
    """
    Construtor de Estratégias de Busca Paginada.
    Subdivide a busca geral por Bairros e Faixas de Preço para contornar o limite de 100 páginas do ZAP.
    """
    def __init__(self, base_url: str = BASE_LISTAGEM_URL):
        self.base_url = base_url.rstrip('/')

    def build_partitions(self) -> List[SearchPartition]:
        """Gera a lista completa de partições de busca subdivididas."""
        partitions: List[SearchPartition] = []

        # 1. Busca Geral por Faixas de Preço
        for faixa in FAIXAS_PRECO:
            params = self._format_price_params(faixa["min"], faixa["max"])
            url = f"{self.base_url}/?{params}" if params else self.base_url
            label = f"João Pessoa (Geral) - {faixa['label']}"
            partitions.append(SearchPartition(label, url))

        # 2. Busca Subdividida por Bairro + Faixa de Preço
        for bairro in BAIRROS_JOAO_PESSOA:
            bairro_slug = bairro.strip().lower()
            bairro_base_url = f"https://www.zapimoveis.com.br/venda/apartamentos/pb+joao-pessoa+{bairro_slug}"
            
            for faixa in FAIXAS_PRECO:
                params = self._format_price_params(faixa["min"], faixa["max"])
                url = f"{bairro_base_url}/?{params}" if params else bairro_base_url
                label = f"{bairro_slug.capitalize()} - {faixa['label']}"
                partitions.append(SearchPartition(label, url))

        return partitions

    @staticmethod
    def _format_price_params(min_val: int = None, max_val: int = None) -> str:
        """Formata os parâmetros de filtro de preço na URL."""
        parts = []
        if min_val is not None:
            parts.append(f"preco-minimo={min_val}")
        if max_val is not None:
            parts.append(f"preco-maximo={max_val}")
        return "&".join(parts)
