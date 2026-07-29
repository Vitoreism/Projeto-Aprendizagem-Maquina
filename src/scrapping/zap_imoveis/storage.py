import os
import json
from typing import List, Dict, Any, Set

try:
    from .config import ARQUIVO_SAIDA
except ImportError:
    from config import ARQUIVO_SAIDA


class StorageManager:
    """
    Gerenciador de Armazenamento e Deduplicação.
    Responsável pela leitura incremental, gravação em lotes e garantia de registros únicos.
    """
    def __init__(self, file_path: str = ARQUIVO_SAIDA):
        self.file_path = file_path
        self.collected_urls: Set[str] = set()
        self.collected_codes: Set[str] = set()
        self._load_existing_data()

    def _load_existing_data(self) -> None:
        """Carrega os dados existentes para popular os índices de deduplicação."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                    for item in records:
                        if isinstance(item, dict):
                            url = item.get('url_anuncio')
                            code = item.get('codigo_imovel')
                            if url:
                                self.collected_urls.add(url.strip())
                            if code:
                                self.collected_codes.add(str(code).strip())
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [!] Alerta ao carregar arquivo de saída existente: {e}")

    def is_already_collected(self, url: str, code: str = None) -> bool:
        """Verifica se uma URL ou código de imóvel já foi coletado previamente."""
        if url and url.strip() in self.collected_urls:
            return True
        if code and str(code).strip() in self.collected_codes:
            return True
        return False

    def save_batch(self, batch_data: List[Dict[str, Any]]) -> int:
        """
        Filtra dados duplicados e salva incrementalmente no arquivo JSON.
        Retorna o número de novos registros gravados nesta chamada.
        """
        if not batch_data:
            return 0

        existing_data = []
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing_data = []

        new_records = []
        for item in batch_data:
            url = item.get('url_anuncio', '')
            code = item.get('codigo_imovel')

            if url and url in self.collected_urls:
                continue
            if code and str(code) in self.collected_codes:
                continue

            new_records.append(item)
            if url:
                self.collected_urls.add(url)
            if code:
                self.collected_codes.add(str(code))

        if new_records:
            existing_data.extend(new_records)
            try:
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"  [!] Erro ao gravar lote no arquivo JSON: {e}")
                return 0

        return len(new_records)

    def get_total_collected(self) -> int:
        """Retorna o número total de imóveis únicos armazenados."""
        return len(self.collected_urls)
