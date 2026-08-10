import os
import json
from typing import List, Dict, Any, Set, Optional

try:
    from .config import ARQUIVO_SAIDA
except ImportError:
    from config import ARQUIVO_SAIDA


class StorageManager:
    """
    Gerenciador de Armazenamento e Deduplicação Global.
    Responsável pela leitura incremental, gravação em lotes e garantia de registros únicos
    cruzando a base principal (11.800+ imóveis) e todas as saídas de trabalhadores paralelos.
    """
    def __init__(self, file_path: str = ARQUIVO_SAIDA):
        self.file_path = file_path
        self.collected_urls: Set[str] = set()
        self.collected_codes: Set[str] = set()
        self._load_existing_data()

    def _load_existing_data(self) -> None:
        """Carrega e indexa todas as bases de dados existentes (base principal + arquivos de todos os trabalhadores)."""
        import glob
        try:
            from .config import DIR_DADOS, ARQUIVO_SAIDA
        except ImportError:
            from config import DIR_DADOS, ARQUIVO_SAIDA

        # 1. Indexa a base principal acumulada (ex: os 11.800 imóveis salvos)
        if os.path.exists(ARQUIVO_SAIDA) and ARQUIVO_SAIDA != self.file_path:
            self._index_file(ARQUIVO_SAIDA, purge_duplicates=False)

        # 2. Indexa arquivos de todos os outros trabalhadores/partições (imoveis_joao_pessoa_zap_*.json)
        worker_files = glob.glob(os.path.join(DIR_DADOS, "imoveis_joao_pessoa_zap_*.json"))
        for wf in worker_files:
            if wf != self.file_path and wf != ARQUIVO_SAIDA:
                self._index_file(wf, purge_duplicates=False)

        # 3. Indexa e purga o arquivo de trabalho do worker atual
        self._index_file(self.file_path, purge_duplicates=True)

    def _index_file(self, target_path: str, purge_duplicates: bool = False) -> None:
        """Carrega e indexa URLs e códigos de um arquivo JSON específico."""
        if not os.path.exists(target_path):
            return

        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                records = json.load(f)

            if not isinstance(records, list):
                return

            valid_records = []
            modified = False

            for item in records:
                if isinstance(item, dict):
                    url = item.get('url_anuncio')
                    code = item.get('codigo_imovel')

                    is_empty = (
                        item.get('preco_venda') is None and
                        item.get('endereco_completo') is None and
                        item.get('anunciante') is None
                    )
                    if is_empty:
                        if purge_duplicates and target_path == self.file_path:
                            modified = True
                        continue

                    url_clean = url.strip() if url else None
                    code_clean = str(code).strip() if code else None

                    if purge_duplicates and target_path == self.file_path:
                        if url_clean and url_clean in self.collected_urls:
                            modified = True
                            continue
                        if code_clean and code_clean in self.collected_codes:
                            modified = True
                            continue

                    if url_clean:
                        self.collected_urls.add(url_clean)
                    if code_clean:
                        self.collected_codes.add(code_clean)

                    if purge_duplicates and target_path == self.file_path:
                        if not item.get('titulo') or item.get('titulo') == 'Sem título':
                            item['titulo'] = self._generate_title_from_url(url)
                            modified = True
                        valid_records.append(item)

            if purge_duplicates and modified and target_path == self.file_path:
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    json.dump(valid_records, f, ensure_ascii=False, indent=4)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [!] Alerta ao carregar arquivo '{os.path.basename(target_path)}': {e}")

    def refresh_indices(self) -> None:
        """Atualiza a memória de URLs/códigos lendo arquivos de saída do disco."""
        self._load_existing_data()

    @staticmethod
    def _generate_title_from_url(url: Optional[str]) -> str:
        if not url:
            return "Imóvel Zap Imóveis"
        import re
        match = re.search(r'/imovel/([^/?]+)', url)
        if match:
            slug = match.group(1)
            slug_clean = re.sub(r'-id-\d+$', '', slug)
            words = [w.capitalize() for w in slug_clean.split('-') if w]
            return " ".join(words)
        return "Imóvel Zap Imóveis"

    def is_already_collected(self, url: str, code: str = None) -> bool:
        """Verifica se uma URL ou código de imóvel já foi coletado previamente na base principal ou em qualquer worker."""
        if url:
            url_clean = url.strip()
            if url_clean in self.collected_urls:
                return True
            import re
            m = re.search(r'id-(\d+)', url_clean)
            if m and str(m.group(1)) in self.collected_codes:
                return True

        if code and str(code).strip() in self.collected_codes:
            return True
        return False

    def save_batch(self, batch_data: List[Dict[str, Any]]) -> int:
        """
        Filtra dados duplicados e salva incrementalmente no arquivo JSON do worker.
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
