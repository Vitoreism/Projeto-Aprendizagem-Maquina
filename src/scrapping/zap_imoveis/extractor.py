import re
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup

try:
    from .config import SELECTORS
except ImportError:
    from config import SELECTORS


class PropertyExtractor:
    """
    Extrator de Atributos do Anúncio (Parser HTML).
    Responsável pela raspagem detalhada dos dados da página de um anúncio individual.
    Com tratamento defensivo completo contra falhas de estrutura DOM ou expressões regulares.
    """
    def __init__(self):
        pass

    def extract(self, html_content: str, url: str) -> Optional[Dict[str, Any]]:
        """Extrai todos os dados estruturados do HTML da página do imóvel."""
        if not html_content:
            return None

        dados: Dict[str, Any] = {'url_anuncio': url}

        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Título
            h1 = soup.find('h1')
            dados['titulo'] = h1.get_text(strip=True) if h1 else None

            # Endereço e Bairro
            self._extract_address(soup, dados)

            # Preço, Condomínio e IPTU
            self._extract_prices(soup, dados)

            # Características (Quartos, Banheiros, Vagas, Área)
            self._extract_amenities(soup, dados)

            # Detalhes Adicionais (Código e Anunciante)
            self._extract_details(soup, url, dados)

            # Comodidades
            self._extract_features(soup, dados)

            # Descrição Completa
            dados['descricao_completa'] = self._extract_description(soup)

            return dados
        except Exception as e:
            print(f"  [!] Exceção prevenida no extrator de HTML ({url}): {e}")
            return dados if dados.get('url_anuncio') else None

    def _extract_prices(self, soup: BeautifulSoup, dados: Dict[str, Any]) -> None:
        try:
            elementos_valor = soup.select(SELECTORS['valores'])
            if elementos_valor:
                venda_txt = elementos_valor[0].get_text(strip=True)
                dados['preco_venda'] = re.sub(r'R\$\s*', '', venda_txt).strip()
                
                for item in elementos_valor[1:]:
                    txt = item.get_text(strip=True)
                    parent = item.find_parent()
                    parent_txt = parent.get_text(strip=True).lower() if parent else ""
                    if 'condom' in parent_txt and 'condominio' not in dados:
                        dados['condominio'] = re.sub(r'R\$\s*', '', txt).replace('/mês', '').strip()
                    elif 'iptu' in parent_txt and 'iptu' not in dados:
                        dados['iptu'] = re.sub(r'R\$\s*', '', txt).strip()
            else:
                tag_preco = soup.select_one(SELECTORS['preco_backup']) or soup.select_one("[class*='price'] b")
                dados['preco_venda'] = re.sub(r'R\$\s*', '', tag_preco.get_text(strip=True)).strip() if tag_preco else None

            if 'condominio' not in dados:
                match_cond = re.search(r'Condom[íi]nio\s*R\$\s*([\d\.]+)', soup.get_text(), re.I)
                dados['condominio'] = match_cond.group(1) if match_cond else None
            if 'iptu' not in dados:
                match_iptu = re.search(r'IPTU\s*R\$\s*([\d\.]+)', soup.get_text(), re.I)
                dados['iptu'] = match_iptu.group(1) if match_iptu else None
        except Exception as e:
            dados.setdefault('preco_venda', None)

    def _extract_address(self, soup: BeautifulSoup, dados: Dict[str, Any]) -> None:
        try:
            tag_addr = soup.select_one(SELECTORS['endereco']) or soup.find('address')
            if tag_addr:
                end_texto = tag_addr.get_text(strip=True)
                dados['endereco_completo'] = end_texto
                
                match_bairro = re.search(r'-\s*([^,]+),\s*Jo[aã]o Pessoa', end_texto, re.I)
                if match_bairro:
                    dados['bairro'] = match_bairro.group(1).strip()
                else:
                    partes = end_texto.split('-')
                    dados['bairro'] = partes[-1].replace('João Pessoa', '').replace('PB', '').strip(', ') if len(partes) > 1 else None
            else:
                dados['endereco_completo'] = None
                dados['bairro'] = None
        except Exception:
            dados['endereco_completo'] = None
            dados['bairro'] = None

    def _extract_amenities(self, soup: BeautifulSoup, dados: Dict[str, Any]) -> None:
        try:
            items_amenities = soup.select(SELECTORS['amenities'])
            for item in items_amenities:
                txt = item.get_text(strip=True)
                txt_norm = txt.lower()
                
                if 'quarto' in txt_norm:
                    dados['quartos'] = re.sub(r'[^\d]', '', txt)
                elif 'banheiro' in txt_norm:
                    dados['banheiros'] = re.sub(r'[^\d]', '', txt)
                elif 'vaga' in txt_norm or 'garagem' in txt_norm:
                    dados['vagas'] = re.sub(r'[^\d]', '', txt)
                elif 'm2' in txt_norm or 'm²' in txt:
                    dados['area_util'] = re.sub(r'[^\d,\.]', '', txt).strip()

            h1_text = dados.get('titulo', '') or ''
            if 'quartos' not in dados:
                m_q = re.search(r'(\d+)\s*Quartos?', h1_text, re.I)
                if m_q: dados['quartos'] = m_q.group(1)
            if 'area_util' not in dados:
                m_a = re.search(r'(\d+)\s*m²', h1_text, re.I)
                if m_a: dados['area_util'] = m_a.group(1)
        except Exception:
            pass

    def _extract_details(self, soup: BeautifulSoup, url: str, dados: Dict[str, Any]) -> None:
        try:
            cod_match = re.search(r'id-(\d+)', url or '')
            if cod_match:
                dados['codigo_imovel'] = cod_match.group(1)
            else:
                cod_txt = soup.find(string=re.compile(r'C[oó]d\.?\s*no\s*Zap', re.I))
                if cod_txt:
                    match_code = re.search(r'\d{4,}', str(cod_txt))
                    dados['codigo_imovel'] = match_code.group(0) if match_code else None
                else:
                    dados['codigo_imovel'] = None

            anunc_tag = soup.select_one(SELECTORS['anunciante']) or soup.select_one("[class*='advertiser']")
            dados['anunciante'] = anunc_tag.get_text(strip=True) if anunc_tag else None
        except Exception:
            dados.setdefault('codigo_imovel', None)
            dados.setdefault('anunciante', None)

    def _extract_features(self, soup: BeautifulSoup, dados: Dict[str, Any]) -> None:
        try:
            comodidades = [li.get_text(strip=True) for li in soup.select(SELECTORS['comodidades_list']) if len(li.get_text(strip=True)) < 50]
            if comodidades:
                dados['comodidades_imovel'] = ", ".join(list(dict.fromkeys(comodidades)))
        except Exception:
            pass

    def _extract_description(self, soup: BeautifulSoup) -> str:
        try:
            desc_tag = soup.select_one(SELECTORS['descricao']) or soup.select_one("[class*='description']")
            return desc_tag.get_text(separator='\n', strip=True) if desc_tag else 'Descrição não encontrada.'
        except Exception:
            return 'Descrição não encontrada.'
