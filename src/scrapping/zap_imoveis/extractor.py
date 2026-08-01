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

            # Título com fallbacks robustos (H1, meta tag, ou URL slug)
            dados['titulo'] = self._extract_title(soup, url)

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

            # Descrição do Imóvel
            dados['descricao_completa'] = self._extract_description(soup)

            # Preenchimento inteligente de lacunas (fallback via descrição)
            self._fill_missing_from_description(dados)

            # Validação de Qualidade Mínima:
            # Se não extraiu pelo menos preço, endereço ou título válido, considera a extração falha.
            tem_dados_validos = any([
                dados.get('preco_venda'),
                dados.get('endereco_completo'),
                dados.get('bairro'),
                dados.get('titulo') and dados.get('titulo') != "Imóvel Zap Imóveis"
            ])

            return dados if tem_dados_validos else None
        except Exception as e:
            print(f"  [!] Exceção prevenida no extrator de HTML ({url}): {e}")
            return None

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
        except Exception:
            dados.setdefault('preco_venda', None)

    def _extract_address(self, soup: BeautifulSoup, dados: Dict[str, Any]) -> None:
        try:
            tag_addr = soup.select_one(SELECTORS['endereco']) or soup.find('address')
            if tag_addr:
                end_texto = tag_addr.get_text(strip=True)
                dados['endereco_completo'] = end_texto
                
                # Regex flexível para extração do bairro
                match_bairro = re.search(r'(?:-\s*|,?\s*)([^,-]+),\s*(?:Jo[aã]o Pessoa|PB)', end_texto, re.I)
                if match_bairro:
                    dados['bairro'] = match_bairro.group(1).strip()
                else:
                    partes = [p.strip() for p in end_texto.split(',') if p.strip()]
                    if len(partes) >= 2:
                        dados['bairro'] = partes[-2].replace('João Pessoa', '').replace('PB', '').strip('- ')
                    else:
                        dados['bairro'] = None
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
                
                if 'suítes' in txt_norm or 'suite' in txt_norm:
                    dados['suites'] = re.sub(r'[^\d]', '', txt)
                elif 'quarto' in txt_norm:
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

    def _fill_missing_from_description(self, dados: Dict[str, Any]) -> None:
        """Preenche campos estruturados que ficaram nulos buscando padrões regex no texto da descrição."""
        desc = dados.get('descricao_completa', '')
        if not desc or desc == 'Descrição não encontrada.':
            return

        if not dados.get('quartos'):
            m = re.search(r'(\d+)\s*(?:quarto|dormit[óo]rio)', desc, re.I)
            if m: dados['quartos'] = m.group(1)

        if not dados.get('banheiros'):
            m = re.search(r'(\d+)\s*(?:banheiro|wc|su[íi]te)', desc, re.I)
            if m: dados['banheiros'] = m.group(1)

        if not dados.get('vagas'):
            m = re.search(r'(\d+)\s*(?:vaga|garagem|garagens)', desc, re.I)
            if m: dados['vagas'] = m.group(1)

        if not dados.get('area_util'):
            m = re.search(r'(\d+(?:[\.,]\d+)?)\s*(?:m²|m2|metros\s+quadrados)', desc, re.I)
            if m: dados['area_util'] = m.group(1).replace(',', '.')

    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extrai o título com seletores hierárquicos e fallback pela URL."""
        try:
            # 1. H1 ou Seletores CSS específicos do Zap Imóveis
            h1 = soup.find('h1') or soup.select_one("[data-testid='title'], h1[class*='title'], [class*='listing-title']")
            if h1 and h1.get_text(strip=True):
                return h1.get_text(strip=True)

            # 2. Meta Tag Open Graph (og:title)
            og_title = soup.find('meta', property='og:title') or soup.find('meta', attrs={'name': 'title'})
            if og_title and og_title.get('content'):
                title_content = og_title['content'].split('|')[0].strip()
                if title_content and len(title_content) > 5:
                    return title_content

            # 3. Fallback: Parse da URL (slug)
            if url:
                match = re.search(r'/imovel/([^/?]+)', url)
                if match:
                    slug = match.group(1)
                    slug_clean = re.sub(r'-id-\d+$', '', slug)
                    words = [w.capitalize() for w in slug_clean.split('-') if w]
                    return " ".join(words)
        except Exception:
            pass

        return "Imóvel Zap Imóveis"
