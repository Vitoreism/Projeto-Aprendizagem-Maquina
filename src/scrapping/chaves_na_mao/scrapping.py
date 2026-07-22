import requests
import gzip
import io
import os
import re
import json
import pandas as pd
from bs4 import BeautifulSoup
import time
import random
from playwright.sync_api import sync_playwright

class ScraperChavesNaMao:
    def __init__(self):
        self.sitemap_index_url = "https://www.chavesnamao.com.br/sitemap.xml" 
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        self.pagina = None
        self.url_atual = None
        self.soup = None

    def get_links_apartamentos_venda_jp(self):
        links_joao_pessoa = []
        print("Acessando o índice do Sitemap principal...")
        response = requests.get(self.sitemap_index_url, headers=self.headers)
        
        if response.status_code != 200:
            print(f"Erro ao acessar Sitemap. Status code: {response.status_code}")
            return []

        sitemaps = re.findall(r'<loc>(.*?)</loc>', response.text)
        sitemaps_venda = [s for s in sitemaps if 'sitemap-venda-imoveis' in s.lower()]
        print(f"Encontrados {len(sitemaps_venda)} sitemaps focados em vendas para processar.\n")
        
        for sitemap_url in sitemaps_venda:
            print(f"Lendo: {sitemap_url.split('/')[-1]}...", end=" ")
            try:
                resp_sub = requests.get(sitemap_url, headers=self.headers)
                if resp_sub.status_code != 200:
                    print("Erro de conexão.")
                    continue
                    
                if sitemap_url.endswith('.gz'):
                    f = io.BytesIO(resp_sub.content)
                    with gzip.GzipFile(fileobj=f) as gz:
                        xml_content = gz.read().decode('utf-8', errors='ignore')
                else:
                    xml_content = resp_sub.text
                    
                urls_imoveis = re.findall(r'<loc>(.*?)</loc>', xml_content)
                encontrados_neste_sitemap = 0
                
                for link_original in urls_imoveis:
                    link_teste = link_original.lower()
                    if ("joao-pessoa" in link_teste and "apartamento" in link_teste and 
                        "venda" in link_teste and not ("aluguel" in link_teste or "locacao" in link_teste)):
                        links_joao_pessoa.append(link_original)
                        encontrados_neste_sitemap += 1
                        
                print(f"Achou {encontrados_neste_sitemap} imóveis. (Total: {len(links_joao_pessoa)})")
                        
            except Exception as e:
                print(f"Erro: {e}")
                
        links_unicos = list(set(links_joao_pessoa))
        print(f"\n✅ EXTRAÇÃO CONCLUÍDA! Total de links únicos garantidos: {len(links_unicos)}")
        return links_unicos

    def _carregar_html_da_pagina(self):
        """
        NOVA FUNÇÃO: Carrega a página e cria o BeautifulSoup UMA ÚNICA VEZ.
        Isso salva processamento e banda de internet.
        """
        try:
            self.pagina.goto(self.url_atual, wait_until="networkidle", timeout=30000)
            html_renderizado = self.pagina.content()
            self.soup = BeautifulSoup(html_renderizado, 'html.parser')
            return True
        except Exception as e:
            print(f"Erro ao carregar a página {self.url_atual}: {e}")
            self.soup = None
            return False

    def extrair_dados_do_anuncio(self):
        """
        Agora essa função não faz requisição web. Ela apenas lê o self.soup que já está pronto!
        """
        if not self.soup: return None
        
        dados_imovel = {'url_anuncio': self.url_atual}
        
        # 1. DADOS BÁSICOS
        titulo_tag = self.soup.find('h1')
        dados_imovel['titulo'] = titulo_tag.get_text(strip=True) if titulo_tag else None

        preco_tag = self.soup.select_one("b span[class*='clamp']") or self.soup.select_one("span[class*='clamp']")
        if preco_tag:
            dados_imovel['preco_venda'] = preco_tag.get_text(strip=True).replace("R$", "").strip()
        else:
            dados_imovel['preco_venda'] = None

        endereco_tag = self.soup.find('address')
        endereco = endereco_tag.get_text(strip=True) if endereco_tag else ""
        dados_imovel['endereco_completo'] = re.sub(r'^Endereço\s+indisponível', '', endereco, flags=re.IGNORECASE).strip()

        # 2. CARACTERÍSTICAS NUMÉRICAS
        ul_tag = self.soup.find('ul', class_=lambda c: c and 'listContent' in c)
        if ul_tag:
            for item in ul_tag.find_all('li', role='listitem'):
                p_tag = item.find('p', attrs={'aria-label': True})
                if p_tag:
                    chave_crua = p_tag.get('aria-label').lower()
                    chave_formatada = chave_crua.replace('-', '_').replace(' ', '_')
                    
                    texto_bruto = p_tag.get_text(strip=True)
                    valor_limpo = re.sub(rf"^{re.escape(chave_crua)}", "", texto_bruto, flags=re.IGNORECASE).strip()
                    
                    if not valor_limpo or valor_limpo == texto_bruto:
                        valor_limpo = re.sub(r"^(área útil|área total|quartos|suítes|banheiros|garagens)\s*", "", texto_bruto, flags=re.IGNORECASE).strip()

                    valor_final = valor_limpo if valor_limpo else texto_bruto

                    if "area" in chave_formatada:
                        valor_final = re.sub(r"m²", "", valor_final, flags=re.IGNORECASE).strip()

                    dados_imovel[chave_formatada] = valor_final

        # 3. EXTRAS E COMODIDADES
        container_principal = self.soup.select_one("div[class*='optionalItemsContainer']")
        if container_principal:
            for secao in container_principal.find_all('span', recursive=False):
                titulo_secao_tag = secao.find('b')
                if titulo_secao_tag:
                    nome_categoria = titulo_secao_tag.get_text(strip=True).lower().replace(' ', '_').replace('á', 'a')
                    itens_lista = secao.select('ul li')
                    lista_de_itens = [item.get_text(strip=True) for item in itens_lista]
                    if lista_de_itens:
                        dados_imovel[f"comodidades_{nome_categoria}"] = ", ".join(lista_de_itens)

        return dados_imovel

    def extrair_descricao_anuncio(self):
        """
        Também não faz mais requisição web. Apenas busca a descrição no self.soup.
        """
        if not self.soup: return "Descrição não encontrada."
        
        descricao_tag = self.soup.find('p', attrs={'aria-label': 'descrição'}) or \
                        self.soup.find('div', class_=lambda c: c and 'description' in c.lower())
        
        if descricao_tag:
            return descricao_tag.get_text(separator="\n", strip=True)
        
        return "Descrição não encontrada."

    def main(self):
        print("🚀 Iniciando pipeline OOP de extração de imóveis - Chaves na Mão (João Pessoa)\n")
        
        links_finais = self.get_links_apartamentos_venda_jp()
        
        if not links_finais:
            print("❌ Nenhum link encontrado. Encerrando o pipeline.")
            return

        print(f"\n⚙️ Iniciando extração de dados para {len(links_finais)} imóveis...")
        
        # Alterado de volta para .json tradicional
        arquivo_saida = "imoveis_joao_pessoa.json"
        lote_dados = []
        
        with sync_playwright() as p:
            navegador = p.chromium.launch(headless=True)
            
            # ATRIBUI A PÁGINA AO ATRIBUTO DA CLASSE!
            self.pagina = navegador.new_page()
            
            for i, link in enumerate(links_finais):
                print(f"[{i+1}/{len(links_finais)}] Processando: {link}")
                
                # ATUALIZA O ESTADO DO ROBÔ PARA O LINK ATUAL
                self.url_atual = link
                
                # CARREGA A PÁGINA E GERA O SOUP (UMA ÚNICA VEZ)
                sucesso = self._carregar_html_da_pagina()
                
                if sucesso:
                    # Agora os métodos chamam apenas o self internamente
                    dados = self.extrair_dados_do_anuncio()
                    
                    if dados:
                        descricao = self.extrair_descricao_anuncio()
                        dados['descricao_completa'] = descricao
                        lote_dados.append(dados)
                
                # Checkpoint para salvar no JSON tradicional a cada 50 imóveis
                if (i + 1) % 50 == 0 or (i + 1) == len(links_finais):
                    if lote_dados:
                        dados_totais = []
                        
                        # Se o arquivo já existir, carrega a lista atual de dentro dele
                        if os.path.exists(arquivo_saida):
                            with open(arquivo_saida, "r", encoding="utf-8") as f:
                                try:
                                    dados_totais = json.load(f)
                                except json.JSONDecodeError:
                                    dados_totais = [] # Proteção caso o arquivo comece vazio ou corrompido
                        
                        # Adiciona os novos 50 imóveis na lista
                        dados_totais.extend(lote_dados)
                        
                        # Sobrescreve o arquivo com a lista atualizada e formatação bonita (indent=4)
                        with open(arquivo_saida, "w", encoding="utf-8") as f:
                            json.dump(dados_totais, f, ensure_ascii=False, indent=4)
                                
                        print(f"💾 Lote salvo com sucesso no arquivo '{arquivo_saida}'.")
                        lote_dados = [] # Limpa a memória do lote para os próximos 50
                
                time.sleep(random.uniform(0.5, 1.5))
                
            navegador.close()
            
        print(f"\n🎉 Pipeline concluído com sucesso! Todos os dados salvos em '{arquivo_saida}'.")

if __name__ == "__main__":
    scraper = ScraperChavesNaMao()
    scraper.main()