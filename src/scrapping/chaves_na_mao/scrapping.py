import requests
import gzip
import io
import re
import json
from typing import Dict
from bs4 import BeautifulSoup
import time
import random
from playwright.sync_api import sync_playwright

def get_links_apartamentos_venda_jp():
    sitemap_index_url = "https://www.chavesnamao.com.br/sitemap.xml" 
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1'
    }
    
    links_joao_pessoa = []
    
    print("Acessando o índice do Sitemap principal...")
    response = requests.get(sitemap_index_url, headers=headers)
    
    if response.status_code != 200:
        print(f"Erro ao acessar Sitemap. Status code: {response.status_code}")
        return []

    # 1. Extração rápida usando Regex
    sitemaps = re.findall(r'<loc>(.*?)</loc>', response.text)
    
    # 2. Ignora tudo que não for sitemap de venda
    sitemaps_venda = [s for s in sitemaps if 'sitemap-venda-imoveis' in s.lower()]
    print(f"Encontrados {len(sitemaps_venda)} sitemaps focados em vendas para processar.\n")
    
    for sitemap_url in sitemaps_venda:
        print(f"Lendo: {sitemap_url.split('/')[-1]}...", end=" ")
        
        try:
            resp_sub = requests.get(sitemap_url, headers=headers)
            if resp_sub.status_code != 200:
                print("Erro de conexão.")
                continue
                
            # Descompacta o .gz em memória
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
                
                # ==========================================
                # 3. FILTROS RIGOROSOS DE BUSCA
                # ==========================================
                tem_joao_pessoa = "joao-pessoa" in link_teste
                tem_apartamento = "apartamento" in link_teste
                tem_venda = "venda" in link_teste
                
                # Trava de segurança extra
                tem_aluguel = "aluguel" in link_teste or "locacao" in link_teste
                
                # Só aprova se atender a todas as suas condições exatas
                if tem_joao_pessoa and tem_apartamento and tem_venda and not tem_aluguel:
                    links_joao_pessoa.append(link_original)
                    encontrados_neste_sitemap += 1
                    
            print(f"Achou {encontrados_neste_sitemap} imóveis. (Total: {len(links_joao_pessoa)})")
                    
        except Exception as e:
            print(f"Erro: {e}")
            
    # Remove qualquer possível duplicata
    links_unicos = list(set(links_joao_pessoa))
    
    print(f"\n✅ EXTRAÇÃO CONCLUÍDA! Total de links únicos garantidos: {len(links_unicos)}")
    return links_unicos


def extrair_dados_do_anuncio(pagina, url):
    """
    Usa o Playwright (reutilizando a página) para renderizar o JavaScript antes de extrair os dados.
    """
    try:
        print(f"Carregando dados estruturados: {url}")
        pagina.goto(url, wait_until="networkidle", timeout=30000)
        
        html_renderizado = pagina.content()
        soup = BeautifulSoup(html_renderizado, 'html.parser')
        dados_imovel = {'url_anuncio': url}
        
        # 1. DADOS BÁSICOS
        titulo_tag = soup.find('h1')
        dados_imovel['titulo'] = titulo_tag.get_text(strip=True) if titulo_tag else None

        preco_tag = soup.select_one("b span[class*='clamp']") or soup.select_one("span[class*='clamp']")
        dados_imovel['preco_venda'] = preco_tag.get_text(strip=True) if preco_tag else None

        endereco_tag = soup.find('address')
        endereco = endereco_tag.get_text(strip=True) if endereco_tag else ""
        dados_imovel['endereco_completo'] = re.sub(r'^Endereço\s+indisponível', '', endereco, flags=re.IGNORECASE).strip()

        # 2. CARACTERÍSTICAS NUMÉRICAS
        ul_tag = soup.find('ul', class_=lambda c: c and 'listContent' in c)
        if ul_tag:
            itens_lista = ul_tag.find_all('li', role='listitem')
            for item in itens_lista:
                p_tag = item.find('p', attrs={'aria-label': True})
                if p_tag:
                    chave_crua = p_tag.get('aria-label').lower()
                    chave_formatada = chave_crua.replace('-', '_').replace(' ', '_')
                    valor = p_tag.get_text(strip=True)
                    dados_imovel[chave_formatada] = valor

        # 3. EXTRAS E COMODIDADES
        container_principal = soup.select_one("div[class*='optionalItemsContainer']")
        if container_principal:
            secoes = container_principal.find_all('span', recursive=False)
            for secao in secoes:
                titulo_secao_tag = secao.find('b')
                if titulo_secao_tag:
                    nome_categoria = titulo_secao_tag.get_text(strip=True).lower().replace(' ', '_').replace('á', 'a')
                    itens_lista = secao.select('ul li')
                    lista_de_itens = [item.get_text(strip=True) for item in itens_lista]
                    if lista_de_itens:
                        dados_imovel[f"comodidades_{nome_categoria}"] = ", ".join(lista_de_itens)

        return dados_imovel

    except Exception as e:
        print(f"Erro ao processar dados de {url}: {e}")
        return None


def extrair_descricao_anuncio(pagina, url):
    """
    Extrai a descrição do imóvel utilizando uma página do Playwright já aberta.
    """
    try:
        print(f"Carregando descrição: {url}")
        # Se a página já estiver no mesmo link, o Playwright otimiza, 
        # mas mantemos o goto para garantir consistência.
        pagina.goto(url, wait_until="networkidle", timeout=30000)
        
        html = pagina.content()
        soup = BeautifulSoup(html, 'html.parser')
        
        descricao_tag = soup.find('p', attrs={'aria-label': 'descrição'}) or \
                        soup.find('div', class_=lambda c: c and 'description' in c.lower())
        
        if descricao_tag:
            return descricao_tag.get_text(separator="\n", strip=True)
        
        return "Descrição não encontrada."

    except Exception as e:
        print(f"Erro ao extrair descrição de {url}: {e}")
        return None


def mapear_todas_caracteristicas(lista_links_teste):
    """
    Roda uma amostra de links abrindo o navegador apenas UMA VEZ e 
    reutilizando a mesma página para todas as requisições.
    """
    todas_as_chaves_do_dicionario = set()
    todas_as_comodidades_comuns = set()
    todas_as_comodidades_privativas = set()
    
    print(f"\n🔍 Mapeando schema de {len(lista_links_teste)} links...")
    
    # Inicia o motor do Playwright fora do loop (uma única vez)
    with sync_playwright() as p:
        navegador = p.chromium.launch(headless=True)
        pagina = navegador.new_page() # Criamos a "aba" compartilhada
        
        for i, link in enumerate(lista_links_teste):
            print(f"\n[{i+1}/{len(lista_links_teste)}] Analisando...")
            
            # Extrai os dados estruturados usando a aba compartilhada
            dados = extrair_dados_do_anuncio(pagina, link)
            
            if dados:
                todas_as_chaves_do_dicionario.update(dados.keys())
                
                if 'comodidades_area_comum' in dados:
                    itens_comuns = dados['comodidades_area_comum'].split(', ')
                    todas_as_comodidades_comuns.update(itens_comuns)
                    
                if 'comodidades_area_privativa' in dados:
                    itens_privativos = dados['comodidades_area_privativa'].split(', ')
                    todas_as_comodidades_privativas.update(itens_privativos)
                    
            time.sleep(random.uniform(0.5, 1.5))
            
        navegador.close() # Fecha o navegador apenas ao terminar tudo
        
    return {
        "chaves_gerais": sorted(list(todas_as_chaves_do_dicionario)),
        "comodidades_comuns": sorted(list(todas_as_comodidades_comuns)),
        "comodidades_privativas": sorted(list(todas_as_comodidades_privativas))
    }


if __name__ == "__main__":
   
   # links_finais = get_links_apartamentos_venda_jp()
    #print(links_finais[:50])
    
   # print("\n\n")
    # Exemplo de teste chamando a função de mapeamento otimizada:
    #link_teste = ["https://www.chavesnamao.com.br/imovel/apartamento-a-venda-4-quartos-com-garagem-pb-joao-pessoa-altiplano-cabo-branco-220m2-RS2400000/id-32288846/"]
    #print(mapear_todas_caracteristicas(link_teste))
    link_teste = "https://www.chavesnamao.com.br/imovel/apartamento-a-venda-4-quartos-com-garagem-pb-joao-pessoa-altiplano-cabo-branco-220m2-RS2400000/id-32288846/"
    
    # Inicia o Playwright e abre a "aba" (pagina)
    with sync_playwright() as p:
        navegador = p.chromium.launch(headless=True)
        pagina = navegador.new_page()
        
        descricao = extrair_descricao_anuncio(pagina, link_teste)
        
        print("\n--- DESCRIÇÃO EXTRAÍDA ---")
        print(descricao)
        
        navegador.close()