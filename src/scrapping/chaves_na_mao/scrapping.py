import requests
import gzip
import io
import re

def get_links_apartamentos_venda_jp():
    sitemap_index_url = "https://www.chavesnamao.com.br/sitemap.xml" 
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
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

if __name__ == "__main__":
    links_finais = get_links_apartamentos_venda_jp()
    print(links_finais[:50])