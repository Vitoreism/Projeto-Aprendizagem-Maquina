import os
import sys

# Diretórios e Arquivos
DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_SAIDA = os.path.join(DIR_ATUAL, "imoveis_joao_pessoa_zap.json")
PAUSE_FLAG_FILE = os.path.join(DIR_ATUAL, "pause.flag")

# Configurações do Scraper
BASE_LISTAGEM_URL = "https://www.zapimoveis.com.br/venda/apartamentos/pb+joao-pessoa/"
TAMANHO_LOTE = 50
TIMEOUT_NAVEGACAO_MS = 45000
PAUSA_ENTRE_PAGINAS_SEC = (1.2, 2.2)
PAUSA_ENTRE_IMOVEIS_SEC = (1.0, 2.0)

# Anti-Bot & Network Headers
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
EXTRA_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
    'Sec-Ch-Ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
}

# Principais Bairros de João Pessoa (PB) para subdivisão de busca
BAIRROS_JOAO_PESSOA = [
    "bessa",
    "manaira",
    "tambau",
    "cabo-branco",
    "altiplano-cabo-branco",
    "jardim-oceania",
    "aeroclube",
    "intermares",
    "portal-do-sol",
    "mangabeira",
    "miramar",
    "estados",
    "expedicionarios",
    "torre",
    "bairro-dos-ipês",
    "tambauzinho",
    "bessa-mar",
    "cristo-redentor",
    "bessa",
    "bayeux",
    "cabedelo"
]

# Faixas de preço para refinamento de busca (em Reais)
FAIXAS_PRECO = [
    {"min": None, "max": 250000, "label": "Até R$ 250k"},
    {"min": 250001, "max": 400000, "label": "R$ 250k a R$ 400k"},
    {"min": 400001, "max": 600000, "label": "R$ 400k a R$ 600k"},
    {"min": 600001, "max": 900000, "label": "R$ 600k a R$ 900k"},
    {"min": 900001, "max": 1300000, "label": "R$ 900k a R$ 1.3M"},
    {"min": 1300001, "max": 2000000, "label": "R$ 1.3M a R$ 2.0M"},
    {"min": 2000001, "max": None, "label": "Acima de R$ 2.0M"}
]

# Seletores CSS Validados
SELECTORS = {
    'valores': 'p.value-item__value',
    'preco_backup': "[data-testid='price-info-value']",
    'endereco': "[data-testid='location-address']",
    'amenities': 'span.amenities-item-text',
    'descricao': 'p.description__content--text',
    'anunciante': "[data-testid='advertiser-name']",
    'comodidades_list': "div[class*='amenities'] li, ul[class*='feature'] li"
}
