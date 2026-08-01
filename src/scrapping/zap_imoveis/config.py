import os
import sys

# Diretórios e Arquivos
DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_SAIDA = os.path.join(DIR_ATUAL, "imoveis_joao_pessoa_zap.json")
PAUSE_FLAG_FILE = os.path.join(DIR_ATUAL, "pause.flag")
STORAGE_STATE_FILE = os.path.join(DIR_ATUAL, "session_state.json")

# Configurações de Resiliência e Ciclo de Vida do Browser
CONTEXT_RECYCLE_EVERY = 80          # Recicla o contexto Playwright a cada N requisições
MAX_RETRIES_PER_URL = 2             # Tentativas de retentativa para URLs bloqueadas
SLIDING_WINDOW_SIZE = 20            # Janela de monitoramento da taxa de bloqueio
BLOCK_RATE_THRESHOLD = 0.15         # Limiar de erro para acionar cadência lentificada (15%)

# Configurações do Scraper e Delays Humanizados
BASE_LISTAGEM_URL = "https://www.zapimoveis.com.br/venda/apartamentos/pb+joao-pessoa/"
TAMANHO_LOTE = 50
TIMEOUT_NAVEGACAO_MS = 45000
PAUSA_ENTRE_PAGINAS_SEC = (2.5, 4.5)
PAUSA_ENTRE_IMOVEIS_SEC = (2.0, 4.5)
INTERVALO_PAUSA_LONGA_IMOVEIS = (8, 15)  # A cada N imóveis, faz uma pausa humana maior
PAUSA_LONGA_DURACAO_SEC = (7.0, 14.0)   # Duração da pausa de leitura humana

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
# Slugs validados empiricamente via HTTP contra o Zap Imóveis (HTTP 404 = inválido)
BAIRROS_JOAO_PESSOA = [
    # --- Bairros costeiros (alta densidade de anúncios) ---
    "bessa",
    "manaira",
    "tambau",
    "cabo-branco",
    "altiplano-cabo-branco",
    "aeroclube",
    "tambauzinho",
    "brisamar",

    # --- Bairros interiores validados ---
    "portal-do-sol",
    "mangabeira",
    "miramar",
    "estados",
    "expedicionarios",
    "torre",
    "ipes",               # Era "bairro-dos-ipês" (404) → corrigido empiricamente
    "cristo-redentor",
    "bancarios",
    "agua-fria",
    "gramame",
    "castelo-branco",
    "pedro-gondim",
    "centro",
    "jaguaribe",
    "cuia",
    "funcionarios",
    "rangel",
    "grotao",
    "penha",

    # REMOVIDOS (HTTP 404 - slug inválido no Zap Imóveis):
    # "jardim-oceania"    → 404 (propriedades capturadas via outras partições)
    # "intermares"        → 404 em todas as variantes testadas
    # "bessa-mar"         → 404 (não é bairro indexado separadamente)
    # "bayeux"            → cidade diferente de João Pessoa
    # "cabedelo"          → cidade diferente de João Pessoa
    # "bessa" (2ª vez)    → duplicata removida
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
