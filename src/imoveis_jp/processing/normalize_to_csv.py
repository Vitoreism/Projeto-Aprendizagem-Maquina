# -*- coding: utf-8 -*-
"""Normaliza o JSON scrapado (chavesnamao) para os CSVs tabulares do projeto.

Entrada:  data/raw/imoveis_joao_pessoa.json  (snapshot bruto, versionado)
Saída:    data/processed/*.csv               (seed reprodutível e versionado)
Uso:      python -m imoveis_jp.processing.normalize_to_csv

Regras de mapeamento e decisões de curadoria: ver docs/scraping.md.
"""
import json, re, csv, statistics, os
from collections import defaultdict, Counter

from imoveis_jp import config

config.ensure_dirs()
SRC = config.ANUNCIOS_JSON
OUTDIR = config.PROCESSED

data = json.load(open(SRC, encoding="utf-8"))

# ── Restauração de acentos/nomes canônicos dos bairros (curadoria verificável) ──
CANON = {
    "manaira": "Manaíra", "jardim oceania": "Jardim Oceania", "cabo branco": "Cabo Branco",
    "tambau": "Tambaú", "bessa": "Bessa", "gramame": "Gramame", "aeroclube": "Aeroclube",
    "mucumagro": "Muçumagro", "paratibe": "Paratibe", "ernesto geisel": "Ernesto Geisel",
    "jardim cidade universitaria": "Jardim Cidade Universitária",
    "altiplano cabo branco": "Altiplano Cabo Branco", "miramar": "Miramar",
    "bancarios": "Bancários", "treze de maio": "Treze de Maio", "cristo redentor": "Cristo Redentor",
    "jardim veneza": "Jardim Veneza", "planalto boa esperanca": "Planalto Boa Esperança",
    "brisamar": "Brisamar", "ernani satiro": "Ernâni Sátiro", "jardim sao paulo": "Jardim São Paulo",
    "valentina de figueiredo": "Valentina de Figueiredo", "torre": "Torre",
}

def neigh(url):
    m = re.search(r"pb-joao-pessoa-(.+?)-(?:[\d.,]+m2-)?rs\d+", url.lower())
    if not m:
        return None
    slug = m.group(1).replace("-", " ").strip()
    return CANON.get(slug, slug.title())

def site_id(url):
    m = re.search(r"/id-(\d+)", url)
    return m.group(1) if m else None

def to_int_money(s):
    if not s: return None
    digits = re.sub(r"[^\d]", "", str(s).split(",")[0])  # descarta centavos após vírgula
    return int(digits) if digits else None

def to_float(s):
    if not s: return None
    s = str(s).replace(".", "").replace(",", ".") if str(s).count(",") else str(s)
    m = re.search(r"[\d.]+", s)
    if not m: return None
    try:
        return float(m.group())
    except ValueError:
        # Ex.: '58.045.100' — o anúncio lista várias áreas no mesmo campo.
        # Trata como ausente em vez de derrubar a normalização inteira.
        return None

def to_int(s):
    if s is None: return 0
    m = re.search(r"\d+", str(s))
    return int(m.group()) if m else 0

# ── Vocabulário controlado de comodidades (Doc 01 §3.9, estendido — decisão de curadoria) ──
# token cru (lower) -> (amenity_name, category)  |  None = descartar (é cômodo/acabamento, não amenity)
AMEN = {
    "piscina": ("pool","leisure"), "piscina infantil": ("pool","leisure"),
    "academia": ("gym","leisure"),
    "playground": ("playground","leisure"), "brinquedoteca": ("playground","leisure"),
    "salão de festas": ("party_hall","leisure"), "salão multiuso": ("party_hall","leisure"),
    "sala(s) de convenções": ("party_hall","leisure"),
    "salão de jogos": ("game_room","leisure"),
    "churrasqueira": ("bbq_area","leisure"), "churrasqueira coletiva": ("bbq_area","leisure"),
    "churrasqueira(s) coletiva(s)": ("bbq_area","leisure"),
    "espaço gourmet": ("gourmet_area","leisure"), "varanda gourmet": ("gourmet_area","leisure"),
    "cozinha gourmet": ("gourmet_area","leisure"),
    "sauna": ("sauna","leisure"),
    "quadra de esporte": ("sports_court","leisure"),
    "cinema": ("cinema","leisure"),
    "spa": ("spa","leisure"), "sala de massagem": ("spa","leisure"), "hidromassagem": ("spa","leisure"),
    "adega": ("wine_cellar","leisure"),
    "vista para o mar": ("sea_view","view"),
    "portaria 24h": ("security_24h","security"), "segurança 24h": ("security_24h","security"),
    "segurança monitorada": ("security_24h","security"), "sistema de câmeras": ("security_24h","security"),
    "circuito interno de tv": ("security_24h","security"), "alarme": ("security_24h","security"),
    "portão eletrônico": ("security_24h","security"),
    "portaria": ("concierge","security"), "guarita": ("concierge","security"),
    "bicicletário": ("bike_rack","convenience"),
    "coworking": ("coworking","convenience"), "escritório": ("coworking","convenience"),
    "lavanderia": ("laundry","convenience"), "lavanderia compartilhada": ("laundry","convenience"),
    "gerador de energia": ("generator","convenience"),
    "ar condicionado": ("air_conditioning","convenience"),
    "infraestrutura de ar condicionado": ("air_conditioning","convenience"),
    "ambiente climatizado": ("air_conditioning","convenience"), "ambiente climatizado ": ("air_conditioning","convenience"),
    "interfone": ("intercom","convenience"),
}
# tokens que sinalizam booleanos do Property (não viram nós de Amenity)
BOOL_ELEV = {"elevador(es)","elevador(es) de serviço"}
BOOL_FURN = {"mobiliado"}
BOOL_PET  = {"aceita pet","aceita animais","área para pets"}
BOOL_BALC = {"sacada","varanda"}

def desc_has(desc, *kw):
    d = (desc or "").lower()
    return any(k in d for k in kw)

props = []
amenity_edges = []       # (prop_id, amenity_name)
amenities_used = {}       # name -> category
price_m2_by_neigh = defaultdict(list)
dropped_tokens = Counter()

for o in data:
    pid = "prop-" + (site_id(o["url_anuncio"]) or str(len(props)+1))
    nb = neigh(o["url_anuncio"])
    price = to_int_money(o.get("preco_venda"))
    area = to_float(o.get("area_util")) or to_float(o.get("area_total"))

    # comodidades cruas
    tokens = []
    for k in ("comodidades_area_privativa","comodidades_area_comum"):
        if o.get(k):
            tokens += [t.strip().lower() for t in o[k].split(",")]

    has_elev = any(t in BOOL_ELEV for t in tokens) or desc_has(o.get("descricao_completa"), "elevador")
    furnished = any(t in BOOL_FURN for t in tokens) or desc_has(o.get("descricao_completa"), "mobiliado")
    pets = any(t in BOOL_PET for t in tokens) or desc_has(o.get("descricao_completa"), "aceita pet","aceita animais","pet friendly")
    balcony = any(t in BOOL_BALC for t in tokens) or desc_has(o.get("descricao_completa"), "varanda","sacada")

    for t in tokens:
        if t in AMEN:
            name, cat = AMEN[t]
            amenity_edges.append((pid, name))
            amenities_used[name] = cat
        elif t in BOOL_ELEV | BOOL_FURN | BOOL_PET | BOOL_BALC:
            pass  # vira booleano, não amenity
        else:
            dropped_tokens[t] += 1

    if price and area and area > 0 and 10000 < price < 20000000:
        price_m2_by_neigh[nb].append(price/area)

    props.append({
        "id": pid,
        "title": (o.get("titulo") or "").strip(),
        "subtype": "Apartment",
        "transaction": "sale",
        "price_sale": price if price else "",
        "price_rent": "",
        "area_m2": area if area else "",
        "bedrooms": to_int(o.get("quartos")),
        "bathrooms": to_int(o.get("banheiros")),
        "parking_spots": to_int(o.get("garagens")),
        "iptu_year": "",
        "condo_fee": "",
        "pets_allowed": str(pets).lower(),
        "accepts_financing": "",     # camada curada (regra 4: price <= program.max_property_value)
        "year_built": "",
        "floor": "",
        "has_elevator": str(has_elev).lower(),
        "has_balcony": str(balcony).lower(),
        "furnished": str(furnished).lower(),
        "neighborhood": nb or "",
        "condominium": "",
        "builder": "",
        "built_year": "",
    })

# dedupe de arestas de amenity
amenity_edges = sorted(set(amenity_edges))

# ── neighborhoods.csv: avg_price_m2_sale DERIVADO (mediana real); índices curados ficam VAZIOS ──
neigh_rows = []
for nb in sorted(nb for nb in price_m2_by_neigh if nb):  # None = bairro não identificado na URL
    vals = price_m2_by_neigh[nb]
    neigh_rows.append({
        "name": nb, "city": "João Pessoa",
        "security_index": "", "mobility_index": "", "infrastructure_index": "",
        "avg_price_m2_sale": round(statistics.median(vals)),  # derivado dos anúncios reais
        "avg_price_m2_rent": "", "appreciation_rate_yoy": "", "population": "", "profile_tags": "",
    })

# ── escrita ──
def write(name, header, rows):
    with open(os.path.join(OUTDIR, name), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader(); w.writerows(rows)

PROP_HDR = ["id","title","subtype","transaction","price_sale","price_rent","area_m2","bedrooms",
    "bathrooms","parking_spots","iptu_year","condo_fee","pets_allowed","accepts_financing",
    "year_built","floor","has_elevator","has_balcony","furnished","neighborhood","condominium","builder","built_year"]
write("properties.csv", PROP_HDR, props)

write("cities.csv", ["name","state"], [{"name":"João Pessoa","state":"PB"}])

NB_HDR = ["name","city","security_index","mobility_index","infrastructure_index",
    "avg_price_m2_sale","avg_price_m2_rent","appreciation_rate_yoy","population","profile_tags"]
write("neighborhoods.csv", NB_HDR, neigh_rows)

write("amenities.csv", ["name","category"],
      [{"name":n,"category":c} for n,c in sorted(amenities_used.items())])

write("rel_amenities.csv", ["owner_kind","owner_key","amenity_name"],
      [{"owner_kind":"property","owner_key":pid,"amenity_name":a} for pid,a in amenity_edges])

# ── relatório ──
print(f"properties.csv     : {len(props)} imóveis")
print(f"neighborhoods.csv  : {len(neigh_rows)} bairros (avg_price_m2_sale derivado; índices curados vazios)")
print(f"amenities.csv      : {len(amenities_used)} comodidades controladas")
print(f"rel_amenities.csv  : {len(amenity_edges)} arestas property->amenity")
print(f"\nBooleans: pets={sum(p['pets_allowed']=='true' for p in props)}, "
      f"elevador={sum(p['has_elevator']=='true' for p in props)}, "
      f"varanda={sum(p['has_balcony']=='true' for p in props)}, "
      f"mobiliado={sum(p['furnished']=='true' for p in props)}")
print(f"\nBairros com >=1 imóvel: {sum(1 for p in props if p['neighborhood'])}/{len(props)} (sem bairro: {sum(1 for p in props if not p['neighborhood'])})")
print("\nComodidades controladas (nome -> categoria):")
for n,c in sorted(amenities_used.items()): print(f"  {n}  ({c})")
print(f"\nTokens DESCARTADOS (cômodos/acabamentos, top 15):")
for t,n in dropped_tokens.most_common(15): print(f"  {n:3d}  {t}")
