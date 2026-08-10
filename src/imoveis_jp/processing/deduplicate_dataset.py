# -*- coding: utf-8 -*-
"""
modulo de deduplicacao cruzada de dados entre portais imobiliarios (issue #9)
identifica anuncios duplicados entre chaves na mao e zapimoveis e realiza a unificacao sem perda de informacao
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from imoveis_jp import config


def normalizar_texto(texto: Any) -> str:
    if pd.isna(texto) or not texto:
        return ""
    s = str(texto).strip().lower()
    subst = {
        "á": "a", "à": "a", "ã": "a", "â": "a", "ä": "a",
        "é": "e", "ê": "e", "ë": "e",
        "í": "i", "î": "i", "ï": "i",
        "ó": "o", "ô": "o", "õ": "o", "ö": "o",
        "ú": "u", "û": "u", "ü": "u",
        "ç": "c", "-": " ", "_": " ", "/": " ", ",": " "
    }
    for orig, sub in subst.items():
        s = s.replace(orig, sub)

    s = re.sub(r"\s+", " ", s).strip()
    return s


#: os 64 bairros oficiais de joao pessoa. esta e a lista fechada: nenhum outro
#: valor pode sair de extrair_bairro, alem de 'nao_informado'.
#:
#: o neighborhoods.csv do projeto NAO serve de referencia aqui -- ele mistura
#: bairro oficial com loteamento, conjunto e ate praia de outro municipio
#: (Camboinha e Cabedelo, Carapibus e Conde), e usa nomes que nao sao os
#: oficiais: 'Altiplano Cabo Branco' onde o bairro se chama Altiplano,
#: 'Jose Americo De Almeida' onde se chama Jose Americo.
BAIRROS_OFICIAIS = (
    "Aeroclube", "Agua Fria", "Altiplano", "Alto do Ceu", "Alto do Mateus",
    "Anatolia", "Bancarios", "Barra de Gramame", "Bessa", "Brisamar", "Cabo Branco",
    "Castelo Branco", "Centro", "Cidade dos Colibris", "Costa do Sol", "Costa e Silva",
    "Cristo Redentor", "Cruz das Armas", "Cuia", "Distrito Industrial", "Ernani Satiro",
    "Ernesto Geisel", "Estados", "Expedicionarios", "Funcionarios", "Gramame", "Grotao",
    "Ilha do Bispo", "Industrias", "Ipes", "Jaguaribe", "Jardim Cidade Universitaria",
    "Jardim Oceania", "Jardim Sao Paulo", "Jardim Veneza", "Joao Agripino",
    "Joao Paulo II", "Jose Americo", "Manaira", "Mandacaru", "Mangabeira", "Miramar",
    "Mucumagro", "Mumbaba", "Mussure", "Oitizeiro", "Padre Ze", "Paratibe",
    "Pedro Gondim", "Penha", "Planalto da Boa Esperanca", "Ponta do Seixas",
    "Portal do Sol", "Roger", "Sao Jose", "Tambau", "Tambauzinho", "Tambia", "Torre",
    "Treze de Maio", "Trincheiras", "Valentina de Figueiredo", "Varadouro", "Varjao",
)

#: nomes que os anuncios usam e que designam um bairro oficial sob outro rotulo.
#: cada linha aqui e uma afirmacao verificavel sobre o mesmo lugar, nao um chute
#: de vizinhanca -- localidade que eu nao consiga confirmar vai para
#: 'nao_informado', porque inventar o bairro mais proximo repetiria o defeito
#: que esta funcao acabou de corrigir.
ALIASES_BAIRRO = {
    # o bairro se chama Altiplano; 'Altiplano Cabo Branco' e o nome comercial.
    # precisa vir com os 3 tokens para vencer 'cabo branco' na especificidade.
    "altiplano cabo branco": "altiplano",
    # 'Varjao (Rangel)' -- os dois nomes designam o mesmo bairro.
    "rangel": "varjao",
    # variantes de escrita do nome oficial
    "planalto boa esperanca": "planalto_da_boa_esperanca",
    "planalto": "planalto_da_boa_esperanca",
    "jose americo de almeida": "jose_americo",
    "jardim 13 de maio": "treze_de_maio",
    "13 de maio": "treze_de_maio",
    "conjunto valentina figueredo i": "valentina_de_figueiredo",
    "valentina figueredo": "valentina_de_figueiredo",
    "valentina": "valentina_de_figueiredo",
    "bairro das industrias": "industrias",
    "cidade universitaria": "jardim_cidade_universitaria",
    "conjunto pedro gondim": "pedro_gondim",
    "geisel": "ernesto_geisel",
    "colibris": "cidade_dos_colibris",
    "seixas": "ponta_do_seixas",
    # loteamento dentro de Gramame, nao bairro proprio
    "loteamento quintas de gramame": "gramame",
    "quintas de gramame": "gramame",
}

#: artigos e conjuncoes nao distinguem bairro: 'valentina figueiredo' e
#: 'valentina de figueiredo' sao o mesmo lugar.
_ARTIGOS = frozenset({"de", "do", "da", "dos", "das", "e"})

#: cauda de cidade/estado, que nunca e bairro. 'joao pessoa' sai como par
#: exato para nao derrubar 'joao paulo ii' nem 'joao agripino'.
_NAO_BAIRRO = frozenset({"joao pessoa", "pb", "paraiba", "brasil", "brazil", "s n", "sn"})


def _normalizar_nome(texto: Any) -> str:
    """Minusculas sem acento, com a pontuacao virando separador de palavra."""
    if texto is None or (not isinstance(texto, str) and pd.isna(texto)):
        return ""
    s = normalizar_texto(texto)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]", " ", s)).strip()


def _tokens(texto: Any) -> frozenset:
    return frozenset(_normalizar_nome(texto).split()) - _ARTIGOS


def _slug(nome: str) -> str:
    return _normalizar_nome(nome).replace(" ", "_")


#: os unicos valores que extrair_bairro pode devolver, alem de 'nao_informado'.
CANONICOS = frozenset(_slug(n) for n in BAIRROS_OFICIAIS)

_desconhecidos = sorted(set(ALIASES_BAIRRO.values()) - CANONICOS)
if _desconhecidos:  # erro de digitacao no mapa de apelidos, nao dado ruim
    raise ValueError(f"ALIASES_BAIRRO aponta para bairro inexistente: {_desconhecidos}")

#: indice por conjunto de tokens, com apelidos e nomes oficiais no mesmo espaco.
_POR_TOKENS: Dict[frozenset, str] = {_tokens(n): _slug(n) for n in BAIRROS_OFICIAIS}
_POR_TOKENS.update({_tokens(a): c for a, c in ALIASES_BAIRRO.items()})

#: do mais especifico para o menos. essa ordenacao E a correcao do bug antigo:
#: a lista escrita a mao tinha 'cabo branco' antes de 'altiplano', entao
#: "Altiplano Cabo Branco" casava com Cabo Branco -- 511 anuncios de um bairro
#: com preco/m2 mediano 26% menor entravam no outro.
#:
#: os apelidos entram nesta ordenacao de proposito. o bairro se chama Altiplano
#: (1 token), entao sozinho ele PERDERIA de 'cabo branco' (2 tokens) no endereco
#: "Altiplano Cabo Branco" e o bug voltaria. O apelido de 3 tokens vence os dois.
_POR_ESPECIFICIDADE = sorted(_POR_TOKENS.items(), key=lambda kv: -len(kv[0]))


def casar_bairro(texto: Any) -> Optional[str]:
    """Nome canonico do bairro contido em `texto`, ou None.

    Nunca inventa: o que nao casa com a lista oficial devolve None e o chamador
    resolve como 'nao_informado'. Era exatamente esse o defeito antigo -- o
    fallback pegava a primeira palavra com mais de 3 letras do endereco e
    produzia 'avenida', 'doutor', 'professor', 'maria' como se fossem bairros.

    Localidade real que nao e bairro oficial (loteamento, conjunto, praia de
    outro municipio) tambem devolve None, a menos que esteja em ALIASES_BAIRRO
    com o bairro que a contem. Chutar o bairro mais proximo seria repetir o
    defeito com outra roupa.
    """
    alvo = _normalizar_nome(texto)
    if not alvo or alvo in _NAO_BAIRRO:
        return None

    tokens = frozenset(alvo.split()) - _ARTIGOS
    if not tokens:
        return None

    exato = _POR_TOKENS.get(tokens)
    if exato:
        return exato

    # o nome precisa caber inteiro dentro do texto ('38 bessa' -> bessa).
    for tokens_conhecidos, canonico in _POR_ESPECIFICIDADE:
        if tokens_conhecidos <= tokens:
            return canonico

    return None


def _campos_do_endereco(endereco: Any) -> List[str]:
    """Campos do endereco, do fim para o comeco -- o bairro fica antes da cidade.

    Os dois portais escrevem diferente, e por isso a quebra e por virgula E por
    hifen:

        chaves na mao: 'Rua X, 155, Jardim Oceania,Joao Pessoa/PB'
        zapimoveis:    'Rua X, 38 - Bessa, Joao Pessoa - PB'

    A versao antiga jogava o endereco inteiro em normalizar_texto, que troca
    virgula e hifen por espaco. Isso achatava a estrutura num fluxo unico de
    palavras e destruia a unica pista confiavel de onde o bairro comeca e
    termina -- dai terem surgido o casamento por substring e o fallback.
    """
    if endereco is None or (not isinstance(endereco, str) and pd.isna(endereco)):
        return []
    campos = [_normalizar_nome(p) for p in re.split(r"[,\-/|]", str(endereco))]
    return [c for c in reversed(campos) if c and c not in _NAO_BAIRRO]


def extrair_bairro(endereco: Any, bairro_informado: Any = None) -> str:
    """Bairro canonico do anuncio, sempre um nome oficial ou 'nao_informado'.

    Ordem de preferencia:

    1. o campo 'bairro' do portal, quando existe e resolve. Ele acerta 99,9%
       das vezes (9.260 de 9.273), mas so o zap o preenche -- o chaves na mao
       nao traz esse campo em nenhum dos 6.473 anuncios.
    2. os campos do endereco, do fim para o comeco.
    3. 'nao_informado'.
    """
    do_portal = casar_bairro(bairro_informado)
    if do_portal:
        return do_portal

    for campo in _campos_do_endereco(endereco):
        achado = casar_bairro(campo)
        if achado:
            return achado

    return "nao_informado"


def converter_preco(val: Any) -> Optional[float]:
    if pd.isna(val) or val is None:
        return None
    s = str(val).strip().replace("R$", "").replace(" ", "")
    if not s or s in ("None", "null", "Nao informado", "nan"):
        return None

    if "." in s and "," not in s:
        partes = s.split(".")
        if len(partes) > 1 and all(len(p) == 3 for p in partes[1:]):
            s = "".join(partes)
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")

    try:
        f = float(s)
        return f if f >= 0 else None
    except ValueError:
        return None


def converter_numero(val: Any) -> Optional[float]:
    if pd.isna(val) or val is None:
        return None
    s = str(val).strip()
    if not s or s in ("None", "null", "Nao informado", "nan"):
        return None

    s_limpo = re.sub(r"[^\d,\.]", "", s)
    if not s_limpo:
        return None

    if "," in s_limpo:
        s_limpo = s_limpo.replace(".", "").replace(",", ".")
    elif "." in s_limpo:
        partes = s_limpo.split(".")
        if len(partes) > 1 and len(partes[-1]) == 3:
            s_limpo = "".join(partes)

    try:
        f = float(s_limpo)
        return f if f >= 0 else None
    except ValueError:
        return None


def carregar_e_normalizar_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    chaves_csv = config.PROCESSED / "imoveis_joao_pessoa_master.csv"
    zap_csv = config.PROCESSED / "imoveis_joao_pessoa_zap_master.csv"

    if not chaves_csv.exists() or not zap_csv.exists():
        print(f"[Erro] Um dos arquivos masters nao existe em {config.PROCESSED}", flush=True)
        sys.exit(1)

    df_chaves = pd.read_csv(chaves_csv, low_memory=False)
    df_zap = pd.read_csv(zap_csv, low_memory=False)

    df_chaves["origem_portal"] = "chaves_na_mao"
    df_zap["origem_portal"] = "zapimoveis"

    df_chaves["preco_norm"] = df_chaves["preco_venda"].apply(converter_preco)
    df_zap["preco_norm"] = df_zap["preco_venda"].apply(converter_preco)

    df_chaves["area_norm"] = df_chaves["area_util"].apply(converter_numero)
    df_zap["area_norm"] = df_zap["area_util"].apply(converter_numero)

    df_chaves["quartos_norm"] = df_chaves["quartos"].apply(converter_numero)
    df_zap["quartos_norm"] = df_zap["quartos"].apply(converter_numero)

    df_chaves["bairro_norm"] = df_chaves["endereco_completo"].apply(lambda e: extrair_bairro(e))
    df_zap["bairro_norm"] = df_zap.apply(lambda row: extrair_bairro(row.get("endereco_completo"), row.get("bairro")), axis=1)

    return df_chaves, df_zap


def executar_deduplicacao_global() -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    print("[OK] Carregando e normalizando datasets de ambos os portais...", flush=True)
    df_chaves, df_zap = carregar_e_normalizar_datasets()

    print(f"     Chaves na Mao: {len(df_chaves)} imoveis")
    print(f"     ZapImoveis:    {len(df_zap)} imoveis")

    mapa_dedup: Dict[tuple, List[Dict[str, Any]]] = {}

    def adicionar_ao_mapa(row_dict, origem):
        row_dict["origem_portal"] = origem
        p = row_dict.get("preco_norm")
        a = row_dict.get("area_norm")
        q = row_dict.get("quartos_norm")
        b = row_dict.get("bairro_norm")

        if p is not None and a is not None and q is not None and not pd.isna(p) and not pd.isna(a) and not pd.isna(q) and p > 10000 and a > 10:
            chave = (b, round(float(p), -2), round(float(a), 0), int(q))
        else:
            url = row_dict.get("url_anuncio") or str(id(row_dict))
            chave = ("unica", url)

        if chave not in mapa_dedup:
            mapa_dedup[chave] = []
        mapa_dedup[chave].append(row_dict)

    for _, r in df_chaves.iterrows():
        adicionar_ao_mapa(r.to_dict(), "chaves_na_mao")

    for _, r in df_zap.iterrows():
        adicionar_ao_mapa(r.to_dict(), "zapimoveis")

    imoveis_deduplicados = []
    total_duplicados_fundidos = 0

    for chave, lista_grupo in mapa_dedup.items():
        if len(lista_grupo) == 1:
            item = lista_grupo[0]
            item["origem_anuncio"] = item["origem_portal"]
            imoveis_deduplicados.append(item)
        else:
            total_duplicados_fundidos += (len(lista_grupo) - 1)
            base_item = dict(lista_grupo[0])
            origens = sorted(list(set(g["origem_portal"] for g in lista_grupo)))
            base_item["origem_anuncio"] = "ambos" if len(origens) > 1 else origens[0]

            for g in lista_grupo[1:]:
                for col, val in g.items():
                    if pd.isna(val) or val is None:
                        continue
                    
                    if col.startswith("comodidade_") or isinstance(val, bool):
                        base_val = bool(base_item.get(col, False))
                        base_item[col] = base_val or bool(val)
                    elif col in ("posicao_solar", "status_construcao", "tipo_unidade"):
                        val_str = str(val).strip()
                        if val_str and val_str != "Nao informado" and base_item.get(col) in (None, "Nao informado", ""):
                            base_item[col] = val_str
                    elif col == "distancia_praia_m":
                        if val is not None and base_item.get(col) is None:
                            base_item[col] = val
                    elif col == "diferenciais_unicos":
                        lst_base = base_item.get("diferenciais_unicos") or []
                        if isinstance(lst_base, str):
                            try:
                                lst_base = json.loads(lst_base)
                            except Exception:
                                lst_base = []
                        lst_nov = val
                        if isinstance(lst_nov, str):
                            try:
                                lst_nov = json.loads(lst_nov)
                            except Exception:
                                lst_nov = []
                        if isinstance(lst_base, list) and isinstance(lst_nov, list):
                            base_item["diferenciais_unicos"] = sorted(list(set(lst_base + lst_nov)))
                    elif col not in base_item or pd.isna(base_item[col]):
                        base_item[col] = val

            imoveis_deduplicados.append(base_item)

    df_dedup = pd.DataFrame(imoveis_deduplicados)

    cols_aux = ["preco_norm", "area_norm", "quartos_norm", "bairro_norm", "origem_portal"]
    df_dedup.drop(columns=[c for c in cols_aux if c in df_dedup.columns], inplace=True, errors="ignore")

    print("\n" + "=" * 65, flush=True)
    print("RESUMO DA DEDUPLICACAO CRUZADA GLOBAL DOS PORTAIS:", flush=True)
    print("=" * 65, flush=True)
    print(f"Total Bruto Inicial Combinado:      {len(df_chaves) + len(df_zap)} imoveis")
    print(f"Total de Anuncios Duplicados Fundidos: {total_duplicados_fundidos} imoveis coincidentes")
    print(f"Total de Imoveis Unicos Finais:      {len(df_dedup)} imoveis unicos no dataset global!")
    print(f"Total de Colunas Unificadas:         {len(df_dedup.columns)} colunas")

    output_csv = config.PROCESSED / "imoveis_joao_pessoa_global_deduplicated.csv"
    output_json = config.PROCESSED / "imoveis_joao_pessoa_global_deduplicated.json"

    df_csv = df_dedup.copy()
    for col in df_csv.columns:
        if df_csv[col].apply(lambda x: isinstance(x, list)).any():
            df_csv[col] = df_csv[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_csv.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[Sucesso] TABELA MÁSTER GLOBAL DEDUPLICADA CSV exportada para: {output_csv}", flush=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(imoveis_deduplicados, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[Sucesso] JSON MÁSTER GLOBAL DEDUPLICADO salvo em: {output_json}", flush=True)

    return df_dedup, imoveis_deduplicados


if __name__ == "__main__":
    executar_deduplicacao_global()
