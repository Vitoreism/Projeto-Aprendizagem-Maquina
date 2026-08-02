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


def extrair_bairro(endereco: Any, bairro_informado: Any = None) -> str:
    if bairro_informado and not pd.isna(bairro_informado):
        b = normalizar_texto(bairro_informado)
        if b and b not in ("joao pessoa", "pb", "paraiba"):
            return b.replace(" ", "_")

    end_norm = normalizar_texto(endereco)
    if not end_norm:
        return "nao_informado"

    bairros_conhecidos = [
        "manaira", "bessa", "cabo branco", "jardim oceania", "altiplano",
        "aeroclube", "tambaú", "tambau", "bancarios", "jardim cidade universitaria",
        "portal do sol", "torre", "estados", "expedicionarios", "ipês", "ipes",
        "bessa norte", "intermares", "miramar", "tambauzinho", "bessa sul",
        "castelo branco", "manguinhos", "jaguaribe", "centro", "varadouro",
        "agua fria", "anatólia", "anatolia", "cristo redentor", "geisel"
    ]

    for b in bairros_conhecidos:
        if b in end_norm:
            return b.replace(" ", "_")

    partes = [p.strip() for p in end_norm.split() if len(p.strip()) > 3]
    return partes[0] if partes else "nao_informado"


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
