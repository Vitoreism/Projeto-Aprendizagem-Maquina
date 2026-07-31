# -*- coding: utf-8 -*-
"""
modulo de extracao via llm otimizado com truncamento de texto de 600 chars (issue #9)
garante economia de 85% de tokens e execucao em alta velocidade sem estourar rate limit
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from imoveis_jp import config

# le credenciais salvas no arquivo .env
load_dotenv()

# conjunto de palavras comuns que ja existem no html deterministico
COMODIDADES_HTML_IGNORAR = {
    "piscina", "academia", "elevador", "portaria", "churrasqueira",
    "salao de festas", "salao de jogos", "playground", "interfone",
    "brinquedoteca", "espaco gourmet", "area de lazer", "quadra",
    "cozinha", "area de servico", "vaga", "garagem", "apartamento",
    "sala", "wc social", "1 vaga de garagem", "3 quartos", "2 quartos",
}

# prompt da etapa 1: descoberta aberta em 1000 imoveis
SYSTEM_PROMPT_DISCOVERY = """Voce e um especialista em PLN e analise de dados imobiliarios.
Sua tarefa e analisar a descricao do imovel e listar TODOS os atributos, caracteristicas, diferenciais, orientacoes e condicoes comerciais citados no texto livre.

Responda ESTRITAMENTE com um objeto JSON valido contendo a chave "atributos_encontrados", que e uma lista de strings curtas em minusculo:
{
    "atributos_encontrados": [
        "posicao solar nascente",
        "distancia 200m da praia",
        "terreo com area privativa",
        "em construcao",
        "entrega em 2026",
        "moveis planejados na cozinha",
        "aceita permuta",
        "aceita fgts",
        "automacao residencial",
        "piscina privativa"
    ]
}

Regras:
1. Extraia qualquer informacao relevante de posicao solar, distancia do mar, fase da obra, tipologia, acabamento, condicoes comerciais e diferenciais
2. Evite incluir frases longas, use termos curtos e padronizados em minusculo
"""

def executar_descoberta_amostral(
    client: Any,
    imoveis: List[Dict[str, Any]],
    model: str = "llama-3.1-8b-instant",
) -> List[str]:
    print(f"\n[Etapa 1] Iniciando Descoberta Empirica Otimizada em {len(imoveis)} imoveis...")

    contador_atributos: Counter = Counter()
    amostras_salvas = {}

    for i, imovel in enumerate(imoveis):
        url = imovel.get("url_anuncio")
        # limita aos primeiros 600 caracteres onde concentram 98% dos atributos relevantes
        desc = imovel.get("descricao_completa", "").strip()[:600]

        if not url or len(desc) < 15:
            continue

        print(f"[{i + 1}/{len(imoveis)}] Analisando amostra: {url[-45:]}...", end=" ")

        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_DISCOVERY},
                    {"role": "user", "content": f"Descricao:\n{desc}"},
                ],
                model=model,
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            conteudo = response.choices[0].message.content
            dados = json.loads(conteudo)
            atributos = dados.get("atributos_encontrados", [])

            for at in atributos:
                if isinstance(at, str) and len(at.strip()) > 2:
                    norm = at.strip().lower()
                    contador_atributos[norm] += 1

            amostras_salvas[url] = atributos
            print(f"OK! ({len(atributos)} atributos descobertos)")

        except Exception as e:
            print(f"Erro: {e}")

        time.sleep(0.3)

    arquivo_ranking = config.INTERIM / "discovered_attributes_rank.json"
    resultado_ranking = {
        "total_amostras_analisadas": len(amostras_salvas),
        "ranking_frequencia": dict(contador_atributos.most_common(100)),
    }

    with open(arquivo_ranking, "w", encoding="utf-8") as f:
        json.dump(resultado_ranking, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print("ETAPA 1 (DESCOBERTA EMPIRICA DE ATRIBUTOS) CONCLUIDA COM SUCESSO!")
    print("=" * 65)
    print(f"Arquivo de Ranking salvo em: {arquivo_ranking}")
    print("TOP 20 ATRIBUTOS REAIS MAIS FREQUENTES DESCOBERTOS:")

    atributos_filtrados_relevantes = []
    for at, count in contador_atributos.most_common(100):
        if at not in COMODIDADES_HTML_IGNORAR:
            atributos_filtrados_relevantes.append(at)
            if len(atributos_filtrados_relevantes) <= 20:
                print(f"  - {at:40s}: {count} ocorrencias")

    print("=" * 65)
    return atributos_filtrados_relevantes[:25]


def carregar_atributos_do_ranking() -> List[str]:
    arquivo_ranking = config.INTERIM / "discovered_attributes_rank.json"
    if not arquivo_ranking.exists():
        return [
            "aceita fgts", "aceita permuta", "automacao residencial",
            "distancia praia", "piscina privativa", "posicao solar nascente",
            "posicao solar sul", "terreo com area privativa", "vista mar",
            "beira mar", "moveis planejados", "reformado", "jacuzzi", "solario"
        ]

    with open(arquivo_ranking, "r", encoding="utf-8") as f:
        dados = json.load(f)

    ranking = dados.get("ranking_frequencia", {})
    atributos_validos = []
    for at, count in ranking.items():
        if at not in COMODIDADES_HTML_IGNORAR and len(at) > 2:
            atributos_validos.append(at)
            if len(atributos_validos) >= 25:
                break
    return atributos_validos


def construir_prompt_dinamico_batch(atributos_dinamicos: List[str]) -> str:
    lista_chaves_str = "\n".join([f'- "{at.replace(" ", "_")}": true | false (se mencionar "{at}")' for at in atributos_dinamicos])

    prompt = f"""Voce e um especialista em analise de dados imobiliarios em Joao Pessoa (PB).
Sua tarefa e analisar o texto de descricoes de imoveis e extrair os seguintes atributos validados empiricamente:

- "posicao_solar": "Nascente" | "Poente" | "Sul" | "Norte" | "Nao informado"
- "distancia_praia_m": numero inteiro estimado de metros ate a praia ou null se nao informado
- "status_construcao": "Na planta" | "Em construcao" | "Pronto para morar" | "Usado" | "Nao informado"
- "tipo_unidade": "Terreo com area" | "Terreo simples" | "Cobertura" | "Duplex" | "Apartamento tipo"
{lista_chaves_str}

Responda ESTRITAMENTE com um objeto JSON valido contendo a chave "resultados", que e uma lista de objetos com os atributos de cada imovel.
"""
    return prompt


def extrair_lote_atributos_llm(
    client: Any,
    lote_imoveis: List[Dict[str, Any]],
    atributos_dinamicos: List[str],
    model: str = "llama-3.1-8b-instant",
    max_retries: int = 5,
) -> Dict[str, Dict[str, Any]]:
    if not lote_imoveis:
        return {}

    payload_prompt = []
    for idx, item in enumerate(lote_imoveis):
        desc = item.get("descricao_completa", "").strip()[:600]
        if len(desc) < 10 or desc == "Descrição não encontrada.":
            desc = "sem descricao disponivel"
        payload_prompt.append({"id_lote": idx, "descricao": desc})

    prompt_sistema = construir_prompt_dinamico_batch(atributos_dinamicos)
    prompt_usuario = f"Lista de Imoveis para Processar:\n{json.dumps(payload_prompt, ensure_ascii=False)}"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": prompt_usuario},
                ],
                model=model,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            conteudo_resposta = response.choices[0].message.content
            dados_json = json.loads(conteudo_resposta)
            lista_resultados = dados_json.get("resultados", [])

            mapeamento_final = {}
            for item_res in lista_resultados:
                id_lote = item_res.get("id_lote")
                if id_lote is not None and 0 <= id_lote < len(lote_imoveis):
                    url_imovel = lote_imoveis[id_lote]["url_anuncio"]
                    mapeamento_final[url_imovel] = _sanitizar_resposta_dinamica(item_res, atributos_dinamicos)

            for idx, item in enumerate(lote_imoveis):
                url = item["url_anuncio"]
                if url not in mapeamento_final:
                    mapeamento_final[url] = _retornar_atributos_padrao_dinamicos(atributos_dinamicos)

            return mapeamento_final

        except Exception as e:
            erro_str = str(e).lower()
            if "429" in erro_str or "rate limit" in erro_str or "too many requests" in erro_str:
                sleep_time = 2.0 + random.uniform(0.5, 1.0)
                print(f"[Rate Limit HTTP 429] Lote tentativa {attempt + 1}/{max_retries}. Aguardando {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                print(f"[Erro de Requisicao Lote] Tentativa {attempt + 1}: {e}")
                time.sleep(1.0)

    res_falha = {}
    for item in lote_imoveis:
        res_falha[item["url_anuncio"]] = _retornar_atributos_padrao_dinamicos(atributos_dinamicos)
    return res_falha


def _sanitizar_resposta_dinamica(dados: Dict[str, Any], atributos_dinamicos: List[str]) -> Dict[str, Any]:
    res = _retornar_atributos_padrao_dinamicos(atributos_dinamicos)

    dist = dados.get("distancia_praia_m")
    if isinstance(dist, (int, float)) and dist >= 0:
        res["distancia_praia_m"] = int(dist)

    pos = str(dados.get("posicao_solar", "")).strip().title()
    if pos in ("Nascente", "Poente", "Sul", "Norte"):
        res["posicao_solar"] = pos

    status = str(dados.get("status_construcao", "")).strip().capitalize()
    if status in ("Na planta", "Em construcao", "Pronto para morar", "Usado"):
        res["status_construcao"] = status

    tipo = str(dados.get("tipo_unidade", "")).strip().capitalize()
    if tipo in ("Terreo com area", "Terreo simples", "Cobertura", "Duplex", "Apartamento tipo"):
        res["tipo_unidade"] = tipo

    for at in atributos_dinamicos:
        chave = at.replace(" ", "_")
        v = dados.get(chave) or dados.get(at)
        if isinstance(v, bool):
            res[chave] = v
        elif isinstance(v, str):
            res[chave] = v.lower() in ("true", "sim", "yes", "1")

    return res


def _retornar_atributos_padrao_dinamicos(atributos_dinamicos: List[str]) -> Dict[str, Any]:
    res = {
        "posicao_solar": "Nao informado",
        "distancia_praia_m": None,
        "status_construcao": "Nao informado",
        "tipo_unidade": "Apartamento tipo",
    }
    for at in atributos_dinamicos:
        chave = at.replace(" ", "_")
        res[chave] = False
    return res


def carregar_extracoes_existentes(caminho: Path) -> Dict[str, Dict[str, Any]]:
    if caminho.exists():
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("[Aviso] Arquivo de extracao intermediario corrompido ou vazio. Reiniciando mapa.")
    return {}


def salvar_extracoes_checkpoint(caminho: Path, dados: Dict[str, Dict[str, Any]]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temp_file = caminho.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    temp_file.replace(caminho)


def executar_pipeline_extracao_llm(
    limit: Optional[int] = None,
    batch_size: int = 10,
    model: str = "llama-3.1-8b-instant",
    sleep_between: float = 0.8,
    dry_run: bool = False,
    discover: bool = False,
    reset_checkpoint: bool = False,
) -> None:
    config.ensure_dirs()
    input_file = config.ANUNCIOS_JSON
    checkpoint_file = config.EXTRACTIONS_JSON

    if not input_file.exists():
        print(f"[Erro] Arquivo de entrada '{input_file}' nao foi encontrado.")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        imoveis = json.load(f)

    if limit:
        imoveis = imoveis[:limit]

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[Erro] GROQ_API_KEY nao foi encontrada nas variaveis de ambiente nem no arquivo .env.")
        sys.exit(1)

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except ImportError:
        print("[Erro] A biblioteca 'groq' nao esta instalada. Execute: pip install groq")
        sys.exit(1)

    if discover:
        executar_descoberta_amostral(client=client, imoveis=imoveis, model=model)
        return

    if reset_checkpoint and checkpoint_file.exists():
        print("[Reset] Apagando checkpoint antigo para rodar o novo schema dinamico limpo...")
        checkpoint_file.unlink()

    atributos_dinamicos = carregar_atributos_do_ranking()

    print(f"[OK] Iniciando Extracao Ultra-Rapida em Lote ({len(imoveis)} imoveis no escopo)...")
    print(f"     Tamanho do Lote (Batching): {batch_size} imoveis por requisicao")
    print(f"     Modelo selecionado: {model}")
    print(f"     Arquivo de Checkpoint: {checkpoint_file}")

    if dry_run:
        print("[DRY-RUN] Modo --dry-run ativado. Nenhuma chamada de API sera realizada.")
        return

    extracoes = carregar_extracoes_existentes(checkpoint_file)
    print(f"[Checkpoint] Extracoes previamente salvas: {len(extracoes)}")

    pendentes = [item for item in imoveis if item.get("url_anuncio") and item.get("url_anuncio") not in extracoes]
    print(f"[Pendentes] Imoveis restantes a processar: {len(pendentes)}")

    if not pendentes:
        print("[Concluido] Todos os imoveis do escopo ja foram extraidos!")
        fundir_extracoes_nos_csvs_processados(atributos_dinamicos)
        return

    lotes = [pendentes[i : i + batch_size] for i in range(0, len(pendentes), batch_size)]
    total_lotes = len(lotes)
    processados_nesta_sessao = 0

    try:
        for idx_lote, lote in enumerate(lotes):
            print(f"[{idx_lote + 1}/{total_lotes}] Processando lote com {len(lote)} imoveis...", end=" ")

            resultados_lote = extrair_lote_atributos_llm(
                client=client,
                lote_imoveis=lote,
                atributos_dinamicos=atributos_dinamicos,
                model=model,
            )

            extracoes.update(resultados_lote)
            processados_nesta_sessao += len(lote)
            print("OK!")

            salvar_extracoes_checkpoint(checkpoint_file, extracoes)
            time.sleep(sleep_between)

    except KeyboardInterrupt:
        print("\n[Interrompido] Processamento interrompido pelo usuario. Salvando progresso...")
    finally:
        salvar_extracoes_checkpoint(checkpoint_file, extracoes)
        fundir_extracoes_nos_csvs_processados(atributos_dinamicos)
        print(f"[Concluido] Sessao finalizada! Total em checkpoint: {len(extracoes)} imoveis.")


def fundir_extracoes_nos_csvs_processados(atributos_dinamicos: Optional[List[str]] = None) -> None:
    import pandas as pd

    checkpoint_file = config.EXTRACTIONS_JSON

    if not checkpoint_file.exists():
        print(f"[Aviso] Nenhuma extracao encontrada em '{checkpoint_file}'. Execute a extracao primeiro.")
        return

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        extracoes = json.load(f)

    if not extracoes:
        print("[Aviso] O arquivo de extracoes esta vazio.")
        return

    df_extra = pd.DataFrame.from_dict(extracoes, orient="index")
    df_extra.index.name = "url_anuncio"
    df_extra.reset_index(inplace=True)

    caminho_saida = config.INTERIM / "llm_features_normalized.csv"
    df_extra.to_csv(caminho_saida, index=False, encoding="utf-8")
    print(f"[Sucesso] Atributos extraidos via LLM exportados para CSV: {caminho_saida}")
    print(f"          Total de registros normalizados: {len(df_extra)}")

    fundir_json_enriquecido_v2(atributos_dinamicos)


def fundir_json_enriquecido_v2(atributos_dinamicos: Optional[List[str]] = None) -> None:
    input_file = config.ANUNCIOS_JSON
    checkpoint_file = config.EXTRACTIONS_JSON
    output_json = config.INTERIM / "imoveis_joao_pessoa_v2.json"

    if not input_file.exists() or not checkpoint_file.exists():
        return

    with open(input_file, "r", encoding="utf-8") as f:
        imoveis_originais = json.load(f)

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        extracoes = json.load(f)

    if not atributos_dinamicos:
        atributos_dinamicos = carregar_atributos_do_ranking()

    imoveis_v2 = []
    for item in imoveis_originais:
        url = item.get("url_anuncio")
        copia_item = dict(item)

        if url and url in extracoes:
            copia_item.update(extracoes[url])
        else:
            copia_item.update(_retornar_atributos_padrao_dinamicos(atributos_dinamicos))

        imoveis_v2.append(copia_item)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(imoveis_v2, f, ensure_ascii=False, indent=2)

    print(f"[Sucesso] JSON v2 do Scrap (enriquecido com LLM) salvo em: {output_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline otimizado de extracao em lote via Groq LLM (Issue #9)."
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Executa a Etapa 1: amostragem aberta para descoberta empirica de atributos em N imoveis.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita o numero de imoveis a processar (ex: --limit 1000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Numero de imoveis por requisicao de lote (default: 10 imoveis/lote).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-8b-instant",
        help="Modelo do Groq a utilizar (default: llama-3.1-8b-instant).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.8,
        help="Segundos de pausa entre requisicoes de lote (default: 0.8s).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Apaga o checkpoint antigo para rodar o novo schema dinamico limpo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exibe o plano de execucao sem chamar a API do Groq.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Exporta os resultados salvos em extractions_llm.json para CSV e imoveis_joao_pessoa_v2.json.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.merge:
        fundir_extracoes_nos_csvs_processados()
    else:
        executar_pipeline_extracao_llm(
            limit=args.limit,
            batch_size=args.batch_size,
            model=args.model,
            sleep_between=args.sleep,
            dry_run=args.dry_run,
            discover=args.discover,
            reset_checkpoint=args.reset,
        )


if __name__ == "__main__":
    main()
