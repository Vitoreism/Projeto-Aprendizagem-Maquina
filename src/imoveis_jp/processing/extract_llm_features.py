# -*- coding: utf-8 -*-
"""
modulo de integracao com a api do groq para extracao de atributos
em texto livre via lote (batching) para rodar a base inteira no mesmo dia (issue #9)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from imoveis_jp import config

# le credenciais salvas no arquivo .env
load_dotenv()

# prompt em lote enviando uma lista de imoveis e exigindo um json de resultados por id
SYSTEM_PROMPT_BATCH = """Voce e um especialista em analise de dados imobiliarios em Joao Pessoa (PB).
Sua tarefa e analisar uma lista de descricoes de imoveis e extrair os atributos de cada um.

Para cada imovel recebido na lista, extraia as seguintes chaves:
- "posicao_solar": "Nascente" | "Poente" | "Sul" | "Norte" | "Nao informado"
- "vista_mar": true | false
- "beira_mar": true | false
- "varanda_gourmet": true | false
- "piso_porcelanato": true | false
- "moveis_projetados": true | false
- "andar_alto": true | false
- "reformado": true | false
- "aceita_permuta": true | false
- "aceita_financiamento": true | false
- "ar_condicionado": true | false
- "area_lazer_privativa": true | false

Responda ESTRITAMENTE com um objeto JSON valido contendo a chave "resultados", que e uma lista de objetos:
{
    "resultados": [
        {
            "id_lote": 0,
            "posicao_solar": "Nascente",
            "vista_mar": true,
            "beira_mar": false,
            "varanda_gourmet": true,
            "piso_porcelanato": false,
            "moveis_projetados": true,
            "andar_alto": true,
            "reformado": false,
            "aceita_permuta": false,
            "aceita_financiamento": false,
            "ar_condicionado": true,
            "area_lazer_privativa": true
        }
    ]
}

Regras:
1. "posicao_solar": "Nascente", "Poente", "Sul" ou "Norte" apenas se explicitado. senao "Nao informado".
2. "vista_mar": true se mencionar vista para o mar ou vista mar.
3. "beira_mar": true se for na av beira mar ou pe na areia.
4. "varanda_gourmet": true se mencionar varanda ou sacada gourmet.
5. "piso_porcelanato": true se mencionar porcelanato.
6. "moveis_projetados": true se mencionar armarios projetados ou moveis planejados.
7. "andar_alto": true se mencionar andar alto ou cobertura.
8. "reformado": true se mencionar reformado ou novo/pronto para morar.
9. "aceita_permuta": true se mencionar aceita permuta ou troca.
10. "aceita_financiamento": true se mencionar aceita financiamento.
11. "ar_condicionado": true se mencionar ar condicionado ou split.
12. "area_lazer_privativa": true se tiver piscina ou churrasqueira privativa.
"""

def extrair_lote_atributos_llm(
    client: Any,
    lote_imoveis: List[Dict[str, Any]],
    model: str = "llama-3.1-8b-instant",
    max_retries: int = 5,
) -> Dict[str, Dict[str, Any]]:
    # se o lote estiver vazio nao faz chamada
    if not lote_imoveis:
        return {}

    # prepara a lista de descricoes para o prompt em lote
    payload_prompt = []
    for idx, item in enumerate(lote_imoveis):
        desc = item.get("descricao_completa", "").strip()
        if len(desc) < 10 or desc == "Descrição não encontrada.":
            desc = "sem descricao disponivel"
        payload_prompt.append({"id_lote": idx, "descricao": desc})

    prompt_usuario = f"Lista de Imoveis para Processar:\n{json.dumps(payload_prompt, ensure_ascii=False)}"

    base_delay = 2.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_BATCH},
                    {"role": "user", "content": prompt_usuario},
                ],
                model=model,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            conteudo_resposta = response.choices[0].message.content
            dados_json = json.loads(conteudo_resposta)
            lista_resultados = dados_json.get("resultados", [])

            # maapeia os resultados de volta para a url do imovel
            mapeamento_final = {}
            for item_res in lista_resultados:
                id_lote = item_res.get("id_lote")
                if id_lote is not None and 0 <= id_lote < len(lote_imoveis):
                    url_imovel = lote_imoveis[id_lote]["url_anuncio"]
                    mapeamento_final[url_imovel] = _validar_e_sanitizar_resposta(item_res)

            # garante que todos os imoveis do lote tenham resultado mesmo se a llm omitir um id
            for idx, item in enumerate(lote_imoveis):
                url = item["url_anuncio"]
                if url not in mapeamento_final:
                    mapeamento_final[url] = _retornar_atributos_padrao()

            return mapeamento_final

        except Exception as e:
            erro_str = str(e).lower()
            # trata estouro de cota (429) com exponential backoff + jitter
            if "429" in erro_str or "rate limit" in erro_str or "too many requests" in erro_str:
                sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0.5, 1.5)
                print(f"[Rate Limit HTTP 429] Lote tentativa {attempt + 1}/{max_retries}. Aguardando {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                print(f"[Erro de Requisicao Lote] Tentativa {attempt + 1}: {e}")
                time.sleep(1.0)

    print(f"[FALHA] Apos {max_retries} tentativas na extracao do lote.")
    res_falha = {}
    for item in lote_imoveis:
        res_falha[item["url_anuncio"]] = _retornar_atributos_padrao()
    return res_falha


def _validar_e_sanitizar_resposta(dados: Dict[str, Any]) -> Dict[str, Any]:
    # garante que o json retornado contem exatamente as chaves e tipos esperados
    padrao = _retornar_atributos_padrao()
    for chave in padrao:
        if chave in dados:
            if isinstance(padrao[chave], bool):
                val = dados[chave]
                if isinstance(val, bool):
                    padrao[chave] = val
                elif isinstance(val, str):
                    padrao[chave] = val.lower() in ("true", "sim", "yes", "1")
            elif chave == "posicao_solar":
                val_str = str(dados[chave]).strip().title()
                if val_str in ("Nascente", "Poente", "Sul", "Norte"):
                    padrao[chave] = val_str
                else:
                    padrao[chave] = "Nao informado"
    return padrao


def _retornar_atributos_padrao() -> Dict[str, Any]:
    # dicionario com valores default em caso de falha ou ausencia
    return {
        "posicao_solar": "Nao informado",
        "vista_mar": False,
        "beira_mar": False,
        "varanda_gourmet": False,
        "piso_porcelanato": False,
        "moveis_projetados": False,
        "andar_alto": False,
        "reformado": False,
        "aceita_permuta": False,
        "aceita_financiamento": False,
        "ar_condicionado": False,
        "area_lazer_privativa": False,
    }


def carregar_extracoes_existentes(caminho: Path) -> Dict[str, Dict[str, Any]]:
    # le o arquivo de checkpoint salvo em data/interim/
    if caminho.exists():
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("[Aviso] Arquivo de extracao intermediario corrompido ou vazio. Reiniciando mapa.")
    return {}


def salvar_extracoes_checkpoint(caminho: Path, dados: Dict[str, Dict[str, Any]]) -> None:
    # gravacao atomica em arquivo temporario para nao corromper o json se o processo for interrompido
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temp_file = caminho.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    temp_file.replace(caminho)


def executar_pipeline_extracao_llm(
    limit: Optional[int] = None,
    batch_size: int = 5,
    model: str = "llama-3.1-8b-instant",
    sleep_between: float = 1.5,
    dry_run: bool = False,
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

    print(f"[OK] Iniciando Pipeline Lote via Groq ({len(imoveis)} imoveis no escopo)...")
    print(f"     Tamanho do Lote (Batching): {batch_size} imoveis por requisicao")
    print(f"     Modelo selecionado: {model}")
    print(f"     Arquivo de Checkpoint: {checkpoint_file}")

    if dry_run:
        print("[DRY-RUN] Modo --dry-run ativado. Nenhuma chamada de API sera realizada.")
        return

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[Erro] GROQ_API_KEY nao foi encontrada nas variaveis de ambiente nem no arquivo .env.")
        print("       Adicione GROQ_API_KEY=gsk_sua_chave no arquivo .env na raiz do projeto.")
        sys.exit(1)

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except ImportError:
        print("[Erro] A biblioteca 'groq' nao esta instalada. Execute: pip install groq")
        sys.exit(1)

    extracoes = carregar_extracoes_existentes(checkpoint_file)
    print(f"[Checkpoint] Extracoes previamente salvas: {len(extracoes)}")

    # filtra imoveis pendentes de processamento
    pendentes = [item for item in imoveis if item.get("url_anuncio") and item.get("url_anuncio") not in extracoes]
    print(f"[Pendentes] Imoveis restantes a processar: {len(pendentes)}")

    if not pendentes:
        print("[Concluido] Todos os imoveis do escopo ja foram extraidos!")
        return

    # agrupa os imoveis pendentes em lotes de tamanho batch_size
    lotes = [pendentes[i : i + batch_size] for i in range(0, len(pendentes), batch_size)]
    total_lotes = len(lotes)
    processados_nesta_sessao = 0

    try:
        for idx_lote, lote in enumerate(lotes):
            print(f"[{idx_lote + 1}/{total_lotes}] Processando lote com {len(lote)} imoveis...", end=" ")

            resultados_lote = extrair_lote_atributos_llm(
                client=client,
                lote_imoveis=lote,
                model=model,
            )

            # atualiza o dicionario principal com os resultados do lote
            extracoes.update(resultados_lote)
            processados_nesta_sessao += len(lote)
            print("OK!")

            # salva o checkpoint no disco a cada lote concluido
            salvar_extracoes_checkpoint(checkpoint_file, extracoes)

            # pausa entre lotes para respeitar rate limit
            time.sleep(sleep_between)

    except KeyboardInterrupt:
        print("\n[Interrompido] Processamento interrompido pelo usuario. Salvando progresso...")
    finally:
        salvar_extracoes_checkpoint(checkpoint_file, extracoes)
        print(f"[Concluido] Sessao finalizada! Total em checkpoint: {len(extracoes)} imoveis.")


def fundir_extracoes_nos_csvs_processados() -> None:
    # le o json de extracoes salvas e converte em dataframe tabular csv
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
    print(f"[Sucesso] Atributos extraidos via LLM exportados para: {caminho_saida}")
    print(f"          Total de registros normalizados: {len(df_extra)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrai caracteristicas estruturadas da descricao dos imoveis em lote usando Groq LLM (Issue #9)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita o numero de imoveis a processar (util para testes).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Numero de imoveis por requisicao de lote (default: 5 imoveis/lote).",
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
        default=1.5,
        help="Segundos de pausa entre requisicoes de lote (default: 1.5s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exibe o plano de execucao sem chamar a API do Groq.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Normaliza e funde os resultados salvos em extractions_llm.json para CSV.",
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
        )


if __name__ == "__main__":
    main()
