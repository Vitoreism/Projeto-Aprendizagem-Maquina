# -*- coding: utf-8 -*-
"""
modulo de extracao via llm para recuperar atributos basicos ausentes no html
como suites, quartos, banheiros, garagens, area e andar contidos na descricao (issue #9)
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

# le credenciais do arquivo .env
load_dotenv()

# prompt focado em recuperar atributos do imovel omitidos nos campos do html
SYSTEM_PROMPT_BATCH = """Voce e um especialista em analise de dados imobiliarios.
Sua tarefa e analisar a descricao em texto de imoveis e recuperar atributos numericos basicos do imovel que o anunciante digitou no texto.

Para cada imovel recebido na lista, extraia:
- "quartos": numero inteiro de quartos/dormitorios ou null se nao informado
- "suites": numero inteiro de suites ou null se nao informado
- "banheiros": numero inteiro de banheiros totais ou null se nao informado
- "garagens": numero inteiro de vagas de garagem ou null se nao informado
- "area_m2": numero (float/int) da area util/privativa em m2 ou null se nao informado
- "andar": numero inteiro do andar do apartamento ou null se nao informado
- "valor_condominio": numero (float/int) do valor da taxa de condominio em R$ ou null se nao informado

Responda ESTRITAMENTE com um objeto JSON valido contendo a chave "resultados", que e uma lista de objetos:
{
    "resultados": [
        {
            "id_lote": 0,
            "quartos": 3,
            "suites": 2,
            "banheiros": 3,
            "garagens": 2,
            "area_m2": 120.0,
            "andar": 7,
            "valor_condominio": 450.0
        }
    ]
}

Regras:
1. Retorne apenas numeros inteiros ou decimais quando explicitamente mencionados no texto
2. Se a informação nao constar na descricao, retorne null
3. "suites": identifique no texto expressoes como "sendo 2 suítes", "2 suítes", "1 suíte"
"""

def extrair_lote_atributos_llm(
    client: Any,
    lote_imoveis: List[Dict[str, Any]],
    model: str = "llama-3.1-8b-instant",
    max_retries: int = 5,
) -> Dict[str, Dict[str, Any]]:
    if not lote_imoveis:
        return {}

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

            mapeamento_final = {}
            for item_res in lista_resultados:
                id_lote = item_res.get("id_lote")
                if id_lote is not None and 0 <= id_lote < len(lote_imoveis):
                    url_imovel = lote_imoveis[id_lote]["url_anuncio"]
                    mapeamento_final[url_imovel] = _sanitizar_atributos_basicos(item_res)

            for idx, item in enumerate(lote_imoveis):
                url = item["url_anuncio"]
                if url not in mapeamento_final:
                    mapeamento_final[url] = _retornar_atributos_padrao()

            return mapeamento_final

        except Exception as e:
            erro_str = str(e).lower()
            if "429" in erro_str or "rate limit" in erro_str or "too many requests" in erro_str:
                sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0.5, 1.5)
                print(f"[Rate Limit HTTP 429] Lote tentativa {attempt + 1}/{max_retries}. Aguardando {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                print(f"[Erro de Requisicao Lote] Tentativa {attempt + 1}: {e}")
                time.sleep(1.0)

    res_falha = {}
    for item in lote_imoveis:
        res_falha[item["url_anuncio"]] = _retornar_atributos_padrao()
    return res_falha


def _sanitizar_atributos_basicos(dados: Dict[str, Any]) -> Dict[str, Any]:
    # garante que os campos recuperados sejam numeros ou None
    res = _retornar_atributos_padrao()
    campos_int = ["quartos", "suites", "banheiros", "garagens", "andar"]
    campos_float = ["area_m2", "valor_condominio"]

    for c in campos_int:
        v = dados.get(c)
        if isinstance(v, (int, float)) and v >= 0:
            res[c] = int(v)
        elif isinstance(v, str) and v.isdigit():
            res[c] = int(v)

    for c in campos_float:
        v = dados.get(c)
        if isinstance(v, (int, float)) and v > 0:
            res[c] = float(v)
        elif isinstance(v, str):
            try:
                val_clean = float(v.replace(",", ".").strip())
                if val_clean > 0:
                    res[c] = val_clean
            except ValueError:
                pass

    return res


def _retornar_atributos_padrao() -> Dict[str, Any]:
    return {
        "quartos": None,
        "suites": None,
        "banheiros": None,
        "garagens": None,
        "area_m2": None,
        "andar": None,
        "valor_condominio": None,
    }


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

    print(f"[OK] Iniciando Recuperacao de Atributos Omitidos no HTML via Groq ({len(imoveis)} imoveis)...")
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

    pendentes = [item for item in imoveis if item.get("url_anuncio") and item.get("url_anuncio") not in extracoes]
    print(f"[Pendentes] Imoveis restantes a processar: {len(pendentes)}")

    if not pendentes:
        print("[Concluido] Todos os imoveis do escopo ja foram extraidos!")
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
        print(f"[Concluido] Sessao finalizada! Total em checkpoint: {len(extracoes)} imoveis.")


def fundir_extracoes_nos_csvs_processados() -> None:
    # le as extracoes de atributos recuperados e exporta para csv
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

    caminho_saida = config.INTERIM / "llm_recovered_attributes.csv"
    df_extra.to_csv(caminho_saida, index=False, encoding="utf-8")
    print(f"[Sucesso] Atributos recuperados via LLM exportados para: {caminho_saida}")
    print(f"          Total de registros normalizados: {len(df_extra)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recupera atributos basicos omitidos no HTML (suites, quartos, garagens, area) via Groq LLM (Issue #9)."
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
        help="Exporta os resultados salvos em extractions_llm.json para CSV.",
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
