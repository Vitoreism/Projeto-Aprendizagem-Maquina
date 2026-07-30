# -*- coding: utf-8 -*-
"""
modulo de integracao com a api do groq para extracao de atributos
em texto livre da descricao completa dos imoveis (issue #9)
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

# le as credenciais salvas no .env
load_dotenv()

# prompt e schema json enviado para a llm
SYSTEM_PROMPT = """Voce e um especialista em analise de dados imobiliarios em Joao Pessoa (PB).
Sua tarefa e analisar o texto da descricao completa de um anuncio de imovel e extrair atributos relevantes.

Responda ESTRITAMENTE com um objeto JSON valido (sem textos explicativos ou marcacoes markdown antes/depois).
O JSON DEVE conter exatamente as seguintes chaves:

{
    "posicao_solar": "Nascente" | "Poente" | "Sul" | "Norte" | "Nao informado",
    "vista_mar": true | false,
    "beira_mar": true | false,
    "varanda_gourmet": true | false,
    "piso_porcelanato": true | false,
    "moveis_projetados": true | false,
    "andar_alto": true | false,
    "reformado": true | false,
    "aceita_permuta": true | false,
    "aceita_financiamento": true | false,
    "ar_condicionado": true | false,
    "area_lazer_privativa": true | false
}

Regras de Extracao:
1. "posicao_solar": Retorne "Nascente", "Poente", "Sul", "Norte" apenas se mencionado explicitamente no texto. Caso contrario, "Nao informado".
2. "vista_mar": true se mencionar vista para o mar, vista mar, vista definitiva do mar, mar a poucos metros.
3. "beira_mar": true se for na av. beira mar, pe na areia, de frente para o mar.
4. "varanda_gourmet": true se mencionar varanda gourmet, sacada gourmet, terraco gourmet.
5. "piso_porcelanato": true se mencionar porcelanato.
6. "moveis_projetados": true se mencionar armarios projetados, moveis planejados, embutidos, armarios na cozinha/quartos.
7. "andar_alto": true se mencionar andar alto, cobertura, duplex nas ultimas pavimentacoes.
8. "reformado": true se mencionar reformado, totalmente atualizado, novo, pronto para morar.
9. "aceita_permuta": true se mencionar aceita permuta, estuda troca, recebe veiculo/imovel de menor valor.
10. "aceita_financiamento": true se mencionar aceita financiamento, apto para financiamento, documentacao ok.
11. "ar_condicionado": true se mencionar ar condicionado, split, infraestrutura para ar condicionado.
12. "area_lazer_privativa": true se o imovel tiver piscina privativa, churrasqueira privativa, jacuzzzi ou terraco privativo.
"""

def extrair_atributos_llm(
    client: Any,
    descricao: str,
    model: str = "llama-3.1-8b-instant",
    max_retries: int = 5,
) -> Optional[Dict[str, Any]]:
    # se a descricao for muito curta ou nula, nao gasta cota de api
    if not descricao or len(descricao.strip()) < 10 or descricao == "Descrição não encontrada.":
        return _retornar_atributos_padrao()

    prompt_usuario = f"Descricao do Imovel:\n\"\"\"\n{descricao.strip()}\n\"\"\""

    base_delay = 2.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_usuario},
                ],
                model=model,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            conteudo_resposta = response.choices[0].message.content
            dados_extraidos = json.loads(conteudo_resposta)
            return _validar_e_sanitizar_resposta(dados_extraidos)

        except Exception as e:
            erro_str = str(e).lower()
            # trata estouro de cota (429) com exponential backoff + jitter
            if "429" in erro_str or "rate limit" in erro_str or "too many requests" in erro_str:
                sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0.5, 1.5)
                print(f"[Rate Limit HTTP 429] Tentativa {attempt + 1}/{max_retries}. Aguardando {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                print(f"[Erro de Requisicao] Tentativa {attempt + 1}: {e}")
                time.sleep(1.0)

    print(f"[FALHA] Apos {max_retries} tentativas na extracao do imovel.")
    return None


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
    model: str = "llama-3.1-8b-instant",
    sleep_between: float = 1.2,
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

    print(f"[OK] Iniciando Pipeline de Extracao via LLM Groq ({len(imoveis)} imoveis no escopo)...")
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

    processados_nesta_sessao = 0
    total_escopo = len(imoveis)

    try:
        for i, imovel in enumerate(imoveis):
            url = imovel.get("url_anuncio")
            if not url:
                continue

            # pula o que ja foi processado anteriormente
            if url in extracoes:
                continue

            descricao = imovel.get("descricao_completa", "")
            print(f"[{i + 1}/{total_escopo}] Extraindo LLM para: {url[-45:]}...", end=" ")

            resultado = extrair_atributos_llm(
                client=client,
                descricao=descricao,
                model=model,
            )

            if resultado is not None:
                extracoes[url] = resultado
                processados_nesta_sessao += 1
                print("OK!")
            else:
                print("Pulado (falha).")

            # salva o checkpoint no disco a cada 10 novos itens
            if processados_nesta_sessao % 10 == 0 and processados_nesta_sessao > 0:
                salvar_extracoes_checkpoint(checkpoint_file, extracoes)
                print(f"[Checkpoint Salvo] Total registrado: {len(extracoes)} imoveis.")

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
        description="Extrai caracteristicas estruturadas da descricao dos imoveis usando Groq LLM (Issue #9)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita o numero de imoveis a processar (util para testes).",
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
        default=1.2,
        help="Segundos de pausa entre requisicoes para respeitar Rate Limit (default: 1.2s).",
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
            model=args.model,
            sleep_between=args.sleep,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
