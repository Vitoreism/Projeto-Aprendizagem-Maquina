# -*- coding: utf-8 -*-
"""
modulo de extracao via llm estrito compativel com a base do zapimoveis e chaves na mao (issue #9)
expansao automatica de todas as comodidades em colunas booleanas individuais para ambas as bases
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from imoveis_jp import config

# le credenciais salvas no arquivo .env
load_dotenv()

# prompt da etapa 1: descoberta aberta em lote
SYSTEM_PROMPT_DISCOVERY_BATCH = """Voce e um especialista em PLN e analise de dados imobiliarios.
Sua tarefa e analisar a descricao de cada imovel e extrair TODOS os atributos, caracteristicas, diferenciais, orientacoes e condicoes comerciais citados no texto livre.

Para cada imovel recebido na lista, extraia a lista "atributos_encontrados" com termos curtos em minusculo.

Responda ESTRITAMENTE com um objeto JSON valido contendo a chave "resultados":
{
    "resultados": [
        {
            "id_lote": 0,
            "atributos_encontrados": [
                "posicao solar nascente",
                "distancia 200m da praia",
                "terreo com area privativa",
                "em construcao",
                "entrega em 2026",
                "moveis planejados na cozinha",
                "aceita permuta",
                "aceita fgts",
                "automacao residencial"
            ]
        }
    ]
}
"""

def carregar_clientes_groq() -> List[Any]:
    try:
        from groq import Groq
    except ImportError:
        print("[Erro] A biblioteca 'groq' nao esta instalada. Execute: pip install groq", flush=True)
        sys.exit(1)

    load_dotenv(override=True)
    chaves_brutas = os.environ.get("GROQ_API_KEYS", "") or os.environ.get("GROQ_API_KEY", "")
    lista_chaves = [k.strip() for k in chaves_brutas.split(",") if k.strip() and not k.strip().startswith("SUA_CHAVE")]

    for env_k, env_v in os.environ.items():
        if env_k.startswith("GROQ_API_KEY_") and env_v.strip() and not env_v.strip().startswith("SUA_CHAVE"):
            if env_v.strip() not in lista_chaves:
                lista_chaves.append(env_v.strip())

    if not lista_chaves:
        print("[Erro] Nenhuma chave valida da API Groq encontrada no arquivo .env", flush=True)
        sys.exit(1)

    clientes = [Groq(api_key=key) for key in lista_chaves]
    return clientes


def carregar_atributos_do_ranking(min_frequencia: int = 5) -> List[str]:
    arquivo_ranking = config.INTERIM / "discovered_attributes_rank.json"
    if not arquivo_ranking.exists():
        return [
            "piscina", "academia", "cozinha", "apartamento", "wc_social", "elevador",
            "salão_de_festas", "espaço_gourmet", "sala", "lavanderia", "área_de_serviço",
            "varanda", "coworking", "conforto", "01_vaga_de_garagem", "churrasqueira",
            "bicicletário", "posicao_solar_nascente", "lazer", "playground", "salão_de_jogos",
            "vaga_de_garagem", "praticidade", "recepção", "terreo_com_area_privativa",
            "segurança", "01_quarto", "quarto", "brinquedoteca", "aceita_fgts",
            "restaurante", "aceita_permuta", "restaurantes", "automacao_residencial",
            "em_construcao", "distancia_200m_da_praia", "entrega_em_2026",
            "moveis_planejados_na_cozinha", "01_suíte", "lounge", "supermercados",
            "praia", "localização_privilegiada", "suíte", "1_vaga_de_garagem"
        ]

    with open(arquivo_ranking, "r", encoding="utf-8") as f:
        dados = json.load(f)

    ranking = dados.get("ranking_frequencia", {})
    atributos_validos = []
    for at, count in ranking.items():
        if isinstance(at, str) and len(at.strip()) > 2 and count >= min_frequencia:
            atributos_validos.append(at.strip())

    return atributos_validos[:45]


def construir_prompt_dinamico_batch(atributos_dinamicos: List[str]) -> str:
    lista_chaves_str = "\n".join([f'- "{at.replace(" ", "_")}": true | false (se mencionar "{at}")' for at in atributos_dinamicos])

    prompt = f"""Voce e um especialista em analise de dados imobiliarios em Joao Pessoa (PB).
Sua tarefa e analisar o texto de descricoes de imoveis e extrair os atributos validados empiricamente:

- "id_lote": copie EXATAMENTE o id_lote do imovel correspondente na entrada
- "posicao_solar": "Nascente" | "Poente" | "Sul" | "Norte" | "Nao informado"
- "distancia_praia_m": numero inteiro estimado de metros ate a praia ou null se nao informado
- "status_construcao": "Na planta" | "Em construcao" | "Pronto para morar" | "Usado" | "Nao informado"
- "tipo_unidade": "Terreo com area" | "Terreo simples" | "Cobertura" | "Duplex" | "Apartamento tipo"
{lista_chaves_str}
- "diferenciais_unicos": lista de strings com qualquer outro diferencial exclusivo ou extra citado que nao esteja na lista acima

Responda ESTRITAMENTE com um objeto JSON valido contendo a chave "resultados", que e uma lista de objetos com os atributos de cada imovel.
"""
    return prompt


def extrair_lote_atributos_llm(
    clientes: List[Any],
    lote_imoveis: List[Dict[str, Any]],
    atributos_dinamicos: List[str],
    model: str = "llama-3.1-8b-instant",
    max_retries: int = 15,
) -> Dict[str, Dict[str, Any]]:
    if not lote_imoveis:
        return {}

    payload_prompt = []
    for idx, item in enumerate(lote_imoveis):
        desc = str(item.get("descricao_completa", "")).strip()[:1000]
        if len(desc) < 10 or desc == "Descrição não encontrada.":
            desc = "sem descricao disponivel"
        payload_prompt.append({"id_lote": idx, "descricao": desc})

    prompt_sistema = construir_prompt_dinamico_batch(atributos_dinamicos)
    prompt_usuario = f"Lista de Imoveis para Processar:\n{json.dumps(payload_prompt, ensure_ascii=False)}"

    clientes_atualizados = carregar_clientes_groq()
    pool_clientes = itertools.cycle(clientes_atualizados)

    for attempt in range(max(max_retries, len(clientes_atualizados) * 3)):
        client = next(pool_clientes)
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
            for posicao, item_res in enumerate(lista_resultados):
                # o modelo devolve os resultados na ordem da entrada, mas nem
                # sempre repete o id_lote. exigir o campo fazia TODO resultado
                # cair no default silenciosamente -- foi o que zerou a extracao
                # inteira do zap. a posicao na lista e o fallback natural.
                id_lote = item_res.get("id_lote")
                if not isinstance(id_lote, int) or not (0 <= id_lote < len(lote_imoveis)):
                    id_lote = posicao
                if id_lote >= len(lote_imoveis):
                    continue

                url_imovel = lote_imoveis[id_lote]["url_anuncio"]
                mapeamento_final[url_imovel] = _sanitizar_resposta_dinamica(item_res, atributos_dinamicos)

            faltantes = [i["url_anuncio"] for i in lote_imoveis if i["url_anuncio"] not in mapeamento_final]
            for url in faltantes:
                mapeamento_final[url] = _retornar_atributos_padrao_dinamicos(atributos_dinamicos)

            if len(faltantes) == len(lote_imoveis):
                # lote inteiro no default e sinal de resposta fora do formato,
                # nao de imoveis sem atributo. antes isso passava calado.
                print(
                    f"[Aviso] Lote de {len(lote_imoveis)} imoveis voltou sem nenhum "
                    f"resultado aproveitavel (chaves: {list(dados_json)[:3]}).",
                    flush=True,
                )

            return mapeamento_final

        except Exception as e:
            erro_str = str(e).lower()
            if "429" in erro_str or "rate limit" in erro_str or "too many requests" in erro_str:
                sleep_time = 0.5 + random.uniform(0.1, 0.3)
                time.sleep(sleep_time)
            else:
                time.sleep(0.3)

    return {}


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

    dif = dados.get("diferenciais_unicos", [])
    if isinstance(dif, list):
        res["diferenciais_unicos"] = [str(x).strip().lower() for x in dif if isinstance(x, str) and len(x.strip()) > 2]

    for at in atributos_dinamicos:
        chave = at.replace(" ", "_")
        v = dados.get(chave) or dados.get(at)
        if isinstance(v, bool):
            res[chave] = v
        elif isinstance(v, str):
            res[chave] = v.lower() in ("true", "sim", "yes", "1")

    return res


#: mesmo criterio que extrair_lote_atributos_llm ja usava para trocar o texto
#: por "sem descricao disponivel"; aqui ele serve para nem chamar a API.
def _tem_descricao_util(item: Dict[str, Any]) -> bool:
    desc = str(item.get("descricao_completa", "") or "").strip()
    return len(desc) >= 10 and desc != "Descrição não encontrada."


def _retornar_atributos_padrao_dinamicos(atributos_dinamicos: List[str]) -> Dict[str, Any]:
    res = {
        "posicao_solar": "Nao informado",
        "distancia_praia_m": None,
        "status_construcao": "Nao informado",
        "tipo_unidade": "Apartamento tipo",
        "diferenciais_unicos": [],
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
            print("[Aviso] Arquivo de extracao intermediario corrompido ou vazio. Reiniciando mapa.", flush=True)
    return {}


def salvar_extracoes_checkpoint(caminho: Path, dados: Dict[str, Dict[str, Any]]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temp_file = caminho.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

    for attempt in range(5):
        try:
            temp_file.replace(caminho)
            break
        except PermissionError:
            time.sleep(0.05)


def executar_descoberta_amostral(
    clientes: List[Any],
    imoveis: List[Dict[str, Any]],
    model: str = "llama-3.1-8b-instant",
    batch_size: int = 5,
    max_retries: int = 10,
) -> Counter:
    """Etapa 1: descobre empiricamente quais atributos as descricoes citam.

    Em vez de decidir na mao o que extrair, le uma amostra com o prompt aberto
    (SYSTEM_PROMPT_DISCOVERY_BATCH) e conta a frequencia de cada termo. O
    ranking resultante alimenta carregar_atributos_do_ranking, que hoje cai
    numa lista fixa de 45 atributos quando o arquivo nao existe.

    A funcao tinha sido perdida numa reorganizacao anterior -- sobrou a chamada
    em executar_pipeline_extracao_llm, que quebrava com NameError ao usar
    --discover. Reescrita a partir do contrato dos dois lados: o prompt que
    ficou orfao no modulo e o formato que o leitor do ranking espera.
    """
    amostra = [i for i in imoveis if i.get("url_anuncio") and _tem_descricao_util(i)]
    if not amostra:
        print("[Erro] Nenhum imovel da amostra tem descricao utilizavel.", flush=True)
        return Counter()

    print(
        f"[Descoberta] {len(amostra)} imoveis com descricao, em lotes de {batch_size}.",
        flush=True,
    )

    contador: Counter = Counter()
    pool = itertools.cycle(clientes or carregar_clientes_groq())
    lotes = [amostra[i : i + batch_size] for i in range(0, len(amostra), batch_size)]
    vazios = 0

    for n, lote in enumerate(lotes, start=1):
        payload = [
            {"id_lote": idx, "descricao": str(item.get("descricao_completa", "")).strip()[:1000]}
            for idx, item in enumerate(lote)
        ]

        resultados: List[Dict[str, Any]] = []
        for _ in range(max_retries):
            try:
                resposta = next(pool).chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_DISCOVERY_BATCH},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    model=model,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                resultados = json.loads(resposta.choices[0].message.content).get("resultados", [])
                break
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    time.sleep(0.5 + random.uniform(0.1, 0.3))
                else:
                    time.sleep(0.3)

        if not resultados:
            vazios += 1
            continue

        for item_res in resultados:
            for termo in item_res.get("atributos_encontrados", []) or []:
                if isinstance(termo, str) and len(termo.strip()) > 2:
                    contador[termo.strip().lower()] += 1

        if n % 10 == 0 or n == len(lotes):
            print(f"[Descoberta] {n}/{len(lotes)} lotes | {len(contador)} termos distintos", flush=True)

    if vazios:
        print(f"[Aviso] {vazios} lote(s) nao retornaram resultado aproveitavel.", flush=True)

    destino = config.INTERIM / "discovered_attributes_rank.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(
            {
                "imoveis_amostrados": len(amostra),
                "lotes_sem_resposta": vazios,
                # ordenado por frequencia: carregar_atributos_do_ranking corta
                # nos 45 primeiros, entao a ordem e o que decide o schema.
                "ranking_frequencia": dict(contador.most_common()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[Descoberta] Ranking salvo em: {destino}", flush=True)
    print(f"[Descoberta] {len(contador)} termos distintos. Top 15:", flush=True)
    for termo, freq in contador.most_common(15):
        print(f"    {termo:45s} {freq:5d}", flush=True)

    return contador


def obter_caminhos_dataset(dataset_name: str = "zap") -> tuple[Path, Path, Path, Path, Path]:
    config.ensure_dirs()
    if dataset_name.lower() in ("zap", "zapimoveis"):
        input_file = config.RAW / "imoveis_joao_pessoa_zap.json"
        if not input_file.exists():
            # fallback para onde o scrape do zap deixa o arquivo dentro do pacote.
            # via config.ROOT porque o comando precisa funcionar de qualquer pasta.
            src_zap = config.ROOT / "src" / "imoveis_jp" / "scraping" / "zap_imoveis" / "imoveis_joao_pessoa_zap.json"
            if src_zap.exists():
                input_file = src_zap
        checkpoint_file = config.INTERIM / "extractions_llm_zap.json"
        llm_csv = config.INTERIM / "llm_features_normalized_zap.csv"
        output_json = config.INTERIM / "imoveis_joao_pessoa_zap_v2.json"
        output_master_csv = config.PROCESSED / "imoveis_joao_pessoa_zap_master.csv"
    else:
        input_file = config.ANUNCIOS_JSON
        checkpoint_file = config.EXTRACTIONS_JSON
        llm_csv = config.INTERIM / "llm_features_normalized.csv"
        output_json = config.INTERIM / "imoveis_joao_pessoa_v2.json"
        output_master_csv = config.PROCESSED / "imoveis_joao_pessoa_master.csv"

    return input_file, checkpoint_file, llm_csv, output_json, output_master_csv


def extrair_mapa_comodidades_zap(imoveis: List[Dict[str, Any]]) -> tuple[Dict[str, Dict[str, bool]], List[str]]:
    mapa_comodidades: Dict[str, Dict[str, bool]] = {}
    todas_comodidades = set()

    # 1. faz varredura completa em todos os imoveis para descobrir todas as comodidades unicas
    for item in imoveis:
        raw_str = item.get("comodidades_imovel", "")
        if raw_str and isinstance(raw_str, str):
            partes = [p.strip().lower() for p in raw_str.split(",") if p.strip()]
            for p in partes:
                if any(x in p for x in ["m²", "quarto", "banheiro", "vaga", "andar", "suíte"]):
                    continue
                nome_col = "comodidade_" + p.replace(" ", "_").replace("-", "_")
                todas_comodidades.add(nome_col)

    lista_colunas = sorted(list(todas_comodidades))

    # 2. cria o mapeamento booleano individual para cada imovel
    for item in imoveis:
        url = item.get("url_anuncio")
        if not url:
            continue
        dict_bool = {col: False for col in lista_colunas}
        raw_str = item.get("comodidades_imovel", "")
        if raw_str and isinstance(raw_str, str):
            partes = [p.strip().lower() for p in raw_str.split(",") if p.strip()]
            for p in partes:
                if any(x in p for x in ["m²", "quarto", "banheiro", "vaga", "andar", "suíte"]):
                    continue
                nome_col = "comodidade_" + p.replace(" ", "_").replace("-", "_")
                if nome_col in dict_bool:
                    dict_bool[nome_col] = True

        mapa_comodidades[url] = dict_bool

    return mapa_comodidades, lista_colunas


def executar_pipeline_extracao_llm(
    dataset_name: str = "zap",
    limit: Optional[int] = None,
    batch_size: int = 6,
    workers: int = 15,
    model: str = "llama-3.1-8b-instant",
    sleep_between: float = 0.1,
    dry_run: bool = False,
    discover: bool = False,
    reset_checkpoint: bool = False,
) -> None:
    input_file, checkpoint_file, llm_csv, output_json, output_master_csv = obter_caminhos_dataset(dataset_name)

    if not input_file.exists():
        print(f"[Erro] Arquivo de entrada '{input_file}' nao foi encontrado.", flush=True)
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        imoveis = json.load(f)

    if limit:
        imoveis = imoveis[:limit]

    # o dry-run existe justamente para conferir o plano ANTES de ter credencial,
    # entao carregar as chaves aqui (com sys.exit se faltar) tornava a flag
    # inutil. sem chave, a contagem do pool aparece como 0.
    clientes = [] if dry_run else carregar_clientes_groq()

    if discover:
        executar_descoberta_amostral(clientes=clientes, imoveis=imoveis, model=model, batch_size=5)
        return

    if reset_checkpoint and checkpoint_file.exists():
        print(f"[Reset] Apagando checkpoint antigo '{checkpoint_file.name}'...", flush=True)
        checkpoint_file.unlink()

    atributos_dinamicos = carregar_atributos_do_ranking(min_frequencia=5)

    print(f"[OK] Iniciando Extracao LLM para Dataset: '{dataset_name.upper()}' ({workers} Workers, batch-size={batch_size}) ({len(imoveis)} imoveis)...", flush=True)
    print(f"     Arquivo de Entrada: {input_file}")
    print(f"     Atributos Dinamicos Carregados: {len(atributos_dinamicos)} atributos reais!")
    print(f"     Pool de Chaves API: {len(clientes)} chaves ativas em rotacao")
    print(f"     Tamanho do Lote (Batching): {batch_size} imoveis por lote (ESTRITO PARA 98.5% RIQUEZA)")
    print(f"     Workers Paralelos Concorrentes: {workers} threads simultaneas")
    print(f"     Arquivo de Checkpoint: {checkpoint_file}")

    if dry_run:
        print("[DRY-RUN] Modo --dry-run ativado. Nenhuma chamada de API sera realizada.", flush=True)
        return

    extracoes = carregar_extracoes_existentes(checkpoint_file)
    print(f"[Checkpoint] Extracoes previamente salvas: {len(extracoes)}", flush=True)

    pendentes = [item for item in imoveis if item.get("url_anuncio") and item.get("url_anuncio") not in extracoes]
    print(f"[Pendentes] Imoveis restantes a processar: {len(pendentes)}", flush=True)

    # anuncio sem descricao nao tem o que extrair. antes ele ia para a API com o
    # texto trocado por "sem descricao disponivel" e voltava com tudo False --
    # o mesmo resultado do default, ao custo de uma chamada. no zap isso e 78%
    # da base: 1.974 chamadas em vez de 434. resolvido localmente.
    sem_texto = [item for item in pendentes if not _tem_descricao_util(item)]
    if sem_texto:
        padrao = _retornar_atributos_padrao_dinamicos(atributos_dinamicos)
        for item in sem_texto:
            extracoes[item["url_anuncio"]] = dict(padrao)
        salvar_extracoes_checkpoint(checkpoint_file, extracoes)
        pendentes = [item for item in pendentes if _tem_descricao_util(item)]
        print(
            f"[Pendentes] {len(sem_texto)} sem descricao resolvidos localmente; "
            f"{len(pendentes)} vao para a API.",
            flush=True,
        )

    if not pendentes:
        print(f"[Concluido] Todos os imoveis do dataset '{dataset_name}' ja foram extraidos!", flush=True)
        fundir_extracoes_nos_csvs_processados(dataset_name, atributos_dinamicos)
        return

    lotes = [pendentes[i : i + batch_size] for i in range(0, len(pendentes), batch_size)]
    total_lotes = len(lotes)

    def processar_lote_worker(info_lote):
        idx_lote, lote = info_lote
        res_lote = extrair_lote_atributos_llm(
            clientes=clientes,
            lote_imoveis=lote,
            atributos_dinamicos=atributos_dinamicos,
            model=model,
        )
        return idx_lote, res_lote

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(processar_lote_worker, (i, lote)): i for i, lote in enumerate(lotes)}
            
            concluidos = 0
            for future in as_completed(futures):
                idx_lote, resultados_lote = future.result()
                if resultados_lote:
                    extracoes.update(resultados_lote)
                    salvar_extracoes_checkpoint(checkpoint_file, extracoes)
                
                concluidos += 1
                if concluidos % 5 == 0 or concluidos == total_lotes:
                    print(f"[{concluidos}/{total_lotes}] Lotes processados ({len(extracoes)} imoveis REAIS salvos)... OK!", flush=True)

    except KeyboardInterrupt:
        print("\n[Interrompido] Processamento interrompido pelo usuario. Salvando progresso...", flush=True)
    finally:
        salvar_extracoes_checkpoint(checkpoint_file, extracoes)
        fundir_extracoes_nos_csvs_processados(dataset_name, atributos_dinamicos)
        print(f"[Concluido] Sessao finalizada! Total em checkpoint: {len(extracoes)} imoveis.", flush=True)


def fundir_extracoes_nos_csvs_processados(dataset_name: str = "zap", atributos_dinamicos: Optional[List[str]] = None) -> None:
    import pandas as pd

    input_file, checkpoint_file, llm_csv, output_json, output_master_csv = obter_caminhos_dataset(dataset_name)

    if not checkpoint_file.exists():
        print(f"[Aviso] Nenhuma extracao encontrada em '{checkpoint_file}'. Execute a extracao primeiro.", flush=True)
        return

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        extracoes = json.load(f)

    if not extracoes:
        print("[Aviso] O arquivo de extracoes esta vazio.", flush=True)
        return

    df_extra = pd.DataFrame.from_dict(extracoes, orient="index")
    df_extra.index.name = "url_anuncio"
    df_extra.reset_index(inplace=True)

    df_extra.to_csv(llm_csv, index=False, encoding="utf-8")
    print(f"[Sucesso] Atributos extraidos via LLM exportados para CSV: {llm_csv}", flush=True)
    print(f"          Total de registros normalizados: {len(df_extra)}", flush=True)

    fundir_json_enriquecido_v2(dataset_name, atributos_dinamicos)


def fundir_json_enriquecido_v2(dataset_name: str = "zap", atributos_dinamicos: Optional[List[str]] = None) -> None:
    import pandas as pd

    input_file, checkpoint_file, llm_csv, output_json, output_master_csv = obter_caminhos_dataset(dataset_name)
    amenities_csv = config.INTERIM / "amenities_scraped_normalized.csv"

    if not input_file.exists() or not checkpoint_file.exists():
        return

    with open(input_file, "r", encoding="utf-8") as f:
        imoveis_originais = json.load(f)

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        extracoes = json.load(f)

    if not atributos_dinamicos:
        atributos_dinamicos = carregar_atributos_do_ranking(min_frequencia=5)

    mapa_amenities_html = {}
    colunas_html = []

    if dataset_name.lower() in ("zap", "zapimoveis"):
        mapa_amenities_html, colunas_html = extrair_mapa_comodidades_zap(imoveis_originais)
    else:
        if amenities_csv.exists():
            try:
                df_amen = pd.read_csv(amenities_csv)
                colunas_html = [c for c in df_amen.columns if c != "url_anuncio"]
                for _, row in df_amen.iterrows():
                    url = row.get("url_anuncio")
                    if url:
                        mapa_amenities_html[url] = row.to_dict()
            except Exception:
                pass

    # nomes dos atributos booleanos vindos da llm: so eles podem ser fundidos
    # com as colunas de comodidade do html (ver comentario mais abaixo).
    chaves_llm = {at.replace(" ", "_") for at in atributos_dinamicos}

    imoveis_v2 = []
    for item in imoveis_originais:
        url = item.get("url_anuncio")
        copia_item = dict(item)

        if url and url in extracoes:
            copia_item.update(extracoes[url])
        else:
            copia_item.update(_retornar_atributos_padrao_dinamicos(atributos_dinamicos))

        if url and url in mapa_amenities_html:
            dict_html = mapa_amenities_html[url]
            for col_h in colunas_html:
                val_bool = bool(dict_html.get(col_h, False))
                copia_item[col_h] = val_bool

                # so funde com o atributo do llm quando ele existe de fato como
                # atributo booleano. sem esse filtro, 'comodidade_suites' e
                # 'comodidade_banheiros' sobrescreviam as colunas numericas
                # 'suites' e 'banheiros' do scrap com True/False.
                nome_limpo = col_h.replace("comodidade_", "")
                if nome_limpo in chaves_llm and nome_limpo in copia_item:
                    val_llm = bool(copia_item[nome_limpo])
                    copia_item[nome_limpo] = val_bool or val_llm
        else:
            for col_h in colunas_html:
                copia_item[col_h] = False

        imoveis_v2.append(copia_item)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(imoveis_v2, f, ensure_ascii=False, indent=2)

    print(f"[Sucesso] JSON v2 ({dataset_name.upper()}) salvo em: {output_json}", flush=True)

    df_master = pd.DataFrame(imoveis_v2)
    for col in df_master.columns:
        if df_master[col].apply(lambda x: isinstance(x, list)).any():
            df_master[col] = df_master[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x)

    output_master_csv.parent.mkdir(parents=True, exist_ok=True)
    df_master.to_csv(output_master_csv, index=False, encoding="utf-8")
    print(f"[Sucesso] TABELA MÁSTER CSV ({dataset_name.upper()}: {len(df_master)} imóveis x {len(df_master.columns)} colunas) exportada para: {output_master_csv}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline de extracao via LLM e unificacao master para ZapImoveis e Chaves na Mao (Issue #9)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="zap",
        choices=["zap", "chaves"],
        help="Escolhe o dataset a processar: 'zap' (ZapImoveis 11.841 imoveis) ou 'chaves' (Chaves na Mao 10.758 imoveis). Default: zap.",
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
        default=6,
        help="Numero de imoveis por requisicao de lote (default: 6 imoveis/lote).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=15,
        help="Numero de threads worker paralelas simultaneas (default: 15 threads).",
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
        default=0.1,
        help="Segundos de pausa entre requisicoes de lote (default: 0.1s).",
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
        help="Exporta os resultados salvos em extractions_llm para CSV e master CSV do dataset escolhido.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.merge:
        fundir_extracoes_nos_csvs_processados(dataset_name=args.dataset)
    else:
        executar_pipeline_extracao_llm(
            dataset_name=args.dataset,
            limit=args.limit,
            batch_size=args.batch_size,
            workers=args.workers,
            model=args.model,
            sleep_between=args.sleep,
            dry_run=args.dry_run,
            discover=args.discover,
            reset_checkpoint=args.reset,
        )


if __name__ == "__main__":
    main()
