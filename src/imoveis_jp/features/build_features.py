# -*- coding: utf-8 -*-
"""
consolida a base global deduplicada numa matriz de features pronta para modelagem

o one-hot das comodidades ja existia, mas gerado por tres caminhos independentes
(llm sobre a descricao, html do chaves na mao e html do zap), o que deixou colunas
duplicadas, colinearidade embutida e nan com significado de portal. este modulo
resolve isso e entrega as categoricas nominais como texto normalizado.

so entram aqui transformacoes deterministicas ou com constante fixa: nada que
aprenda estatistica dos dados, porque isso teria que acontecer depois do split.
por esse criterio saiu tambem o one-hot das nominais, que era pd.get_dummies
sobre a base inteira -- o conjunto de colunas era definido usando as linhas que
virariam teste. ele vive agora no OneHotEncoder dentro do Pipeline de
imoveis_jp.models.train, ajustado so no fold de treino.

duas features foram removidas pelo mesmo criterio:

- 'bairro_preco_m2_medio', que vinha de neighborhoods.csv. o arquivo parecia
  fonte externa, mas correlaciona 0,996 com a mediana de preco/m2 calculada
  desta propria base (erro relativo mediano de 2,4%): era uma agregacao do
  alvo, ou seja, vazamento;
- 'anunciante_qtd_anuncios', contagem calculada sobre a base inteira.

o efeito de bairro continua coberto pela coluna categorica 'bairro', codificada
dentro do Pipeline. para target encoding, o lugar certo tambem e la, ajustado
so no treino.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd
from imoveis_jp import config
from imoveis_jp.processing.deduplicate_dataset import (
    converter_numero,
    converter_preco,
    extrair_bairro,
)
from imoveis_jp.processing.enrich_from_description import enriquecer
from imoveis_jp.processing.extract_amenities_from_scrap import (
    mapear_sinonimos,
    normalizar_texto,
)

ENTRADA = config.PROCESSED / "imoveis_joao_pessoa_global_deduplicated.csv"
SAIDA_CSV = config.PROCESSED / "features_matrix.csv"
SAIDA_RELATORIO = config.INTERIM / "relatorio_consolidacao.json"

#: colunas de texto livre e identificadores que nao entram na matriz numerica.
COLUNAS_DESCARTADAS = [
    "titulo",
    "endereco_completo",
    "descricao_completa",
    "comodidades_area_privativa",
    "comodidades_area_comum",
    "comodidades_imovel",
    "diferenciais_unicos",
    "codigo_imovel",
    "anunciante",
    "distancia_praia_m",  # 99,6% nulo
]

#: colunas numericas do scrap que chegam como texto ('625.000', '--', 'Isento').
COLUNAS_NUMERICAS = [
    "preco_venda",
    "area_util",
    "area_total",
    "quartos",
    "suites",
    "banheiros",
    "garagens",
    "vagas",
    "condominio",
    "iptu",
]

#: faixas plausiveis para apartamento em joao pessoa. o que cai fora vira nan:
#: sao erros de digitacao do anunciante ou do separador de milhar do scrap
#: (havia preco de r$470 milhoes, area de 58 milhoes de m2 e 155.560 garagens).
LIMITES_PLAUSIBILIDADE = {
    "preco_venda": (20_000.0, 50_000_000.0),
    "area_util": (15.0, 2_000.0),
    "area_total": (15.0, 5_000.0),
    "quartos": (0.0, 10.0),
    "suites": (0.0, 10.0),
    "banheiros": (0.0, 15.0),
    "garagens": (0.0, 10.0),
    "condominio": (0.0, 20_000.0),
    "iptu": (0.0, 100_000.0),
}

#: colunas numericas corrompidas pela colisao de nomes com 'comodidade_<x>'
#: na fusao html+llm; sao reconstruidas a partir do json bruto.
COLUNAS_REPARAVEIS = ["suites", "banheiros", "quartos", "garagens", "vagas", "area_util"]

#: binarias que nao descrevem comodidade: termo generico de marketing ou
#: contador que ja existe como coluna numerica propria.
NAO_COMODIDADES = {
    "apartamento",
    "sala",
    "quarto",
    "01_quarto",
    "suite",
    "01_suite",
    "suites",
    "banheiros",
    "conforto",
    "lazer",
    "praticidade",
    "localizacao_privilegiada",
    # so aparece explicita no texto em 8,4% dos imoveis, contra 92,2% com
    # 'garagens' (numerica) > 0 -- mede mencao no anuncio, nao vaga real.
    "vaga_garagem",
}

#: falsos positivos do casamento por substring de mapear_sinonimos:
#: 'supermercados' contem "mar" (praia) e 'carpete' contem "pet" (aceita animais).
EXCECOES_SINONIMO = {"supermercado", "supermercados", "carpete"}

#: nominais que ainda precisam de one-hot de verdade.
CATEGORICAS = ["posicao_solar", "status_construcao", "tipo_unidade", "origem_anuncio"]

FREQUENCIA_MINIMA = 0.01

#: o corte por FREQUENCIA_MINIMA acima continua aqui porque atua sobre as
#: comodidades ANTES de virarem coluna: e decisao de vocabulario, nao selecao
#: de feature. ja o agrupamento de bairro raro e o corte de dummy rara saem
#: daqui -- contavam linhas da base inteira, incluindo as que virariam teste --
#: e viram min_frequency do OneHotEncoder, ajustado por fold.
#:
#: o corte espelhado de dummy quase-constante (FREQUENCIA_MAXIMA_DUMMY = 0,99,
#: vindo do fix de limpeza do one-hot) NAO foi reimplantado, por dois motivos:
#:
#: 1. nao ha "max_frequency" no OneHotEncoder, e o substituto obvio -- um
#:    VarianceThreshold -- e simetrico: calibrado para p > 0,99 ele corta
#:    var < 0,0099, ou seja, tambem tudo com p < 0,01. Isso reimporia o corte
#:    de 1% que esta branch acabou de remover: 1% do treino sao 103 anuncios,
#:    e os 329 bairros com suporte voltariam a ser 38.
#: 2. o custo de manter as tres colunas quase-constantes e proximo de zero. Sem
#:    drop='first', cada grupo de one-hot ja soma 1, entao a categoria dominante
#:    e combinacao linear das outras: a Ridge a absorve na regularizacao e a
#:    arvore nao acha ganho para dividir em 99,6%/0,4%.
#:
#: o item 2 e argumento estrutural, nao medicao -- e fica PENDENTE de verificacao.
#: importancia por permutacao sobre o teste responderia isso com numero: se essas
#: colunas aparecerem com importancia nao-nula, este comentario e que esta errado.

#: toda binaria de comodidade sai prefixada, para nunca colidir com uma coluna
#: numerica de mesmo nome -- foi assim que 'suites' e 'banheiros' se perderam.
PREFIXO_COMODIDADE = "com_"


def canonizar(nome_coluna: str) -> str:
    """Nome canonico de uma binaria, unificando os tres geradores de one-hot."""
    base = nome_coluna
    if base.startswith("comodidade_"):
        base = base[len("comodidade_") :]
    base = normalizar_texto(base)
    if base in EXCECOES_SINONIMO:
        return base
    return mapear_sinonimos(base)


def colunas_binarias(df: pd.DataFrame) -> List[str]:
    """Colunas cujo unico conteudo e True/False (ou nan de portal ausente)."""
    binarias = []
    for col in df.columns:
        # numericas ficam de fora: 0/1 compara igual a False/True em pandas.
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != bool:
            continue
        valores = df[col].dropna()
        if len(valores) and valores.isin([True, False]).all():
            binarias.append(col)
    return binarias


def _mapa_bruto() -> Dict[str, Dict[str, Any]]:
    """Indice url -> registro original, dos dois portais, para reparo numerico."""
    mapa: Dict[str, Dict[str, Any]] = {}
    caminhos = [config.ANUNCIOS_JSON, config.RAW / "imoveis_joao_pessoa_zap.json"]

    for caminho in caminhos:
        if not caminho.exists():
            print(f"[Aviso] Json bruto ausente: {caminho}", flush=True)
            continue
        with open(caminho, "r", encoding="utf-8") as f:
            registros = json.load(f)
        for item in registros:
            url = item.get("url_anuncio")
            if url:
                mapa[url] = item

    return mapa


def _e_numerico(valor: Any) -> bool:
    return converter_numero(valor) is not None


def reparar_numericos(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Restaura do json bruto as celulas numericas sobrescritas por True/False."""
    mapa = _mapa_bruto()
    reparos: Dict[str, int] = {}

    for col in COLUNAS_REPARAVEIS:
        if col not in df.columns:
            continue

        df[col] = df[col].astype(object).where(df[col].notna(), None)
        corrompidas = ~df[col].map(_e_numerico).astype(bool) & df[col].notna()
        indices = df.index[corrompidas]
        if len(indices) == 0:
            continue

        recuperados = 0
        for idx in indices:
            original = mapa.get(df.at[idx, "url_anuncio"], {})
            valor = original.get(col)
            if converter_numero(valor) is not None:
                df.at[idx, col] = valor
                recuperados += 1
            else:
                df.at[idx, col] = None

        reparos[col] = recuperados
        print(
            f"[Reparo] '{col}': {len(indices)} celulas invalidas, "
            f"{recuperados} recuperadas do json bruto.",
            flush=True,
        )

    return df, reparos


def converter_colunas_numericas(df: pd.DataFrame) -> pd.DataFrame:
    """'625.000' -> 625000.0, '--'/'Isento'/'Nao informado' -> nan."""
    for col in COLUNAS_NUMERICAS:
        if col not in df.columns:
            continue
        conversor = converter_preco if col in ("preco_venda", "condominio", "iptu") else converter_numero
        serie = df[col].astype(object).where(df[col].notna(), None)
        if col in ("condominio", "iptu"):
            # 'Isento' e ausencia de cobranca, nao ausencia de informacao.
            serie = serie.map(lambda v: 0 if isinstance(v, str) and v.strip().lower() == "isento" else v)
        df[col] = serie.map(conversor).astype("float64")

    # o zap traz 'vagas' e o chaves 'garagens': e o mesmo atributo.
    if "vagas" in df.columns and "garagens" in df.columns:
        df["garagens"] = df["garagens"].fillna(df["vagas"])
        df.drop(columns=["vagas"], inplace=True)

    return df


def aplicar_limites(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Anula valores fora da faixa plausivel; sem isso o Pearson vira ruido."""
    fora_da_faixa: Dict[str, int] = {}

    for col, (minimo, maximo) in LIMITES_PLAUSIBILIDADE.items():
        if col not in df.columns:
            continue
        invalidos = df[col].notna() & ((df[col] < minimo) | (df[col] > maximo))
        quantidade = int(invalidos.sum())
        if quantidade:
            df.loc[invalidos, col] = pd.NA
            fora_da_faixa[col] = quantidade

    if fora_da_faixa:
        detalhe = ", ".join(f"{c}={n}" for c, n in sorted(fora_da_faixa.items()))
        print(f"[Limites] Valores implausiveis anulados: {detalhe}.", flush=True)

    return df, fora_da_faixa


def consolidar_binarias(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Funde as binarias equivalentes numa coluna canonica por OR logico.

    Resolve de uma vez: nan de portal ausente (vira False), duplicatas por acento
    e falta de mapeamento de sinonimo no ramo zap, e a colinearidade perfeita
    entre '<x>' (llm) e 'comodidade_<x>' (html), que eram a mesma informacao.
    """
    origem = colunas_binarias(df)
    grupos: Dict[str, List[str]] = {}
    for col in origem:
        grupos.setdefault(canonizar(col), []).append(col)

    consolidadas = {}
    for canonico, fontes in grupos.items():
        bloco = df[fontes].fillna(False).astype(bool)
        consolidadas[PREFIXO_COMODIDADE + canonico] = bloco.any(axis=1).astype("int8")

    df = df.drop(columns=origem)
    df = pd.concat([df, pd.DataFrame(consolidadas, index=df.index)], axis=1)

    fundidas = {k: sorted(v) for k, v in grupos.items() if len(v) > 1}
    print(
        f"[Comodidades] {len(origem)} colunas binarias -> {len(grupos)} canonicas "
        f"({len(fundidas)} grupos fundidos).",
        flush=True,
    )

    return df, {
        "total_origem": len(origem),
        "total_canonicas": len(grupos),
        "canonicas": sorted(PREFIXO_COMODIDADE + c for c in grupos),
        "fundidas": fundidas,
    }


def filtrar_comodidades(df: pd.DataFrame, canonicas: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Remove pseudo-atributos e comodidades raras demais para sustentar sinal."""
    presentes = [c for c in canonicas if c in df.columns]

    def _generico(coluna: str) -> bool:
        return coluna[len(PREFIXO_COMODIDADE) :] in NAO_COMODIDADES

    descartadas_generico = [c for c in presentes if _generico(c)]
    restantes = [c for c in presentes if not _generico(c)]

    frequencia = df[restantes].mean()
    descartadas_raras = sorted(frequencia[frequencia < FREQUENCIA_MINIMA].index.tolist())
    mantidas = sorted(frequencia[frequencia >= FREQUENCIA_MINIMA].index.tolist())

    df = df.drop(columns=descartadas_generico + descartadas_raras)

    print(
        f"[Filtro] {len(descartadas_generico)} pseudo-atributos e "
        f"{len(descartadas_raras)} comodidades com frequencia < "
        f"{FREQUENCIA_MINIMA:.0%} removidas. Restam {len(mantidas)}.",
        flush=True,
    )

    return df, {
        "pseudo_atributos": sorted(descartadas_generico),
        "raras": descartadas_raras,
        "mantidas": mantidas,
    }


def normalizar_bairro(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche o bairro (42,6% nulo, so o zap traz) a partir do endereco.

    Extracao linha a linha, sem estatistica agregada: pode rodar antes do split.
    O agrupamento das categorias raras NAO acontece aqui -- ele depende de
    contar quantos imoveis cada bairro tem, e essa contagem sobre a base inteira
    usaria as linhas de teste. Fica a cargo do OneHotEncoder(min_frequency=...)
    dentro do Pipeline, que conta so no fold de treino.
    """
    if "bairro" not in df.columns:
        df["bairro"] = None

    def _texto(valor: Any) -> Any:
        return valor if isinstance(valor, str) else None

    enderecos = df.get("endereco_completo", pd.Series([None] * len(df), index=df.index))
    df["bairro"] = [
        extrair_bairro(_texto(end), _texto(bai))
        for end, bai in zip(enderecos, df["bairro"])
    ]

    print(f"[Bairro] {df['bairro'].nunique()} categorias distintas extraidas.", flush=True)
    return df


def normalizar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza o texto das nominais, sem codificar.

    O one-hot saiu daqui de proposito. Antes era pd.get_dummies sobre a base
    inteira, o que definia o conjunto de colunas usando tambem as linhas que
    virariam teste -- vazamento estrutural. Agora as categoricas chegam como
    texto na matriz e o OneHotEncoder e ajustado dentro do Pipeline, com
    handle_unknown para categoria que so aparece no teste.
    """
    alvos = [c for c in CATEGORICAS + ["bairro"] if c in df.columns]
    for col in alvos:
        df[col] = df[col].fillna("nao_informado").map(normalizar_texto)

    print(f"[Categoricas] {len(alvos)} colunas normalizadas como texto: {', '.join(alvos)}.", flush=True)
    return df


def construir_matriz() -> pd.DataFrame:
    if not ENTRADA.exists():
        print(f"[Erro] Base deduplicada nao encontrada: {ENTRADA}", flush=True)
        sys.exit(1)

    config.ensure_dirs()
    print(f"[Info] Lendo {ENTRADA}...", flush=True)
    bruto = pd.read_csv(ENTRADA, low_memory=False)
    print(f"[Info] Entrada: {bruto.shape[0]} imoveis x {bruto.shape[1]} colunas.", flush=True)

    df = bruto.copy()
    df, reparos = reparar_numericos(df)
    df, info_binarias = consolidar_binarias(df)
    df = converter_colunas_numericas(df)
    df, fora_da_faixa = aplicar_limites(df)
    # depois dos limites de proposito: celula anulada por implausibilidade tambem
    # pode ser recuperada da descricao.
    df, preenchidos_do_texto = enriquecer(df)
    df, info_filtro = filtrar_comodidades(df, info_binarias["canonicas"])
    df = normalizar_bairro(df)
    df = normalizar_categoricas(df)

    df = df.drop(columns=[c for c in COLUNAS_DESCARTADAS if c in df.columns])

    colunas = ["url_anuncio"] + [c for c in df.columns if c != "url_anuncio"]
    df = df[colunas]

    SAIDA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAIDA_CSV, index=False, encoding="utf-8")

    relatorio = {
        "entrada": {"linhas": int(bruto.shape[0]), "colunas": int(bruto.shape[1])},
        "saida": {"linhas": int(df.shape[0]), "colunas": int(df.shape[1])},
        "reparos_numericos": reparos,
        "valores_implausiveis_anulados": fora_da_faixa,
        "preenchidos_da_descricao": preenchidos_do_texto,
        "binarias": info_binarias,
        "filtro": info_filtro,
    }
    SAIDA_RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    with open(SAIDA_RELATORIO, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print("MATRIZ DE FEATURES CONSTRUIDA")
    print("=" * 65)
    print(f"Entrada: {bruto.shape[0]} x {bruto.shape[1]}")
    print(f"Saida:   {df.shape[0]} x {df.shape[1]}")
    print(f"Arquivo: {SAIDA_CSV}")
    print(f"Relatorio de consolidacao: {SAIDA_RELATORIO}")
    print("=" * 65)

    return df


if __name__ == "__main__":
    construir_matriz()
