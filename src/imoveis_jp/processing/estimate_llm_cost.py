"""
Script para estimativa de tokens e custos para a Issue #9 (Extração via LLM).
"""
import json
import sys
from pathlib import Path
from imoveis_jp import config

def estimar_tokens_e_custo():
    data_path = config.ANUNCIOS_JSON
    print(f"Lendo dataset de: {data_path}")

    if not data_path.exists():
        print(f"Erro: Arquivo {data_path} nao encontrado.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        imoveis = json.load(f)

    total_imoveis = len(imoveis)
    descricoes_validas = [
        item.get("descricao_completa", "").strip()
        for item in imoveis
        if item.get("descricao_completa") and item.get("descricao_completa") != "Descrição não encontrada."
    ]

    tam_chars = [len(d) for d in descricoes_validas]
    total_chars = sum(tam_chars)
    media_chars = total_chars / len(tam_chars) if tam_chars else 0

    # Estimativa de tokens
    est_tokens_desc = total_chars / 3.7
    est_tokens_prompt = len(descricoes_validas) * 150  # Instruções do prompt
    est_tokens_output = len(descricoes_validas) * 80   # Resposta em JSON estruturado
    total_est_input_tokens = est_tokens_desc + est_tokens_prompt
    total_est_tokens = total_est_input_tokens + est_tokens_output

    print("=" * 65)
    print("RELATORIO DE ESTIMATIVA DE TOKENS E CUSTOS (ISSUE #9)")
    print("=" * 65)
    print(f"Total de imoveis cadastrados:           {total_imoveis:,}")
    print(f"Imoveis com descricao valida:           {len(descricoes_validas):,}")
    print(f"Media de caracteres por descricao:      {media_chars:.1f} caracteres")
    print(f"Maior descricao:                        {max(tam_chars):,} caracteres")
    print(f"Menor descricao:                        {min(tam_chars):,} caracteres")
    print(f"Total acumulado de caracteres:          {total_chars:,} caracteres")
    print("-" * 65)
    print("ESTIMATIVA DE TOKENS:")
    print(f"  - Tokens de Entrada (Descricoes + Prompts): ~{int(total_est_input_tokens):,} tokens")
    print(f"  - Tokens de Saida (Respostas JSON):        ~{int(est_tokens_output):,} tokens")
    print(f"  - TOTAL ESTIMADO DE TOKENS:               ~{int(total_est_tokens):,} tokens")
    print("-" * 65)
    print("ANALISE DE COTAS DO GROQ (PLANO GRATUITO - FREE TIER):")
    print("  Modelo 'llama-3.1-8b-instant':")
    print("    * Custo: R$ 0,00 (Gratuito)")
    print("    * Cota diaria (TPD): 500.000 tokens / dia")
    print("    * Cota de requisicoes (RPD): 14.400 req / dia (~30 RPM)")
    print(f"    * Tempo estimado para processar 100% da base no Groq Free:")
    print(f"      -> ~{total_est_input_tokens / 500000:.1f} dias (se enviado 1 imovel/req no limite diario do Free Tier)")
    print(f"      -> ~{total_est_input_tokens / (500000 * 3):.1f} dias (se agrupado em batches de 3 imoveis/req)")
    print("=" * 65)

if __name__ == "__main__":
    estimar_tokens_e_custo()
