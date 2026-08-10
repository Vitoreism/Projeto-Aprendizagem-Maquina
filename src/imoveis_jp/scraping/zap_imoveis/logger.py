import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# LOG_FILE e REPORT_FILE passaram a morar no config, que e de onde o scraper.py
# sempre tentou importa-los. Reexportados aqui para nao quebrar quem os importa
# deste modulo.
try:
    from .config import LOG_FILE, REPORT_FILE
except ImportError:
    from config import LOG_FILE, REPORT_FILE


class LoggerManager:
    """
    Gerenciador de Observabilidade e Logging de Execução.
    Registra eventos em tempo real no console e em arquivo de log (`scraping.log`),
    além de gerar um relatório estruturado de encerramento (`execution_report.json`).
    """
    def __init__(self, log_path: str = LOG_FILE, report_path: str = REPORT_FILE):
        self.log_path = log_path
        self.report_path = report_path
        self.start_timestamp = datetime.now()
        
        self.metrics = {
            "inicio_execucao": self.start_timestamp.isoformat(),
            "particoes_processadas": 0,
            "total_links_coletados": 0,
            "novos_links_unicos": 0,
            "sucesso_extracao": 0,
            "falhas_extracao": 0,
            "checkpoints_salvos": 0,
            "reciclagens_contexto": 0,
            "desafios_anti_bot": 0,
            "retentativas_sucesso": 0,
            "particoes_detalhes": []
        }
        
        self._init_log_file()

    def _init_log_file(self):
        """Inicializa o arquivo de log gravando o cabeçalho da sessão."""
        header = (
            f"\n=======================================================\n"
            f"SESSÃO DE RASPAGEM INICIADA EM: {self.start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"=======================================================\n"
        )
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(header)
        except OSError:
            pass

    def log(self, message: str, level: str = "INFO", print_console: bool = True):
        """Registra uma mensagem formatada com timestamp no log e console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        
        if print_console:
            print(message)
            
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(formatted + "\n")
        except OSError:
            pass

    def log_phase(self, phase_name: str, description: str):
        """Registra a transição de fase da raspagem."""
        msg = f"\n[FASE: {phase_name.upper()}] {description}"
        self.log(msg, level="PHASE")

    def log_partition_start(self, partition_index: int, total_partitions: int, label: str):
        """Registra o início do processamento de uma partição."""
        self.metrics["particoes_processadas"] += 1
        msg = f"[*] Partição [{partition_index}/{total_partitions}]: {label}"
        self.log(msg, level="PARTITION")

    def log_partition_result(self, label: str, pages: int, new_links: int, total_partition_links: int):
        """Registra o resumo de links coletados na partição."""
        self.metrics["total_links_coletados"] += total_partition_links
        self.metrics["novos_links_unicos"] += new_links
        
        detail = {
            "label": label,
            "paginas": pages,
            "novos_links": new_links,
            "total_links_pagina": total_partition_links
        }
        self.metrics["particoes_detalhes"].append(detail)
        
        msg = f"   [+] Partição '{label}' concluída: {pages} págs | {new_links} novos links | {total_partition_links} totais"
        self.log(msg, level="PARTITION_RESULT", print_console=False)

    def log_extraction(self, index: int, total: int, title: str, success: bool, error: Optional[str] = None):
        """Registra o resultado da extração de um anúncio individual."""
        if success:
            self.metrics["sucesso_extracao"] += 1
        else:
            self.metrics["falhas_extracao"] += 1
            err_msg = f" (Erro: {error})" if error else ""
            self.log(f"   [-] Falha no imóvel {index}/{total}: {title}{err_msg}", level="ERROR", print_console=False)

    def log_checkpoint(self, saved_batch_count: int, total_accumulated: int):
        """Registra a gravação de um checkpoint de dados."""
        self.metrics["checkpoints_salvos"] += 1
        msg = f"  [CHECKPOINT #{self.metrics['checkpoints_salvos']}] +{saved_batch_count} imóveis salvos. Base total: {total_accumulated}"
        self.log(msg, level="CHECKPOINT")

    def log_context_recycle(self, reason: str = "Rotina Periódica"):
        """Registra a reciclagem do contexto do Playwright."""
        self.metrics["reciclagens_contexto"] += 1
        msg = f"  [♻ CONTEXT RECYCLE #{self.metrics['reciclagens_contexto']}] {reason}"
        self.log(msg, level="RECYCLE")

    def log_resilience_event(self, event_type: str, details: str):
        """Registra evento de mitigação anti-bot, backoff ou retentativa."""
        if "desafio" in event_type.lower() or "cloudflare" in event_type.lower():
            self.metrics["desafios_anti_bot"] += 1
        elif "retentativa sucesso" in event_type.lower():
            self.metrics["retentativas_sucesso"] += 1
            
        msg = f"  [🛡 RESILIÊNCIA - {event_type.upper()}] {details}"
        self.log(msg, level="RESILIENCE")

    def generate_final_report(self) -> Dict[str, Any]:
        """Gera e salva o relatório estruturado de finalização (execution_report.json)."""
        end_timestamp = datetime.now()
        duration = end_timestamp - self.start_timestamp

        self.metrics["fim_execucao"] = end_timestamp.isoformat()
        self.metrics["duracao_total"] = str(duration).split('.')[0]
        self.metrics["duracao_segundos"] = round(duration.total_seconds(), 2)

        try:
            with open(self.report_path, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, ensure_ascii=False, indent=4)
        except OSError as e:
            self.log(f"Erro ao salvar relatório final: {e}", level="ERROR")

        resumo = (
            f"\n=======================================================\n"
            f"[RELATÓRIO FINAL DE EXECUÇÃO E OBSERVABILIDADE]\n"
            f"=======================================================\n"
            f"Duração Total          : {self.metrics['duracao_total']}\n"
            f"Partições Processadas   : {self.metrics['particoes_processadas']}\n"
            f"Links Novos Coletados   : {self.metrics['novos_links_unicos']}\n"
            f"Sucessos de Extração    : {self.metrics['sucesso_extracao']}\n"
            f"Falhas de Extração     : {self.metrics['falhas_extracao']}\n"
            f"Checkpoints Efetuados   : {self.metrics['checkpoints_salvos']}\n"
            f"Arquivo de Log Completo : {self.log_path}\n"
            f"Relatório Estruturado  : {self.report_path}\n"
            f"======================================================="
        )
        self.log(resumo, level="REPORT")
        return self.metrics
