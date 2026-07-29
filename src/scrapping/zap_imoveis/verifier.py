import re
from enum import Enum
from typing import Tuple
from bs4 import BeautifulSoup


class PageStatus(Enum):
    OK = "OK"
    CLOUDFLARE_CHALLENGE = "CLOUDFLARE_CHALLENGE"
    SOFT_BAN = "SOFT_BAN"
    LAYOUT_CHANGED = "LAYOUT_CHANGED"
    NOT_FOUND = "NOT_FOUND"
    ERROR = "ERROR"


class PageIntegrityVerifier:
    """
    Verificador de Integridade de Páginas e Evasão Anti-Bot.
    Distingue requisições bem-sucedidas de desafios Cloudflare (HTTP ou HTML),
    Soft-Bans mascarados em HTTP 200 e anomalias de layout.
    """
    
    CLOUDFLARE_TITLE_SIGNATURES = [
        "attention required",
        "just a moment",
        "cloudflare",
        "security check",
        "access denied",
        "ddos protection",
        "verifique se você é humano",
        "verify you are human"
    ]
    
    CLOUDFLARE_BODY_SIGNATURES = [
        "cf-challenge",
        "cf-turnstile",
        "ray id:",
        "enable javascript and cookies",
        "please wait...",
        "challenge-running",
        "g-recaptcha"
    ]

    @classmethod
    def verify(cls, status_code: int, page_title: str, html_content: str) -> Tuple[PageStatus, str]:
        """
        Avalia a resposta HTTP e o HTML para determinar a integridade do acesso.
        
        Returns:
            Tuple[PageStatus, str]: (Status detectado, Mensagem explicativa)
        """
        # 1. Checagem direta de status HTTP de bloqueio ou 404
        if status_code in (403, 429, 503):
            return PageStatus.CLOUDFLARE_CHALLENGE, f"Bloqueio de Rede HTTP {status_code}"

        if status_code == 404:
            return PageStatus.NOT_FOUND, "Página não encontrada (404)"

        title_lower = (page_title or "").lower()
        html_lower = (html_content or "")[:10000].lower()

        # 2. Checagem de assinaturas Cloudflare no título
        for sig in cls.CLOUDFLARE_TITLE_SIGNATURES:
            if sig in title_lower:
                return PageStatus.CLOUDFLARE_CHALLENGE, f"Desafio Cloudflare no Título: '{page_title}'"

        # 3. Checagem de assinaturas de desafio no corpo HTML
        for sig in cls.CLOUDFLARE_BODY_SIGNATURES:
            if sig in html_lower:
                return PageStatus.CLOUDFLARE_CHALLENGE, f"Assinatura Cloudflare/Captcha detectada: '{sig}'"

        # 4. Resposta nula ou HTML truncado sem assinatura -> Soft-Ban
        if not html_content or len(html_content.strip()) < 500:
            return PageStatus.SOFT_BAN, "Resposta nula ou HTML truncado (< 500 bytes)"

        # 5. Detecção de Soft-Ban (HTTP 200 com HTML de bloqueio oculto)
        tem_zap_brand = "zap" in html_lower or "olx" in html_lower or "imovel" in html_lower
        if not tem_zap_brand and len(html_content) < 3000:
            return PageStatus.SOFT_BAN, "Soft-Ban detectado (HTTP 200 sem marca e conteúdo reduzido)"

        # 6. Sanidade da estrutura DOM
        soup = BeautifulSoup(html_content[:30000], 'html.parser')
        
        tem_h1 = bool(soup.find('h1'))
        tem_links = bool(soup.find('a', href=True))
        tem_main_struct = bool(soup.find(['main', 'body', 'div']))

        if not (tem_links or tem_h1 or tem_main_struct):
            return PageStatus.LAYOUT_CHANGED, "Estrutura HTML incompatível com o padrão do Zap Imóveis"

        return PageStatus.OK, "Página íntegra"
