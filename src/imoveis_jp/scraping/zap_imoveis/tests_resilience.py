import sys
import os

try:
    from imoveis_jp.scraping.zap_imoveis.verifier import PageIntegrityVerifier, PageStatus
    from imoveis_jp.scraping.zap_imoveis.rate_limiter import AdaptiveRateLimiter
    from imoveis_jp.scraping.zap_imoveis.config import STORAGE_STATE_FILE, CONTEXT_RECYCLE_EVERY
except ImportError:
    from verifier import PageIntegrityVerifier, PageStatus
    from rate_limiter import AdaptiveRateLimiter
    from config import STORAGE_STATE_FILE, CONTEXT_RECYCLE_EVERY


def test_verifier():
    print("=== TESTANDO PAGE INTEGRITY VERIFIER ===")
    # 1. Test HTTP 403
    status, reason = PageIntegrityVerifier.verify(403, "Zap Imóveis", "<html></html>")
    assert status == PageStatus.CLOUDFLARE_CHALLENGE, f"Esperado CLOUDFLARE_CHALLENGE, obteve {status}"
    print(f"[OK] HTTP 403 -> {status.value} ({reason})")

    # 2. Test Cloudflare Title
    status, reason = PageIntegrityVerifier.verify(200, "Just a moment... - Cloudflare", "<html><body>cf-turnstile</body></html>")
    assert status == PageStatus.CLOUDFLARE_CHALLENGE, f"Esperado CLOUDFLARE_CHALLENGE, obteve {status}"
    print(f"[OK] Title Challenge -> {status.value} ({reason})")

    # 3. Test Soft Ban
    status, reason = PageIntegrityVerifier.verify(200, "Blocked", "<html><body>Access Restricted</body></html>")
    assert status == PageStatus.SOFT_BAN, f"Esperado SOFT_BAN, obteve {status}"
    print(f"[OK] Soft Ban -> {status.value} ({reason})")

    # 4. Test 404 Not Found
    status, reason = PageIntegrityVerifier.verify(404, "Não Encontrado", "<html>404 Not Found</html>")
    assert status == PageStatus.NOT_FOUND, f"Esperado NOT_FOUND, obteve {status}"
    print(f"[OK] HTTP 404 -> {status.value} ({reason})")

    # 5. Test OK Page
    valid_html = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Apartamento 3 Quartos para Venda em Manaira, João Pessoa - Zap Imóveis</title>
            <meta name="description" content="Confira este lindo apartamento de 3 quartos à venda no bairro Manaíra em João Pessoa.">
        </head>
        <body>
            <header>
                <div class="logo">Zap Imóveis</div>
            </header>
            <main>
                <h1>Apartamento 3 Quartos à Venda em Manaira</h1>
                <div class="price">R$ 450.000</div>
                <address>Rua Professora Maria Sales, Manaira, João Pessoa - PB</address>
                <a href="/imovel/12345678/">Ver detalhes do anúncio do imóvel</a>
                <p>Excelente oportunidade de investimento próximo à praia de Manaíra. Imóvel com área útil de 85m², 1 vaga de garagem e condomínio completo.</p>
            </main>
            <footer>Zap Imóveis 2026 - Todos os direitos reservados</footer>
        </body>
    </html>
    """
    status, reason = PageIntegrityVerifier.verify(200, "Apartamento para Venda", valid_html)
    assert status == PageStatus.OK, f"Esperado OK, obteve {status}"
    print(f"[OK] OK Page -> {status.value} ({reason})\n")


def test_rate_limiter():
    print("=== TESTANDO ADAPTIVE RATE LIMITER ===")
    limiter = AdaptiveRateLimiter(window_size=10, threshold=0.15)
    
    # Registra sucessos
    for _ in range(8):
        limiter.record_result(True)
    assert not limiter.is_throttled(), "Não deveria estar throttled"
    print(f"[OK] Taxa de erro normal ({limiter.failure_rate:.0%}): Throttled = {limiter.is_throttled()}")

    # Registra falhas para ultrapassar limiar
    limiter.record_result(False)
    limiter.record_result(False)
    limiter.record_result(False)
    assert limiter.is_throttled(), "Deveria estar throttled"
    print(f"[OK] Taxa de erro alta ({limiter.failure_rate:.0%}): Throttled = {limiter.is_throttled()}")

    delay_normal = limiter.calculate_delay((2.0, 4.0))
    print(f"[OK] Delay adaptativo sob penalidade: {delay_normal}s\n")


if __name__ == "__main__":
    test_verifier()
    test_rate_limiter()
    print("ALL RESILIENCE UNIT TESTS PASSED SUCCESSFULLY!")
