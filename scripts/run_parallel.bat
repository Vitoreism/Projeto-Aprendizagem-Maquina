@echo off
REM Lanca 3 workers do scraper em janelas separadas (scrape paralelo).
REM Uso: de dois cliques neste arquivo, OU rode  scripts\run_parallel.bat  na raiz do repo.
REM Cada worker pega uma fatia disjunta e escreve no seu proprio arquivo em data\raw\.
REM Ao terminarem todas, funda com:
REM    .venv\Scripts\python.exe -m imoveis_jp.scraping.chaves_na_mao.merge_parts

setlocal
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
set MOD=imoveis_jp.scraping.chaves_na_mao.scraper

start "scrape 0/3" cmd /k "%PY% -m %MOD% --shard 0/3"
start "scrape 1/3" cmd /k "%PY% -m %MOD% --shard 1/3"
start "scrape 2/3" cmd /k "%PY% -m %MOD% --shard 2/3"

echo.
echo Tres janelas foram abertas (shards 0, 1 e 2 de 3).
echo Quando as tres terminarem (ou voce der Ctrl+C em cada uma), funda com:
echo    %PY% -m imoveis_jp.scraping.chaves_na_mao.merge_parts
echo.
pause
