try:
    from .scraper import ScraperEngine
except ImportError:
    from scraper import ScraperEngine

if __name__ == "__main__":
    engine = ScraperEngine()
    engine.run()
