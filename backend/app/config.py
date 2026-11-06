from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_backend: str = "gguf"
    gguf_repo: str = "arunmcops/AgriParam-GGUF"
    gguf_file: str = "AgriParam-Q4_K_M.gguf"
    gguf_path: str = ""
    n_ctx: int = 2048

    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = "none"
    openai_model: str = "agriparam"

    translator: str = "none"
    it2_en_indic: str = "ai4bharat/indictrans2-en-indic-dist-200M"
    it2_indic_en: str = "ai4bharat/indictrans2-indic-en-dist-200M"

    embed_model: str = "intfloat/multilingual-e5-small"
    cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_pool: int = 8        # e5 candidates passed to the cross-encoder
    ce_min: float = 0.0         # refuse below this reranker score (clearly unrelated question)
    ce_decisive: float = 4.5    # above this the passages clearly answer the question; vocabulary gaps are ignored
    top_k: int = 3
    min_score: float = 0.80  # weak safety net; the cross-encoder (ce_min, ce_decisive) is the main relevance gate

    docs_dir: Path = ROOT / "data" / "docs"
    index_path: Path = ROOT / "data" / "index.npz"
    frontend_dir: Path = ROOT / "frontend"


settings = Settings()

# AgriParam's Hindi output was unreliable (replied in English, invented facts), so only English is
# handled by the LLM. Everything else, including Hindi, goes through IndicTrans2 in and out.
NATIVE_LANGS = {"en"}

LANGUAGES = {
    "en": "English",
    "hi": "हिन्दी",
    "or": "ଓଡ଼ିଆ",
    "kn": "ಕನ್ನಡ",
    "ta": "தமிழ்",
    "te": "తెలుగు",
    "mr": "मराठी",
    "bn": "বাংলা",
    "pa": "ਪੰਜਾਬੀ",
    "gu": "ગુજરાતી",
}
