"""Chunk documents, embed with multilingual-e5, cosine search with numpy."""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import settings

CHUNK_WORDS = 110   # AgriParam context is only 2048 tokens, so keep chunks small
OVERLAP_WORDS = 20


@dataclass
class Chunk:
    id: int
    source: str
    page: int | None
    text: str


_FOOTER = re.compile(r"\n?\s*\d{1,3}\s*\n\s*Wheat Cultivation in India\s*$|^\s*Wheat Cultivation in India\s*\n", re.M)


def _strip_running_text(text: str) -> str:
    """Remove the page number + 'Wheat Cultivation in India' footer that PDF extraction glues onto passages."""
    return _FOOTER.sub("\n", text)


def _read(path: Path) -> list[tuple[int | None, str]]:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return [(i + 1, _strip_running_text(p.extract_text() or "")) for i, p in enumerate(PdfReader(str(path)).pages)]
    if path.suffix.lower() in {".txt", ".md"}:
        return [(None, path.read_text(encoding="utf-8", errors="ignore"))]
    return []


def _split(text: str) -> list[str]:
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    step = CHUNK_WORDS - OVERLAP_WORDS
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, max(len(words), 1), step) if words[i:i + CHUNK_WORDS]]


def _split_markdown(text: str) -> list[str]:
    """Hand-written .md sources: one chunk per paragraph, prefixed with its '##' heading.

    Used for the cleaned-tables file, where each paragraph is one table row and must not be cut
    in half or merged with its neighbour.
    """
    pieces, heading = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if para.startswith("#"):
            heading = para.lstrip("# ").strip() if para.startswith("##") else heading
            continue
        for part in _split(para):
            pieces.append(f"{heading}. {part}" if heading else part)
    return pieces


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(settings.docs_dir.rglob("*")):
        if path.name.lower() == "readme.md":
            continue
        for page, text in _read(path):
            pieces = _split_markdown(text) if path.suffix.lower() == ".md" else _split(text)
            for piece in pieces:
                if len(piece.split()) >= 15:
                    chunks.append(Chunk(len(chunks), path.name, page, piece))
    return chunks


def _fingerprint(chunks: list[Chunk]) -> str:
    h = hashlib.sha1(settings.embed_model.encode())
    for c in chunks:
        h.update(c.text.encode())
    return h.hexdigest()


_GENERIC = set("""the and for are how what when which where who whom whose why does did can could should would will
shall may might must this that these those with from into onto over under about after before during between
through while than then there their they them you your our out any all each every some such not only own same
very just also use used using need needs needed much many more most less least best good type types kind way ways
get give make made take have has had been being was were its per one two three apply applied sow sown sowing grow
crop crops farmer farmers field fields wheat gehu time right proper please tell want know like help share harm
harming harmful affect
decide decides deciding choose select pick determine calculate find finding check know tell explain describe
suggest recommend recommended advise advice guide manage handle prepare start begin finish stop avoid prevent
protect save reduce increase improve raise maximize minimize ensure follow try work works working happen happens
cause causes problem problems issue issues method methods process step steps stage stages amount quantity
number level levels often usually generally normally always never again already still enough little lot lots
long short high low early late first last next small large big different another other different
farm farming grower growers land plot acre acres hectare hectares question answer information details detail
info today tomorrow week month year years season seasons this next
safe safely turn turning turns become becoming becomes look looks looking seem seems show shows showing see
appear appears signs sign way fall falls falling grows growing grown happen done doing going come comes
better worse good bad well fast slow quick quickly easy easily simple simply cheap cheaply cost costs dates date name names list few""".split())


def _strip_suffix(w: str) -> str:
    for s in ("ing", "ed", "es", "s", "ly"):
        if w.endswith(s) and len(w) - len(s) >= 4:
            return w[: -len(s)]
    return w


class Retriever:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        from sentence_transformers import CrossEncoder
        self.model = SentenceTransformer(settings.embed_model)
        # A cross-encoder reads question and passage together, so its scores are far better calibrated than
        # the bi-encoder's compressed 0.84-0.93 band: off-topic questions score strongly negative.
        self.ce = CrossEncoder(settings.cross_encoder)
        self.chunks: list[Chunk] = []
        self.emb = np.zeros((0, 1), dtype=np.float32)
        self.build()

    def build(self):
        self.chunks = load_chunks()
        self._build_vocab()
        if not self.chunks:
            return
        fp = _fingerprint(self.chunks)
        p = settings.index_path
        if p.exists():
            data = np.load(p, allow_pickle=False)
            if str(data["fp"]) == fp:
                self.emb = data["emb"]
                return
        self.emb = self.model.encode(
            [f"passage: {c.text}" for c in self.chunks],
            normalize_embeddings=True, batch_size=32, show_progress_bar=False,
        ).astype(np.float32)
        np.savez(p, emb=self.emb, fp=fp)

    def _build_vocab(self):
        seq = re.findall(r"[a-z]{3,}", " ".join(c.text for c in self.chunks).lower())
        toks = set(seq) | {a + b for a, b in zip(seq, seq[1:])}  # joined pairs: "ladybird" ~ "lady bird"
        self._p4 = {w[:4] for w in toks}
        self._p5 = {w[:5] for w in toks}

    def unseen_terms(self, question: str) -> list[str]:
        """Content words of the question that appear nowhere in the corpus (e.g. 'blast', 'subsidy').

        Similarity scores alone cannot tell "wheat blast" from other wheat disease passages, so a question
        whose key term the sources never mention is out of scope, however high its best chunk scores.
        """
        out = []
        for w in re.findall(r"[a-z]{4,}", question.lower()):
            if w in _GENERIC:
                continue
            s = _strip_suffix(w)
            if not ((s[:5] in self._p5) if len(s) >= 5 else (s in self._p4)):
                out.append(w)
        return out

    def search_rerank(self, query: str, k: int | None = None) -> list[tuple[Chunk, float, float]]:
        """Top candidates by e5, reordered by cross-encoder. Returns (chunk, e5_score, rerank_score)."""
        cands = self.search(query, max(settings.rerank_pool, k or settings.top_k))
        if not cands:
            return []
        ce = self.ce.predict([(query, c.text) for c, _ in cands])
        ranked = sorted(zip(cands, ce), key=lambda x: -x[1])[: k or settings.top_k]
        return [(c, e5, float(s)) for ((c, e5), s) in ranked]

    def search(self, query: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        scores = self.emb @ q
        idx = np.argsort(-scores)[: k or settings.top_k]
        return [(self.chunks[i], float(scores[i])) for i in idx]
