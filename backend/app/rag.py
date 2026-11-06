"""End-to-end pipeline: (translate in) -> retrieve + rerank -> gates -> generate -> guards -> (translate out)."""
import re
import time

from . import hinglish
from .config import settings
from .llm import HELP_EN, NO_ANSWER, VAGUE_EN, load_llm
from .retrieval import _GENERIC, Retriever, _strip_suffix
from .translate import load_translator

MAX_CHARS = 500
_QSTOP = set("what how when where which why who whom does did can could should would will shall may might the this "
             "that these those yeh woh kya hain hai please tell about and for you your its any there here one ones "
             "from with have has are was were not tell show".split())
_GREETING = re.compile(
    r"^\W*(hi|hello|hey|namaste|namaskar|good (morning|evening|afternoon)|thanks|thank you|thankyou|dhanyavaad|"
    r"dhanyawad|shukriya|ok|okay|help|who are you|what can you do|what do you do|नमस्ते|नमस्कार|धन्यवाद|शुक्रिया|"
    r"आप कौन हैं|आप क्या कर सकते हैं|मदद)\W*$", re.I)


def _letters(text: str) -> list[str]:
    return [ch for ch in text if ch.isalpha()]


class Pipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = load_llm()
        self.translator = load_translator()

    def supported(self, lang: str) -> bool:
        return lang == "en" or self.translator is not None

    def _reply(self, text, lang, mode, t, extra=None):
        """Fixed (non-generated) reply, translated when the user's language is not English."""
        english, out = text, text
        if lang != "en" and self.translator is not None:
            out = self.translator.translate(text, "en", lang)
        res = {"answer": out, "lang": lang, "grounded": False, "best_score": 0.0, "rerank_score": None,
               "unsupported_numbers": [], "mode": mode, "summary": None, "unseen_terms": [], "english_question": None,
               "english_answer": english if out != english else None, "sources": [], "timings": t,
               "model": self.llm.name}
        res.update(extra or {})
        return res

    def ask(self, question: str, lang: str = "en", top_k: int | None = None) -> dict:
        t = {"translate_in_ms": 0, "retrieve_ms": 0, "generate_ms": 0, "translate_out_ms": 0}
        text = re.sub(r"\s+", " ", question or "").strip()[:MAX_CHARS]
        letters = _letters(text)

        # --- input edge cases: empty/junk, greetings, script and language detection -----------------
        has_nonlatin = any(ord(ch) > 0x24F for ch in letters)
        out_lang = "hi" if (lang == "en" and has_nonlatin) else lang   # typed Devanagari -> answer in Hindi
        if out_lang != "en" and self.translator is None:
            out_lang = "en"
        if len(letters) < 2 or _GREETING.match(text):
            return self._reply(HELP_EN, out_lang, "help", t)
        if has_nonlatin and self.translator is None:
            return self._reply(HELP_EN, "en", "help", t)

        s = time.perf_counter()
        src = (lang if lang != "en" else "hi") if has_nonlatin else "en"
        q = text
        if src != "en":
            q = self.translator.translate(text, src, "en")
        elif hinglish.looks_hinglish(text):
            q = hinglish.to_english(text)
        t["translate_in_ms"] = _ms(s)

        # A question with no topic word at all ("what is this?") cannot be searched: ask for detail instead.
        if not [w for w in re.findall(r"[a-z]{3,}", q.lower()) if w not in _QSTOP]:
            return self._reply(VAGUE_EN, out_lang, "help", t, {"english_question": q if q != text else None})

        # --- retrieval and gates -----------------------------------------------------------------
        s = time.perf_counter()
        # The corpus is wheat-only, so every question is implicitly about wheat ("when to apply urea?").
        # Say so in the search query, otherwise the reranker marks wheat passages as off-topic.
        search_q = q if re.search(r"\bwheat\b", q, re.I) else "For wheat crop: " + q
        hits = self.retriever.search_rerank(search_q, top_k)
        t["retrieve_ms"] = _ms(s)
        best = max((e5 for _, e5, _ in hits), default=0.0)
        ce = hits[0][2] if hits else -99.0
        crop = out_of_scope_crop(text, q) or out_of_scope_live(text, q)
        unseen = self.retriever.unseen_terms(q)
        # The reranker is the main relevance signal; the vocabulary gap only matters when it is not decisive.
        # "Decisive" is judged on the question as the user asked it: the implicit "wheat" prefix can make an
        # unrelated question ("capital of India?") look relevant, and must not override a missing key word.
        ce_asked = ce if search_q == q else float(self.retriever.ce.predict([(q, hits[0][0].text)])[0]) if hits else ce
        unseen_blocks = bool(unseen) and ce_asked < settings.ce_decisive
        grounded = bool(hits) and best >= settings.min_score and ce >= settings.ce_min and not crop and not unseen_blocks

        s = time.perf_counter()
        unsupported: list[str] = []
        mode = "generated" if grounded else "refused"
        passages = [c.text for c, _, _ in hits]
        summary = None
        if grounded:
            # Source-first: the answer is always the exact text of the best-ranked passage. On our test sets
            # that passage held the right fact 95% of the time, versus 76% for the model's own wording, and
            # no guard setting made the model's text more accurate than the quote.
            mode = "quoted_source"
            answer = QUOTE_PREFIX + passages[0]
            try:
                draft = self.llm.generate(q, passages, "en")
            except Exception:   # model server down or timed out: the quote alone is still a full answer
                draft = ""
            # The model's plain-language summary is shown only if it stays strictly inside the best passage.
            unsupported = unsupported_numbers(draft, [passages[0]]) if draft else []
            if draft and not unsupported and not is_echo(draft, q) and novel_ratio(draft, passages[0]) <= 0.10:
                summary = draft
        else:
            answer = NO_ANSWER["en"]
        t["generate_ms"] = _ms(s)

        english_answer = answer if out_lang != "en" else None
        s = time.perf_counter()
        if out_lang != "en":
            answer = self.translator.translate(answer, "en", out_lang)
            summary = None   # unverified model text is not translated; the source quote is the answer
        t["translate_out_ms"] = _ms(s)

        return {
            "answer": answer,
            "lang": out_lang,
            "grounded": grounded,
            "best_score": round(best, 3),
            "rerank_score": round(ce, 2),
            "unsupported_numbers": unsupported,
            "mode": mode,
            "summary": summary,
            "unseen_terms": unseen,
            "english_question": q if q != question else None,
            "english_answer": english_answer,
            "sources": [
                {"id": i + 1, "source": c.source, "page": c.page, "score": round(e5, 3), "rerank": round(r, 2),
                 "text": c.text}
                for i, (c, e5, r) in enumerate(hits)
            ],
            "timings": t,
            "model": self.llm.name,
        }


# The corpus covers wheat only. Retrieval scores are compressed (wheat passages score 0.84+ for any
# "sowing" question), so a threshold alone cannot tell "paddy" from "wheat". Refuse other crops by name.
_OTHER_CROPS = [
    "paddy", "rice", "dhan", "maize", "corn", "makka", "mustard", "sarson", "potato", "aloo", "sugarcane",
    "ganna", "cotton", "kapas", "gram", "chana", "pea", "lentil", "masoor", "tomato", "onion", "pyaz",
    "bajra", "jowar", "barley", "arhar", "tur", "soybean", "groundnut", "banana", "mango", "brinjal",
    "धान", "चावल", "मक्का", "सरसों", "आलू", "गन्ना", "कपास", "चना", "मटर", "मसूर", "टमाटर", "प्याज़", "प्याज",
    "बाजरा", "ज्वार", "जौ", "अरहर", "तुअर", "सोयाबीन", "मूंगफली", "केला", "बैंगन",
]
_WHEAT = ("wheat", "gehu", "gehun", "गेहूं", "गेहूँ", "गेहु")
_OTHER_RE = re.compile(r"(?<![\w\u0900-\u097F])(" + "|".join(map(re.escape, _OTHER_CROPS)) + r")(?![\w\u0900-\u097F])", re.I)
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december"]


# Residue questions ("burning paddy straw before wheat") are in scope: the bulletin covers them.
_RESIDUE = ("straw", "residue", "stubble", "parali", "stalk", "पराली", "पुआल", "अवशेष", "ठूंठ")
# Live or commercial data that no agronomy bulletin contains.
_LIVE_RE = re.compile(
    r"\b(mandi|market|msp|support price|price|prices|bhav|bhaw|rate today|today|tomorrow|weather|forecast|"
    r"insurance|pmfby|kisan credit|subsidy|loan|"
    # general knowledge and chit-chat: the reranker cannot reject these because the bulletin itself
    # mentions India and population, so name the usual probes explicitly
    r"capital|president|prime minister|chief minister|population|cricket|football|movie|film|song|joke|poem|"
    r"story|recipe|bake|cook|translate|election|celebrity|news)\b|"
    r"मंडी|बाज़ार|बाजार|भाव|कीमत|समर्थन मूल्य|मौसम|बीमा|सब्सिडी|ऋण|राजधानी|प्रधानमंत्री|मुख्यमंत्री|जनसंख्या|क्रिकेट|फिल्म|गाना|चुटकुला|कहानी|रेसिपी",
    re.I)


def out_of_scope_live(*questions: str) -> str | None:
    m = _LIVE_RE.search(" ".join(questions))
    return m.group(1) if m and m.group(1) else (m.group(0) if m else None)


def out_of_scope_crop(*questions: str) -> str | None:
    """Return the first non-wheat crop named in the question(s), unless wheat is named too."""
    text = " ".join(questions).lower()
    if any(w in text for w in _WHEAT) or any(w in text for w in _RESIDUE):
        return None
    m = _OTHER_RE.search(text)
    return m.group(1) if m else None


_META = re.compile(
    r"^\W*(the|this|your) (question|query)\b|\b(the|an) answer (will|is going to) be\b|^\W*tell me\b|\byou (asked|are asking)\b",
    re.I)


def is_echo(answer: str, question: str) -> bool:
    """The model repeated the question, or replied with a question of its own, instead of answering."""
    if answer.strip().endswith(("?", "？")):
        return True
    # Meta-answers that talk about the question instead of answering it.
    if _META.search(answer):
        return True
    a = set(re.findall(r"\w+", answer.lower()))
    q = set(re.findall(r"\w+", question.lower()))
    return bool(a) and len(a & q) / len(a) >= 0.8


QUOTE_PREFIX = "From the source: "


def novel_ratio(answer: str, passage: str) -> float:
    """Share of the answer's content words that do not occur in the passage (0 = fully inside it)."""
    ctx = {_strip_suffix(w)[:5] for w in re.findall(r"[a-z]{3,}", passage.lower())}
    ws = [w for w in re.findall(r"[a-z]{4,}", re.sub(r"\[\d+\]", "", answer).lower()) if w not in _GENERIC]
    return sum(_strip_suffix(w)[:5] not in ctx for w in ws) / len(ws) if ws else 1.0


_NUM = re.compile(r"\d+(?:\.\d+)?")
_UNIT_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kg|g|gm|ml|l|litres?|liters?|%|per ?cent|t/ha|q/ha)(?![a-z])", re.I)


def unsupported_numbers(answer: str, passages: list[str]) -> list[str]:
    """Numbers with 2+ digits (or decimals) in the answer that appear nowhere in the context."""
    answer = re.sub(r"\[\d+\]", "", answer)  # citation markers like [1]
    ctx = set(_NUM.findall(" ".join(passages)))
    bad = {n for n in _NUM.findall(answer) if (len(n) >= 2 or "." in n) and n not in ctx}
    # Single-digit doses ("5 kg zinc sulphate") only count when they carry a unit.
    bad |= {n for n in _UNIT_NUM.findall(answer) if n not in ctx}
    low_ctx = " ".join(passages).lower()
    # Product or brand names the sources never mention (e.g. invented "Azotacha", misspelt "Folucin").
    bad |= {m for m in re.findall(r"(?<![.!?:]\s)(?<!^)\b([A-Z][a-z]{4,})\b", answer) if m.lower() not in low_ctx}
    bad |= {m.title() for m in _MONTHS if re.search(rf"\b{m}\b", answer.lower()) and m not in low_ctx}
    return sorted(bad)


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
