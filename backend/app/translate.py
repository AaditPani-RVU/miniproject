"""Optional IndicTrans2 wrapper. Only loaded when TRANSLATOR=indictrans2."""
import re

from .config import settings

FLORES = {
    "en": "eng_Latn", "hi": "hin_Deva", "or": "ory_Orya", "kn": "kan_Knda",
    "ta": "tam_Taml", "te": "tel_Telu", "mr": "mar_Deva", "bn": "ben_Beng",
    "pa": "pan_Guru", "gu": "guj_Gujr",
}

# IndicTrans2 renders the farmer's word "khad" as "manure", which retrieves the wrong passages.
# In this corpus (chemical-fertilizer advice) it nearly always means fertilizer, so rewrite it first.
_HI_QUERY_GLOSSARY = {
    "रासायनिक खाद": "chemical fertilizer", "खाद": "fertilizer", "खादें": "fertilizer",
    # Crop-disease and agronomy terms IndicTrans2 gets wrong (e.g. "रतुआ" came out as "rheumatoid arthritis").
    "पीला रतुआ": "yellow rust (stripe rust) disease", "पीले रतुआ": "yellow rust (stripe rust) disease",
    "रतुआ": "rust disease", "कंडुआ": "smut disease", "दीमक": "termite", "माहू": "aphid",
    "गिरने": "lodging", "गिरना": "lodging", "सिंचाई": "irrigation", "बुवाई": "sowing",
    "जीरो टिलेज": "zero tillage", "उर्वरक": "fertilizer",
}

_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")


class Translator:
    def __init__(self):
        import torch
        from IndicTransToolkit.processor import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.proc = IndicProcessor(inference=True)
        self.models = {}
        for key, name in (("to_en", settings.it2_indic_en), ("from_en", settings.it2_en_indic)):
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True).to(self.device).eval()
            self.models[key] = (tok, mdl)

    def _run(self, key: str, sents: list[str], src: str, tgt: str) -> list[str]:
        tok, mdl = self.models[key]
        batch = self.proc.preprocess_batch(sents, src_lang=FLORES[src], tgt_lang=FLORES[tgt])
        inputs = tok(batch, truncation=True, padding="longest", max_length=256,
                     return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            # use_cache=False: IndicTrans2's remote code breaks on the newer transformers cache classes.
            out = mdl.generate(**inputs, use_cache=False, num_beams=5, num_return_sequences=1, max_length=256)
        dec = tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return self.proc.postprocess_batch(dec, lang=FLORES[tgt])

    def translate(self, text: str, src: str, tgt: str) -> str:
        if src == tgt or not text.strip():
            return text
        if src == "hi" and tgt == "en":
            for k, v in sorted(_HI_QUERY_GLOSSARY.items(), key=lambda kv: -len(kv[0])):
                text = re.sub(rf"(?<![\u0900-\u097F]){k}(?![\u0900-\u097F])", v, text)
        sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
        key = "to_en" if tgt == "en" else "from_en"
        return " ".join(self._run(key, sents, src, tgt))


def load_translator():
    if settings.translator == "indictrans2":
        return Translator()
    return None
