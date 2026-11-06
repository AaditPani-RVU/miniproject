"""Generator backends: local GGUF (llama.cpp / Metal), OpenAI-compatible server, or extractive."""
from .config import settings

SYSTEM = (
    "You are an agricultural advisory assistant for Indian farmers. "
    "Answer ONLY from the numbered context passages. If the context does not contain the answer, "
    "say you do not know and suggest contacting the local Krishi Vigyan Kendra (KVK). "
    "Be short and practical. Cite passages like [1]. "
    "Pick the passage that matches the crop condition in the question (zone, irrigated or rainfed, timely or late sown) "
    "and copy its numbers and units exactly; never calculate or guess a number. {lang_rule}"
)
LANG_RULE = {
    "en": "Reply in English.",
    "hi": "Reply in Hindi (Devanagari script).",
}
SCOPE_EN = ("wheat cultivation in India (varieties, sowing time and seed rate, fertilizer, irrigation, "
            "weeds, pests and diseases, harvest and storage)")
NO_ANSWER = {
    "en": f"I could not find this in my sources, which cover only {SCOPE_EN}. "
          "Please contact your nearest Krishi Vigyan Kendra (KVK).",
    "hi": "मुझे यह जानकारी मेरे स्रोतों में नहीं मिली। कृपया अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) से संपर्क करें।",
}
VAGUE_EN = ("I am not sure what you would like to know. Please tell me which part of wheat farming you are "
            "asking about, for example sowing time, fertilizer dose, irrigation, a weed, a pest or a disease.")
HELP_EN = (f"Namaste! I answer questions about {SCOPE_EN}, using ICAR documents. "
           "Please ask me a question about growing wheat, for example: "
           "\"What fertilizer dose should I use for timely sown wheat in western UP?\"")


STOP = ["<user>", "<assistant>", "</s>"]


def build_messages(question: str, passages: list[str], lang: str) -> list[dict]:
    """AgriParam's chat template drops the system role entirely, so everything goes in ONE user turn."""
    ctx = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    rules = SYSTEM.format(lang_rule=LANG_RULE.get(lang, LANG_RULE["en"]))
    tail = {"hi": "उत्तर केवल हिंदी (देवनागरी) में दें।", "en": "Answer in English."}.get(lang, "")
    content = f"{rules}\n\nContext:\n{ctx}\n\nQuestion: {question}\n{tail}"
    return [{"role": "user", "content": content}]


class Extractive:
    name = "extractive"

    def generate(self, question, passages, lang):
        return passages[0] if passages else NO_ANSWER["en"]


class GGUF:
    def __init__(self):
        from llama_cpp import Llama
        kw = dict(n_ctx=settings.n_ctx, n_gpu_layers=-1, verbose=False)
        if settings.gguf_path:
            self.llm = Llama(model_path=settings.gguf_path, **kw)
        else:
            self.llm = Llama.from_pretrained(settings.gguf_repo, settings.gguf_file, **kw)
        self.name = f"gguf:{settings.gguf_file}"

    def generate(self, question, passages, lang):
        out = self.llm.create_chat_completion(
            messages=build_messages(question, passages, lang),
            max_tokens=350, temperature=0.2, repeat_penalty=1.1, stop=STOP,
        )
        return out["choices"][0]["message"]["content"].strip()


class OpenAICompat:
    def __init__(self):
        self.name = f"openai:{settings.openai_model}"

    def generate(self, question, passages, lang):
        import httpx
        r = httpx.post(
            f"{settings.openai_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={"model": settings.openai_model, "temperature": 0.0, "max_tokens": 350, "stop": STOP,
                  "repeat_penalty": 1.1,  # llama-server extension; stops number-run degeneration
                  "messages": build_messages(question, passages, lang)},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def load_llm():
    return {"gguf": GGUF, "openai": OpenAICompat}.get(settings.llm_backend, Extractive)()
