"""Roman-script Hindi ("Hinglish") support.

Farmers often type Hindi in Latin letters ("gehu mein khad kitna dalen"). IndicTrans2 expects Devanagari and
the embedder handles romanised Hindi poorly, so a small word-for-word glossary rewrites such questions into
plain English before retrieval. It is a heuristic, not a transliteration model, and only fires when at least two
glossary words are present.
"""
import re

_G = {
    # wheat / crops
    "gehu": "wheat", "gehun": "wheat", "gehoon": "wheat", "fasal": "crop", "beej": "seed", "bij": "seed",
    "dhan": "paddy", "chawal": "rice", "makka": "maize", "sarson": "mustard", "ganna": "sugarcane",
    "aloo": "potato", "alu": "potato", "chana": "gram", "kapas": "cotton", "sabzi": "vegetable", "sabji": "vegetable",
    # actions / question words
    "kitna": "how much", "kitni": "how much", "kitne": "how many", "kab": "when", "kaise": "how", "kya": "what",
    "kahan": "where", "kyun": "why", "kaun": "which", "konsi": "which", "kaunsi": "which", "kon": "which",
    "dalna": "apply", "dalen": "apply", "dale": "apply", "dalni": "apply", "daalna": "apply", "daalen": "apply",
    "lagana": "plant", "lagayen": "plant", "karna": "do", "karni": "do", "karen": "do", "kare": "do",
    "chahiye": "should", "chahie": "should", "hota": "is", "hai": "is", "hain": "are", "ho": "is",
    # topics
    "khad": "fertilizer", "khaad": "fertilizer", "urvarak": "fertilizer", "buwai": "sowing", "buai": "sowing",
    "bowai": "sowing", "bivai": "sowing", "sinchai": "irrigation", "sichai": "irrigation", "pani": "water",
    "dimak": "termite", "deemak": "termite", "mahu": "aphid", "ilaj": "treatment", "dawa": "medicine",
    "dawai": "medicine", "bimari": "disease", "rog": "disease", "kharpatwar": "weed", "katai": "harvest",
    "kataai": "harvest", "bhandaran": "storage", "bhandar": "storage", "mitti": "soil", "jaanch": "test",
    "janch": "test", "ropai": "transplanting", "kisam": "variety", "kism": "variety", "kismen": "varieties",
    "paidawar": "yield", "upaj": "yield", "mandi": "market", "bhav": "price", "bhaw": "price", "daam": "price",
    "mausam": "weather", "baarish": "rain", "barish": "rain", "aaj": "today", "kal": "tomorrow",
    # grammar glue
    "mein": "in", "me": "in", "ki": "of", "ka": "of", "ke": "of", "ko": "to", "se": "from", "par": "on",
    "aur": "and", "ya": "or", "nahi": "not", "kam": "less", "jyada": "more", "zyada": "more", "sahi": "right",
    # more spellings and everyday farm phrasing
    "pila": "yellow", "peela": "yellow", "peele": "yellow", "pili": "yellow", "ratua": "rust", "rataua": "rust",
    "rust": "rust", "baar": "times", "bar": "times", "lagta": "needed", "lagti": "needed", "lagte": "needed",
    "kami": "deficiency", "dur": "remove", "door": "remove", "liye": "for", "sabse": "most", "achhi": "good",
    "achha": "good", "accha": "good", "acchi": "good", "behtar": "best", "dena": "give", "dein": "give", "den": "give",
    "diya": "given", "gir": "lodging", "giri": "lodging", "girna": "lodging", "gira": "lodging", "paudhe": "plants", "paudha": "plant",
    "kharab": "damaged", "nuksan": "damage", "bachaye": "protect", "bachayen": "protect", "bachane": "protect",
    "rokne": "prevent", "roktham": "prevent", "upay": "remedy", "tarika": "method", "samasya": "problem",
    "keede": "insects", "keeda": "insect", "kide": "insects", "chhidkav": "spray", "chidkav": "spray",
    "chhidkaav": "spray", "spray": "spray", "tapman": "temperature", "garmi": "heat", "sardi": "cold",
    "dalein": "apply", "daalein": "apply", "daalen": "apply", "daliye": "apply", "daaliye": "apply",
    "lagayein": "plant", "lagaye": "plant", "boyen": "sow", "boye": "sow", "bona": "sow", "boni": "sow",
    "kategi": "harvest", "katne": "harvest", "kate": "harvest", "kab": "when", "kitani": "how much",
    "kisme": "varieties", "kismein": "varieties", "variety": "variety", "tab": "then", "agar": "if",
    "samay": "time", "pehle": "before", "baad": "after",
}
_DROP = {"karein", "kare", "karen", "karna", "karni", "karne", "kijiye", "kijie", "karte", "karta", "karti", "gaya",
         "gayi", "gaye", "raha", "rahi", "rahe", "hoga", "hogi", "honi", "hona", "to", "bhi", "hi", "na", "toh",
         "ji", "jee", "yeh", "ye", "woh", "wo", "is", "us", "isme", "usme", "kuch", "koi"}
_GLUE = {"mein", "me", "ki", "ka", "ke", "ko", "se", "par", "aur", "ya", "hai", "hain", "ho", "hota"}
_WORD = re.compile(r"[A-Za-z]+")


def looks_hinglish(text: str) -> bool:
    """True when Latin-script text has >=2 Hindi glossary words, at least one of them content-bearing."""
    words = [w.lower() for w in _WORD.findall(text)]
    hits = [w for w in words if w in _G or w in _DROP]
    return len(hits) >= 2 and any(w in _G and w not in _GLUE for w in hits)


def to_english(text: str) -> str:
    """Word-for-word rewrite. Unknown words (urea, DAP, zinc ...) pass through unchanged."""
    out = []
    for w in _WORD.findall(text):
        lw = w.lower()
        if lw in _DROP:
            continue
        out.append(_G.get(lw, w))
    return " ".join(out)
