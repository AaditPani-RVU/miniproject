const $ = (id) => document.getElementById(id);
const chat = $("chat"), form = $("form"), q = $("q"), send = $("send"), langSel = $("lang");

const EXAMPLES = {
  en: ["What fertilizer dose should I use for timely sown wheat in western UP?", "How do I control termites in wheat?", "Which wheat varieties suit late sowing in eastern UP?"],
  hi: ["पश्चिमी यूपी में समय पर बोई गई गेहूं में कितना उर्वरक डालें?", "गेहूं में पीला रतुआ रोग का इलाज क्या है?", "गेहूं की फसल को कितनी सिंचाई चाहिए?"],
};

// Readable names for the files in data/docs, used in the citation line under each answer.
const SOURCE_TITLES = {
  "iiwbr_cleaned_tables.md": "ICAR-IIWBR, Wheat Cultivation in India (Extension Bulletin 52, 2014)",
  "iiwbr_wheat_cultivation_in_india_2014.pdf": "ICAR-IIWBR, Wheat Cultivation in India (Extension Bulletin 52, 2014)",
  "indian_farming_2023_wheat_fertilizer_practices_nwpz.pdf": "Indian Farming, Sept 2023: Farmer's fertilizer practices in wheat",
};
const sourceTitle = (f) => SOURCE_TITLES[f] || f;

let questionNo = 0;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function renderChips() {
  const box = $("chips");
  box.replaceChildren();
  for (const t of EXAMPLES[langSel.value] || EXAMPLES.en) {
    const c = el("button", "chip", t);
    c.type = "button";
    c.onclick = () => { q.value = t; form.requestSubmit(); };
    box.append(c);
  }
}

async function init() {
  try {
    const [h, langs] = await Promise.all([
      fetch("/api/health").then((r) => r.json()),
      fetch("/api/languages").then((r) => r.json()),
    ]);
    for (const l of langs) {
      const o = el("option", "", l.name + (l.enabled ? "" : " (needs translator)"));
      o.value = l.code; o.disabled = !l.enabled;
      langSel.append(o);
    }
    $("status").textContent = `${h.chunks} passages indexed`;
    $("status").classList.add("ok");
  } catch {
    $("status").textContent = "backend offline";
    $("status").classList.add("bad");
  }
  renderChips();
}

function addUser(text) {
  $("empty")?.remove();
  questionNo += 1;
  const m = el("div", "msg user");
  m.append(el("span", "no", "Q." + String(questionNo).padStart(2, "0")), el("span", "q", text));
  chat.append(m);
  m.scrollIntoView({ behavior: "smooth", block: "end" });
}

function typing() {
  const m = el("div", "msg bot");
  const t = el("span", "typing");
  t.append(el("i"), el("i"), el("i"));
  m.append(t);
  chat.append(m);
  m.scrollIntoView({ behavior: "smooth", block: "end" });
  return m;
}

// "From the source: <Section heading>. <text>"  ->  { kicker, body } (English answers only).
function splitQuote(answer) {
  const prefix = "From the source: ";
  if (!answer.startsWith(prefix)) return { kicker: "", body: answer };
  const rest = answer.slice(prefix.length);
  const i = rest.indexOf(". ");
  if (i > 0 && i < 90 && rest.slice(0, i).split(" ").length <= 12) {
    return { kicker: rest.slice(0, i), body: rest.slice(i + 2) };
  }
  return { kicker: "", body: rest };
}

function renderBot(node, r) {
  node.replaceChildren();
  const help = r.mode === "help";
  node.classList.toggle("ungrounded", !r.grounded && !help);
  node.classList.toggle("help", help);

  const label = help ? "Kisan Sahayak"
    : !r.grounded ? "Not in my sources"
    : "Exact text from the source";
  node.append(el("span", "stamp", label));

  const { kicker, body } = r.grounded && r.lang === "en" ? splitQuote(r.answer) : { kicker: "", body: r.answer };
  if (kicker) node.append(el("p", "kicker", kicker));
  node.append(el("div", "answer", body));

  if (r.grounded && r.sources.length) {
    const s = r.sources[0];
    const cite = el("p", "cite");
    cite.append(el("b", "", "Source "), document.createTextNode(sourceTitle(s.source) + (s.page ? ", p." + s.page : "")));
    node.append(cite);
  }

  if (r.summary) {
    const sm = el("div", "summary");
    sm.append(el("b", "", "AI summary · check it against the source text above"));
    sm.append(el("span", "", r.summary));
    node.append(sm);
  }
  if (r.english_answer) {
    const e = el("details");
    e.append(el("summary", "", "English version (to check the translation)"));
    const box = el("div", "src");
    if (r.english_question) box.append(el("b", "", "Question as understood: " + r.english_question));
    box.append(el("span", "", r.english_answer));
    e.append(box);
    node.append(e);
  }
  if (r.sources.length && r.grounded) {
    const d = el("details");
    d.append(el("summary", "", `Other passages consulted (${r.sources.length})`));
    for (const s of r.sources) {
      const box = el("div", "src");
      box.append(el("b", "", `[${s.id}] ${sourceTitle(s.source)}${s.page ? " · p." + s.page : ""} · match ${s.score}`));
      box.append(el("span", "", s.text));
      d.append(box);
    }
    node.append(d);
  }
  const t = r.timings, secs = (ms) => (ms / 1000).toFixed(1) + "s";
  const trans = t.translate_in_ms + t.translate_out_ms;
  node.append(el("div", "meta", `search ${secs(t.retrieve_ms)}` + (t.generate_ms ? ` · model ${secs(t.generate_ms)}` : "") +
    (trans ? ` · translation ${secs(trans)}` : "")));
  node.scrollIntoView({ behavior: "smooth", block: "end" });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = q.value.trim();
  if (!text || send.disabled) return;
  q.value = ""; q.style.height = "auto";
  addUser(text);
  send.disabled = true;
  const node = typing();
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, lang: langSel.value }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    renderBot(node, await res.json());
  } catch (err) {
    node.classList.add("ungrounded");
    node.replaceChildren(el("span", "stamp", "Something went wrong"), el("div", "answer", err.message));
  } finally {
    send.disabled = false; q.focus();
  }
});

q.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
q.addEventListener("input", () => { q.style.height = "auto"; q.style.height = q.scrollHeight + "px"; });
langSel.addEventListener("change", renderChips);

init();
