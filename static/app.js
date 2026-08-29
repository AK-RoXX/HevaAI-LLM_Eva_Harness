const $ = (id) => document.getElementById(id);
const file = $("file");
const drop = $("drop");
const result = $("result");
async function refresh() {
  try {
    const h = await fetch("/health").then((r) => r.json());
    $("status").textContent = h.llm_configured
      ? "● API ready"
      : "● Gemini key missing";
    $("status").classList.toggle("ok", h.llm_configured);
    const d = await fetch("/documents").then((r) => r.json());
    $("docCount").textContent =
      `${d.length} document${d.length === 1 ? "" : "s"}`;
    $("docs").innerHTML = d
      .map(
        (x) =>
          `<div class="doc">${escapeHtml(x.filename)}<small>${x.document_id}</small></div>`,
      )
      .join("");
  } catch (e) {
    $("status").textContent = "● API unavailable";
  }
}
file.onchange = () => upload(file.files[0]);
drop.ondragover = (e) => {
  e.preventDefault();
  drop.style.borderColor = "var(--accent)";
};
drop.ondragleave = () => (drop.style.borderColor = "");
drop.ondrop = (e) => {
  e.preventDefault();
  drop.style.borderColor = "";
  upload(e.dataTransfer.files[0]);
};
async function upload(f) {
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  drop.querySelector("strong").textContent = "Uploading…";
  try {
    const r = await fetch("/documents", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw Error(j.detail || "Upload failed");
    drop.querySelector("strong").textContent = `Uploaded · ${j.chunks} chunks`;
    refresh();
  } catch (e) {
    drop.querySelector("strong").textContent = e.message;
  }
}
$("ask").onclick = async () => {
  const q = $("question").value.trim();
  if (!q) return;
  $("ask").disabled = true;
  $("ask").textContent = "Thinking…";
  result.classList.remove("hidden");
  result.innerHTML = "<span>Retrieving evidence and generating answer…</span>";
  try {
    const r = await fetch("/qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    const j = await r.json();
    if (!r.ok) throw Error(j.detail || "Request failed");
    result.innerHTML = `<div class="answer">${escapeHtml(j.answer)}</div><div class="meta"><span class="pill">Confidence ${(j.confidence * 100).toFixed(0)}%</span><span class="pill">${j.abstained ? "Abstained" : "Answered"}</span><span class="pill">${escapeHtml(j.model)}</span></div>${j.citations.map((c) => `<div class="cite"><b>${escapeHtml(c.chunk_id)} · relevance ${(c.relevance * 100).toFixed(1)}%</b><p>${escapeHtml(c.text)}</p></div>`).join("")}`;
  } catch (e) {
    result.innerHTML = `<span class="error">${escapeHtml(e.message)}</span>`;
  } finally {
    $("ask").disabled = false;
    $("ask").innerHTML = "Ask the documents <span>↗</span>";
  }
};
function escapeHtml(s) {
  return String(s).replace(
    /[&<>'"]/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[
        c
      ],
  );
}
refresh();
