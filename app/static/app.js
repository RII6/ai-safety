const form = document.getElementById("scan-form");
const input = document.getElementById("repo");
const btn = document.getElementById("scan-btn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");

function setStatus(html, isError) {
  statusEl.hidden = false;
  statusEl.className = "status" + (isError ? " error" : "");
  statusEl.innerHTML = html;
}

function fieldChips(fields) {
  return Object.entries(fields)
    .map(([k, v]) => `<span>${k} <b>${v}</b></span>`)
    .join("");
}

function render(data) {
  const v = data.verdict;
  const metrics = data.metrics
    .map(
      (m) => `
      <div class="metric">
        <h3>${m.title}</h3>
        <div class="headline">${m.headline}</div>
        <p class="what">${m.what}</p>
        <p class="read">${m.read}</p>
        <div class="fields">${fieldChips(m.fields)}</div>
      </div>`
    )
    .join("");

  const meta = data.meta;
  resultEl.innerHTML = `
    <div class="verdict ${v.code}">
      <div class="repo">${data.repo}${data.from_cache ? " · cached" : ""}</div>
      <span class="badge">${v.label}</span>
      <p><span class="label">Diagnosis</span><br>${v.diagnosis}</p>
      <p><span class="label">Recommendation</span><br>${v.recommendation}</p>
    </div>
    ${metrics}
    <div class="meta">
      ${meta.params ? (meta.params / 1e6).toFixed(0) + "M params · " : ""}
      ${meta.sample}/${meta.sample} prompts · ${meta.device}/${meta.dtype} · ${meta.elapsed_s}s
    </div>`;
  resultEl.hidden = false;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const repo = input.value.trim();
  if (!repo) return;

  btn.disabled = true;
  resultEl.hidden = true;
  setStatus(`<span class="spinner"></span>Scanning <b>${repo}</b> — loading the model and probing internal state…`);

  try {
    const res = await fetch("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo }),
    });
    const data = await res.json();
    if (!res.ok) {
      setStatus(data.error || `Request failed (${res.status}).`, true);
    } else {
      statusEl.hidden = true;
      render(data);
    }
  } catch (err) {
    setStatus("Network error: " + err.message, true);
  } finally {
    btn.disabled = false;
  }
});
