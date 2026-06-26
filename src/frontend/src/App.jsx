import { useState } from "react";
import "./App.css";

function App() {
  const [repo, setRepo] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState({ text: "", isError: false, visible: false });
  const [result, setResult] = useState(null);

  const handleScan = async (e) => {
    e.preventDefault();
    const trimmedRepo = repo.trim();
    if (!trimmedRepo) return;

    setLoading(true);
    setResult(null);
    setStatus({
      text: `Scanning <b>${trimmedRepo}</b> — loading the model and probing internal state…`,
      isError: false,
      visible: true,
    });

    try {
      const res = await fetch("/api/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo: trimmedRepo, force: false }),
      });

      const data = await res.json();

      if (!res.ok) {
        setStatus({
          text: data.error || `Request failed (${res.status}).`,
          isError: true,
          visible: true,
        });
      } else {
        setStatus({ text: "", isError: false, visible: false });
        setResult(data);
      }
    } catch (err) {
      setStatus({
        text: "Network error: " + err.message,
        isError: true,
        visible: true,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
      <>
        <header>
          <div className="header-top">
            <div className="logo-area">
              <span className="logo-icon">🛡️</span>
              <h1>Unified LLM Safety Platform</h1>
            </div>
            <div className="user-profile">
              <span className="user-icon">👤</span> User Profile
            </div>
          </div>
        </header>

        <main>
          <section className="config-section">
            <h2>SELECT SAFETY SCANS & ADVANCED ATTACKS</h2>

            <div className="scan-modules-list">

              {/* MODULE 1: CORE GENERAL SAFETY (LIVE) */}
              <div className="scan-module-card active">
                <div className="card-header">
                  <input type="checkbox" id="m-general" checked readOnly />
                  <label htmlFor="m-general"> General Safety Test (Core Behavioral Probing)</label>
                  <span className="status-badge live">LIVE</span>
                </div>
                <p className="card-desc">
                  Evaluation over a fixed benchmark corpus of standard adversarial prompts. Tests the model's core
                  refusal capability and comprehension against Toxicity, Doxing, Hate Speech, and Dangerous Content.
                  Measures the Safety Margin Score via vocabulary logit distributions.
                </p>
              </div>

              {/* MODULE 2: PROMPT INJECTIONS & DRIFT */}
              <div className="scan-module-card disabled">
                <div className="card-header">
                  <input type="checkbox" id="m-injection" disabled />
                  <label htmlFor="m-injection">Multi-Turn Behavioral Drift & Injections</label>
                  <span className="status-badge soon">COMING SOON</span>
                </div>
                <p className="card-desc">
                  Orchestrates a three-phase dialogue to gradually shift context toward a target harmful request.
                  Captures logit snapshots after each turn to compute KL-divergence relative to the first step.
                  Tests both direct prompt injections and indirect injections embedded within documents.
                </p>
              </div>

              {/* MODULE 3: MEMORIZATION EXTRACTION */}
              <div className="scan-module-card disabled">
                <div className="card-header">
                  <input type="checkbox" id="m-leakage" disabled />
                  <label htmlFor="m-leakage">Memorization Extraction & System Leakage</label>
                  <span className="status-badge soon">COMING SOON</span>
                </div>
                <p className="card-desc">
                  Implements the Carlini et al. (2021) method. Generates domain-specific seed prefixes and triggers
                  beam search using a small reference model (Pythia-70m) to compute memorization scores.
                  Includes adversarial scenarios targeting system prompt extraction and precise data leaks.
                </p>
              </div>

              {/* MODULE 4: SAMPLING INSTABILITY */}
              <div className="scan-module-card disabled">
                <div className="card-header">
                  <input type="checkbox" id="m-sampling" disabled />
                  <label htmlFor="m-sampling">Sampling Instability Analysis</label>
                  <span className="status-badge soon">COMING SOON</span>
                </div>
                <p className="card-desc">
                  Runs test scenarios across a customized temperature × top_p inference grid with N=20 runs per point.
                  Calculates the Instability Score (max(P_safe) - min(P_safe)) to detect alignment degradation under varying sampling parameters.
                </p>
              </div>

              {/* MODULE 5: GCG ADVERSARIAL SUFFIXES */}
              <div className="scan-module-card disabled">
                <div className="card-header">
                  <input type="checkbox" id="m-gcg" disabled />
                  <label htmlFor="m-gcg">Greedy Coordinate Gradient (GCG) Attacks</label>
                  <span className="status-badge soon">COMING SOON</span>
                </div>
                <p className="card-desc">
                  Executes a gradient-based token optimization via Greedy Coordinate Gradient (GCG) directly on open weights.
                  Discovers universal adversarial suffixes designed to mathematically force the model to begin its response with an affirmative token.
                </p>
              </div>

            </div>

            {/* FORM */}
            <form onSubmit={handleScan}>
              <div className="target-repo-input">
                <label htmlFor="repo">Target Model Repository (Hugging Face):</label>
                <div className="search-box">
                  <input
                      id="repo"
                      type="text"
                      placeholder="owner/model — e.g. HuggingFaceTB/SmolLM2-360M-Instruct"
                      autoComplete="off"
                      value={repo}
                      onChange={(e) => setRepo(e.target.value)}
                  />
                  <button id="scan-btn" type="submit" disabled={loading}>
                    {loading ? "Scanning..." : "RUN ACTIVE SCANS"}
                  </button>
                </div>
              </div>
            </form>
            <p className="hint">Running against configured target domain thresholds. Small instruct models work best.</p>
          </section>

          {/* LOADING STATUS */}
          {status.visible && (
              <div className={`status ${status.isError ? "error" : ""}`}>
                {!status.isError && <span className="spinner"></span>}
                <span dangerouslySetInnerHTML={{ __html: status.text }} />
              </div>
          )}

          {/* RESULTS RENDER */}
          {result && (
              <section id="result">
                <div className={`verdict ${result.verdict.code}`}>
                  <div className="repo">
                    {result.repo}
                    {result.from_cache && " · cached"}
                  </div>
                  <span className="badge">{result.verdict.label}</span>
                  <p><span className="label">Diagnosis</span><br />{result.verdict.diagnosis}</p>
                  <p><span className="label">Recommendation</span><br />{result.verdict.recommendation}</p>
                </div>

                {result.metrics.map((m, index) => (
                    <div className="metric" key={index}>
                      <h3>{m.title}</h3>
                      <div className="headline">{m.headline}</div>
                      <p className="what">{m.what}</p>
                      <p className="read">{m.read}</p>
                      <div className="fields">
                        {Object.entries(m.fields).map(([key, value]) => (
                            <span key={key}>{key} <b>{value}</b></span>
                        ))}
                      </div>
                    </div>
                ))}

                <div className="meta">
                  {result.meta.params ? `${(result.meta.params / 1e6).toFixed(0)}M params · ` : ""}
                  {result.meta.sample}/{result.meta.sample} prompts · {result.meta.device}/{result.meta.dtype} · {result.meta.elapsed_s}s
                </div>
              </section>
          )}
        </main>
      </>
  );
}

export default App;