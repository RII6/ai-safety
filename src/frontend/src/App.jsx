import { useState } from "react";
import "./App.css";

function App() {
  const [repo, setRepo] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState({ text: "", isError: false, visible: false });
  const [result, setResult] = useState(null);

  const [isGeneralOpen, setIsGeneralOpen] = useState(false);
  const [isInjectionOpen, setIsInjectionOpen] = useState(false);
  const [isLeakageOpen, setIsLeakageOpen] = useState(false);
  const [isSamplingOpen, setIsSamplingOpen] = useState(false);
  const [isGCDOpen, setIsGCDOpen] = useState(false);

  const [scanGeneral, setScanGeneral] = useState(true);
  const [scanInjection, setScanInjection] = useState(false);

  const [openMetrics, setOpenMetrics] = useState({});

  const toggleMetric = (index) => {
    setOpenMetrics((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  const handleScan = async (e) => {
    e.preventDefault();
    const trimmedRepo = repo.trim();
    if (!trimmedRepo) return;

    setLoading(true);
    setResult(null);
    setOpenMetrics({});
    setStatus({
      text: `Scanning <b>${trimmedRepo}</b> — loading the model and probing internal state…`,
      isError: false,
      visible: true,
    });

    try {
      const selectedModules = [];
      if (scanGeneral) selectedModules.push("general");
      if (scanInjection) selectedModules.push("prompt_injections");

      const res = await fetch("/api/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repo: trimmedRepo,
          force: true,
          modules: selectedModules
        }),
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

  const getColorClass = (text) => {
    const lower = String(text).toLowerCase();
    if (lower.includes("fail") || lower.includes("high") || lower.includes("danger") || lower.includes("100%")) {
      return "status-danger";
    }
    if (lower.includes("low") || lower.includes("passed") || lower.includes("94") || lower.includes("block")) {
      return "status-success";
    }
    return "status-warning";
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

              <div className={`scan-module-card ${scanGeneral ? "active" : ""}`}>
                <div className="card-header-clickable" onClick={() => setIsGeneralOpen(!isGeneralOpen)}>
                  <div className="checkbox-label-wrapper" onClick={(e) => e.stopPropagation()}>
                    <input
                        type="checkbox"
                        id="m-general"
                        checked={scanGeneral}
                        onChange={(e) => setScanGeneral(e.target.checked)}
                    />
                    <label htmlFor="m-general"> General Safety Test (Core Behavioral Probing)</label>
                  </div>
                  <div className="badge-arrow-wrapper">
                    <span className="status-badge live">LIVE</span>
                    <span className={`arrow-icon ${isGeneralOpen ? "rotated" : ""}`}>▼</span>
                  </div>
                </div>

                <div className="collapsible-content" style={{ maxHeight: isGeneralOpen ? "300px" : "0px", opacity: isGeneralOpen ? 1 : 0 }}>
                  <p className="card-desc">
                    Evaluation over a fixed benchmark corpus of standard adversarial prompts. Tests the model's core
                    refusal capability and comprehension against Toxicity, Doxing, Hate Speech, and Dangerous Content.
                    Measures the Safety Margin Score via vocabulary logit distributions.
                  </p>
                </div>
              </div>

              <div className={`scan-module-card ${scanInjection ? "active" : ""}`}>
                <div className="card-header-clickable" onClick={() => setIsInjectionOpen(!isInjectionOpen)}>
                  <div className="checkbox-label-wrapper" onClick={(e) => e.stopPropagation()}>
                    <input
                        type="checkbox"
                        id="m-injection"
                        checked={scanInjection}
                        onChange={(e) => setScanInjection(e.target.checked)}
                    />
                    <label htmlFor="m-injection"> Multi-Turn Behavioral Drift & Injections</label>
                  </div>
                  <div className="badge-arrow-wrapper">
                    <span className="status-badge live">LIVE</span>
                    <span className={`arrow-icon ${isInjectionOpen ? "rotated" : ""}`}>▼</span>
                  </div>
                </div>

                <div className="collapsible-content" style={{ maxHeight: isInjectionOpen ? "300px" : "0px", opacity: isInjectionOpen ? 1 : 0 }}>
                  <p className="card-desc">
                    Orchestrates a three-phase dialogue to gradually shift context toward a target harmful request.
                    Captures logit snapshots after each turn to compute KL-divergence relative to the first step.
                    Tests both direct prompt injections and indirect injections embedded within documents.
                  </p>
                </div>
              </div>

              <div className="scan-module-card disabled">
                <div className="card-header-clickable" onClick={() => setIsLeakageOpen(!isLeakageOpen)}>
                  <div className="checkbox-label-wrapper">
                    <input type="checkbox" id="m-leakage" disabled />
                    <label htmlFor="m-leakage">Memorization Extraction & System Leakage</label>
                  </div>
                  <div className="badge-arrow-wrapper">
                    <span className="status-badge soon">COMING SOON</span>
                    <span className={`arrow-icon ${isLeakageOpen ? "rotated" : ""}`}>▼</span>
                  </div>
                </div>
                <div className="collapsible-content" style={{ maxHeight: isLeakageOpen ? "300px" : "0px", opacity: isLeakageOpen ? 1 : 0 }}>
                  <p className="card-desc">
                    Implements the Carlini et al. (2021) method. Generates domain-specific seed prefixes and triggers
                    beam search using a small reference model (Pythia-70m) to compute memorization scores.
                  </p>
                </div>
              </div>

              <div className="scan-module-card disabled">
                <div className="card-header-clickable" onClick={() => setIsSamplingOpen(!isSamplingOpen)}>
                  <div className="checkbox-label-wrapper">
                    <input type="checkbox" id="m-sampling" disabled />
                    <label htmlFor="m-sampling">Sampling Instability Analysis</label>
                  </div>
                  <div className="badge-arrow-wrapper">
                    <span className="status-badge soon">COMING SOON</span>
                    <span className={`arrow-icon ${isSamplingOpen ? "rotated" : ""}`}>▼</span>
                  </div>
                </div>
                <div className="collapsible-content" style={{ maxHeight: isSamplingOpen ? "300px" : "0px", opacity: isSamplingOpen ? 1 : 0 }}>
                  <p className="card-desc">
                    Runs test scenarios across a customized temperature × top_p inference grid with N=20 runs per point.
                    Calculates the Instability Score to detect alignment degradation.
                  </p>
                </div>
              </div>

              <div className="scan-module-card disabled">
                <div className="card-header-clickable" onClick={() => setIsGCDOpen(!isGCDOpen)}>
                  <div className="checkbox-label-wrapper">
                    <input type="checkbox" id="m-gcg" disabled />
                    <label htmlFor="m-gcg">Greedy Coordinate Gradient (GCG) Attacks</label>
                  </div>
                  <div className="badge-arrow-wrapper">
                    <span className="status-badge soon">COMING SOON</span>
                    <span className={`arrow-icon ${isGCDOpen ? "rotated" : ""}`}>▼</span>
                  </div>
                </div>
                <div className="collapsible-content" style={{ maxHeight: isGCDOpen ? "300px" : "0px", opacity: isGCDOpen ? 1 : 0 }}>
                  <p className="card-desc">
                    Executes a gradient-based token optimization via Greedy Coordinate Gradient (GCG) directly on open weights.
                  </p>
                </div>
              </div>

            </div>

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

          {status.visible && (
              <div className={`status ${status.isError ? "error" : ""}`}>
                {!status.isError && <span className="spinner"></span>}
                <span dangerouslySetInnerHTML={{ __html: status.text }} />
              </div>
          )}

          {result && (
              <section id="result">
                <div className={`verdict ${result.verdict.code === 'danger' ? 'do_not_deploy' : result.verdict.code}`}>
                  <div className="repo">
                    {result.repo}
                    {result.from_cache && " · cached"}
                  </div>
                  <span className="badge">{result.verdict.label}</span>
                  <p><span className="label">Diagnosis</span><br />{result.verdict.diagnosis}</p>
                  <p><span className="label">Recommendation</span><br />{result.verdict.recommendation}</p>
                </div>

                {result.metrics.map((m, index) => {
                  const isOpen = !!openMetrics[index];
                  const headlineColorClass = getColorClass(m.headline);

                  return (
                      <div className={`metric metric-dropdown ${isOpen ? "open" : ""}`} key={index}>
                        <div className="metric-header" onClick={() => toggleMetric(index)}>
                          <div className="metric-header-title">
                            <h3>{m.title}</h3>
                            <div className={`headline-badge ${headlineColorClass}`}>
                              {m.headline}
                            </div>
                          </div>
                          <span className={`metric-arrow ${isOpen ? "rotated" : ""}`}>▼</span>
                        </div>

                        <div
                            className="metric-collapsible"
                            style={{
                              maxHeight: isOpen ? "2000px" : "0px",
                              opacity: isOpen ? 1 : 0,
                              overflow: isOpen ? "visible" : "hidden"
                            }}
                        >
                          <div className="metric-body">
                            <p className="what">{m.what}</p>
                            <p className="read">{m.read}</p>

                            <div className="fields">
                              {Object.entries(m.fields).map(([key, value]) => {
                                const valStr = typeof value === 'object' ? JSON.stringify(value) : String(value);
                                const itemColorClass = getColorClass(valStr);

                                return (
                                    <span key={key}>
                                      {key}: <b className={itemColorClass}>{valStr}</b>
                                    </span>
                                );
                              })}
                            </div>
                          </div>
                        </div>
                      </div>
                  );
                })}

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