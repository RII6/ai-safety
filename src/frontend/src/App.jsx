import { useState } from "react";
import "./App.css";
import Header from "./components/Header";
import ConfigSection from "./components/ConfigSection";
import StatusDisplay from "./components/StatusDisplay";
import ResultSection from "./components/ResultSection";
import notificationSound from './public/notification.mp3';
import RecentScans from './components/RecentScans';

function App() {
  const [repo, setRepo] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState({ text: "", isError: false, visible: false });
  const [result, setResult] = useState(null);
  const [openMetrics, setOpenMetrics] = useState({});

  const [scanGeneral, setScanGeneral] = useState(true);
  const [scanInjection, setScanInjection] = useState(false);
  const [scanObfuscation, setScanObfuscation] = useState(false);
  const [scanSampling, setScanSampling] = useState(false);
  const [scanGcg, setScanGcg] = useState(false);

  let audioCtx = null;

  const ensureAudioContext = async () => {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      await audioCtx.resume();
    }
    return audioCtx;
  };

  const playNotificationSound = async () => {
    try {
      const ctx = audioCtx;
      const response = await fetch(notificationSound);
      if (!response.ok) throw new Error('Failed to load audio file');
      const arrayBuffer = await response.arrayBuffer();
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      const gainNode = ctx.createGain();
      gainNode.gain.value = 0.6;
      source.connect(gainNode);
      gainNode.connect(ctx.destination);
      source.start(0);
    } catch (error) {
      console.warn('MP3 playback failed, using synthetic sound:', error);
      try {
        const ctx = audioCtx;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = 880;
        osc.type = 'sine';
        gain.gain.setValueAtTime(0.4, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.3);
      } catch (e) {
        console.error('Synthetic sound failed:', e);
      }
    }
  };

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

    await ensureAudioContext();

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
      if (scanObfuscation) selectedModules.push("obfuscation");
      if (scanSampling) selectedModules.push("sampling");
      if (scanGcg) selectedModules.push("gcg");

      const res = await fetch("/api/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repo: trimmedRepo,
          force: true,
          modules: selectedModules,
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
        await playNotificationSound();
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
        <Header />
        <main>
          <ConfigSection
              repo={repo}
              setRepo={setRepo}
              loading={loading}
              onSubmit={handleScan}
              scanGeneral={scanGeneral}
              setScanGeneral={setScanGeneral}
              scanInjection={scanInjection}
              setScanInjection={setScanInjection}
              scanObfuscation={scanObfuscation}
              setScanObfuscation={setScanObfuscation}
              scanSampling={scanSampling}
              setScanSampling={setScanSampling}
              scanGcg={scanGcg}
              setScanGcg={setScanGcg}
          />

          <StatusDisplay
              text={status.text}
              isError={status.isError}
              visible={status.visible}
          />

          {result && (
              <ResultSection
                  result={result}
                  openMetrics={openMetrics}
                  toggleMetric={toggleMetric}
              />
          )}
          <RecentScans />
        </main>
      </>
  );
}

export default App;