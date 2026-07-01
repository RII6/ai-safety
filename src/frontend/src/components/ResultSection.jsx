import React from 'react';
import MetricDropdown from './MetricDropdown';

const ResultSection = ({ result, openMetrics, toggleMetric }) => {
    const verdictClass = result.verdict.code === 'danger' ? 'do_not_deploy' : result.verdict.code;

    return (
        <section id="result">
            <div className={`verdict ${verdictClass}`}>
                <div className="repo">
                    {result.repo}
                    {result.from_cache && " · cached"}
                </div>
                <span className="badge">{result.verdict.label}</span>
                <p><span className="label">Diagnosis</span><br />{result.verdict.diagnosis}</p>
                <p><span className="label">Recommendation</span><br />{result.verdict.recommendation}</p>
            </div>

            {result.metrics.map((metric, index) => (
                <MetricDropdown
                    key={index}
                    metric={metric}
                    index={index}
                    isOpen={!!openMetrics[index]}
                    onToggle={toggleMetric}
                />
            ))}

            <div className="meta">
                {result.meta.params ? `${(result.meta.params / 1e6).toFixed(0)}M params · ` : ""}
                {result.meta.sample}/{result.meta.sample} prompts · {result.meta.device}/{result.meta.dtype} · {result.meta.elapsed_s}s
            </div>
        </section>
    );
};

export default ResultSection;