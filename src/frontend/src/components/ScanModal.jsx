import { useState, useEffect } from 'react';
import ResultSection from './ResultSection';

export default function ScanModal({ scanId, onClose }) {
    const [report, setReport] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [openMetrics, setOpenMetrics] = useState({});

    const toggleMetric = (index) => {
        setOpenMetrics((prev) => ({
            ...prev,
            [index]: !prev[index],
        }));
    };

    useEffect(() => {
        if (!scanId) return;
        const fetchReport = async () => {
            setLoading(true);
            try {
                const res = await fetch(`/api/reports/${scanId}`);
                if (!res.ok) throw new Error('Failed to load report');
                const data = await res.json();
                setReport(data);
                setOpenMetrics({});
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };
        fetchReport();
    }, [scanId]);

    if (!scanId) return null;

    const formatDate = (isoString) => {
        if (!isoString) return '';
        const date = new Date(isoString);
        return date.toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <div className="modal-header-left">
                        <h2 className="modal-title">
                            {report?.repo || 'Loading...'}
                        </h2>
                        {report?.meta?.created_at && (
                            <span className="modal-date">
                {formatDate(report.meta.created_at)}
              </span>
                        )}
                    </div>
                    <button className="modal-close" onClick={onClose}>
                        ×
                    </button>
                </div>

                {loading && <div className="modal-loading">Loading...</div>}
                {error && <div className="modal-error">Error: {error}</div>}
                {report && (
                    <div className="modal-report">
                        <ResultSection
                            result={report}
                            openMetrics={openMetrics}
                            toggleMetric={toggleMetric}
                        />
                    </div>
                )}
            </div>
        </div>
    );
}