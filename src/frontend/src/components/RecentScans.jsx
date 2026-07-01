import { useState, useEffect } from 'react';
import ScanModal from './ScanModal';

export default function RecentScans({ refreshKey }) {
    const [scans, setScans] = useState([]);
    const [loading, setLoading] = useState(false);
    const [selectedScanId, setSelectedScanId] = useState(null);

    const loadScans = async () => {
        try {
            setLoading(true);
            const res = await fetch('/api/reports');
            if (res.ok) {
                const data = await res.json();
                setScans(data);
            }
        } catch (e) {
            console.error('Failed to load scans:', e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadScans();
    }, [refreshKey]);

    const handleItemClick = (scanId) => {
        setSelectedScanId(scanId);
    };

    return (
        <>
            <div className="recent-scans">
                <div className="recent-scans-header">
                    <h3>Recent Scans</h3>
                    <button className="refresh-btn" onClick={loadScans} disabled={loading}>
                        {loading ? '...' : '⟳'}
                    </button>
                </div>
                {scans.length === 0 ? (
                    <p className="no-scans">No scans yet. Run your first scan!</p>
                ) : (
                    <ul className="scan-list">
                        {scans.map((scan) => (
                            <li
                                key={scan.id}
                                className="scan-item"
                                onClick={() => handleItemClick(scan.id)}
                                style={{ cursor: 'pointer' }}
                            >
                                <span className="scan-repo">{scan.repo}</span>
                                <span className={`scan-verdict ${scan.verdict}`}>
                  {scan.verdict}
                </span>
                            </li>
                        ))}
                    </ul>
                )}
            </div>

            {selectedScanId && (
                <ScanModal
                    scanId={selectedScanId}
                    onClose={() => setSelectedScanId(null)}
                />
            )}
        </>
    );
}