import React from 'react';
import { getColorClass, getHeadlineColor } from '../utils/colorUtils';

const MetricDropdown = ({ metric, index, isOpen, onToggle }) => {
    return (
        <div className={`metric metric-dropdown ${isOpen ? "open" : ""}`}>
            <div className="metric-header" onClick={() => onToggle(index)}>
                <div className="metric-header-title">
                    <h3>{metric.title}</h3>
                    <div className={`headline-badge ${getHeadlineColor(metric.headline)}`}>
                        {metric.headline}
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
                    <p className="what">{metric.what}</p>
                    <p className="read">{metric.read}</p>
                    <div className="fields">
                        {Object.entries(metric.fields).map(([key, value]) => {
                            const valStr = typeof value === 'object' ? JSON.stringify(value) : String(value);
                            const itemColorClass = getColorClass(key, value);
                            const displayValue = (value === null || value === undefined) ? 'N/A' : valStr;
                            return (
                                <span key={key}>
                  {key}: <b className={itemColorClass}>{displayValue}</b>
                </span>
                            );
                        })}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MetricDropdown;