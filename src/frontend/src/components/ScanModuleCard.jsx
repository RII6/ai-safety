import React from 'react';

const ScanModuleCard = ({
                            id,
                            title,
                            description,
                            isActive,
                            isDisabled = false,
                            checked,
                            onChange,
                            isOpen,
                            onToggle,
                            statusBadge,
                        }) => {
    return (
        <div className={`scan-module-card ${isActive ? "active" : ""} ${isDisabled ? "disabled" : ""}`}>
            <div className="card-header-clickable" onClick={onToggle}>
                <div className="checkbox-label-wrapper" onClick={(e) => e.stopPropagation()}>
                    <input
                        type="checkbox"
                        id={id}
                        checked={checked}
                        onChange={(e) => onChange(e.target.checked)}
                        disabled={isDisabled}
                    />
                    <label htmlFor={id}>{title}</label>
                </div>
                <div className="badge-arrow-wrapper">
                    {statusBadge && (
                        <span className={`status-badge ${statusBadge.className}`}>{statusBadge.label}</span>
                    )}
                    <span className={`arrow-icon ${isOpen ? "rotated" : ""}`}>▼</span>
                </div>
            </div>

            <div
                className="collapsible-content"
                style={{ maxHeight: isOpen ? "300px" : "0px", opacity: isOpen ? 1 : 0 }}
            >
                <p className="card-desc">{description}</p>
            </div>
        </div>
    );
};

export default ScanModuleCard;