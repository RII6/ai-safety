import React from 'react';

const StatusDisplay = ({ text, isError, visible }) => {
    if (!visible) return null;
    return (
        <div className={`status ${isError ? "error" : ""}`}>
            {!isError && <span className="spinner"></span>}
            <span dangerouslySetInnerHTML={{ __html: text }} />
        </div>
    );
};

export default StatusDisplay;