import React from 'react';

const Header = () => {
    return (
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
    );
};

export default Header;