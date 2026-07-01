import React, { useRef, useEffect } from 'react';

const CustomSelect = ({
                          value,
                          onChange,
                          onFocus,
                          onKeyDown,
                          isOpen,
                          onToggle,
                          filteredOptions,
                          highlightedIndex,
                          onHighlight,
                          onSelect,
                          placeholder,
                      }) => {
    const dropdownRef = useRef(null);

    useEffect(() => {
        const handleClickOutside = (e) => {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
                onToggle(false);
                onHighlight(-1);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [onToggle, onHighlight]);

    return (
        <div className="custom-select-wrapper" ref={dropdownRef}>
            <input
                id="repo"
                type="text"
                placeholder={placeholder}
                autoComplete="off"
                value={value}
                onChange={onChange}
                onFocus={onFocus}
                onKeyDown={onKeyDown}
            />
            <span className="select-arrow">▼</span>
            {isOpen && (
                <ul className="dropdown-list">
                    {filteredOptions.length > 0 ? (
                        filteredOptions.map((model, index) => (
                            <li
                                key={model}
                                className={index === highlightedIndex ? 'dropdown-item-highlighted' : 'dropdown-item'}
                                onMouseEnter={() => onHighlight(index)}
                                onMouseLeave={() => onHighlight(-1)}
                                onMouseDown={(e) => {
                                    e.preventDefault();
                                    onSelect(model);
                                }}
                            >
                                {model}
                            </li>
                        ))
                    ) : (
                        <li className="dropdown-empty">No models found</li>
                    )}
                </ul>
            )}
        </div>
    );
};

export default CustomSelect;