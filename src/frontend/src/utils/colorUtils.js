import { colorRules } from '../constants/colorRules';

export const getColorClass = (key, value) => {
    if (value === null || value === undefined) return '';
    const rule = colorRules[key];
    if (!rule) return '';

    if (rule.type === 'severity') {
        return rule.mapping[String(value).toLowerCase()] || '';
    }

    const num = parseFloat(value);
    if (isNaN(num)) return '';

    const [good, bad] = rule.thresholds;
    if (key === 'fail_rate' || key === 'asr_conditional' || key === 'worst_case_bypass_rate'
        || key === 'worst_sampling_fail_rate') {
        if (num <= good) return rule.colors[0];
        if (num <= bad)  return rule.colors[1];
        return rule.colors[2];
    }
    if (rule.type === 'auroc' || rule.type === 'cohens_d') {
        if (num >= good) return rule.colors[0];
        if (num >= bad)  return rule.colors[1];
        return rule.colors[2];
    }
    if (rule.type === 'margin') {
        if (num >= good) return rule.colors[0];
        if (num >= bad)  return rule.colors[1];
        return rule.colors[2];
    }
    return '';
};

export const getHeadlineColor = (text) => {
    const lower = text.toLowerCase();

    if (lower.includes('n/a') || lower.includes('null') || lower.includes('no refused')) {
        return 'status-neutral';
    }
    const match = text.match(/([\d.]+)%?/);
    if (!match) return '';

    const num = parseFloat(match[1]);
    if (isNaN(num)) return '';

    if (text.includes('%')) {
        if (num <= 10) return 'status-success';
        if (num <= 30) return 'status-warning';
        return 'status-danger';
    }

    if (lower.includes('auroc') || lower.includes('cohens')) {
        if (num >= 0.8) return 'status-success';
        if (num >= 0.6) return 'status-warning';
        return 'status-danger';
    }

    if (lower.includes('high') || lower.includes('fail')) return 'status-danger';
    if (lower.includes('medium')) return 'status-warning';
    if (lower.includes('low') || lower.includes('passed')) return 'status-success';

    return '';
};