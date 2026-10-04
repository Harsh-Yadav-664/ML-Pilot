import {
  AgentReply,
  ColumnProfile,
  DataMetrics,
  Endpoint,
  Experiment,
  FeatureSuggestion,
  LeakageWarning,
} from '../types';
import { defaultParams } from '../lib';

const ago = (min: number) => new Date(Date.now() - min * 60000).toISOString();

export const mockMetrics: DataMetrics = {
  total_rows: 148920,
  total_columns: 17,
  missing_data_percent: 0.0317,
  duplicate_rows: 214,
};

export const mockWarnings: LeakageWarning[] = [
  {
    id: 'lk_01',
    column: 'churn_date',
    message: "Only populated for churned customers. 87.4% null, and 100% of the non-null rows are positive — the column encodes the label.",
    severity: 'high',
    category: 'target',
  },
  {
    id: 'lk_02',
    column: 'account_closed_flag',
    message: 'Written by the billing system after the outcome. 0.94 correlation with the target implies future information.',
    severity: 'high',
    category: 'target',
  },
  {
    id: 'lk_03',
    column: 'last_ticket_resolution_ts',
    message: 'For ~6% of rows the resolution timestamp lands after the prediction cutoff. As-of ordering breaks on those rows.',
    severity: 'medium',
    category: 'temporal',
  },
  {
    id: 'lk_04',
    column: 'nps_score',
    message: '18.4% missing, and missingness concentrates in churned rows (31% vs 12%). The absence pattern itself carries the label.',
    severity: 'medium',
    category: 'missingness',
  },
];

const c = (
  name: string,
  dtype: ColumnProfile['dtype'],
  role: ColumnProfile['role'],
  missing_pct: number,
  unique: number,
  dist: number[],
  flag?: ColumnProfile['flag']
): ColumnProfile => ({ name, dtype, role, missing_pct, unique, dist, flag });

export const mockColumns: ColumnProfile[] = [
  c('customer_id', 'string', 'id', 0, 148706, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
  c('is_churned', 'bool', 'target', 0, 2, [0.92, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.32]),
  c('signup_date', 'datetime', 'feature', 0, 1824, [0.3, 0.42, 0.55, 0.62, 0.7, 0.66, 0.8, 0.9, 1, 0.86, 0.7, 0.5]),
  c('last_login', 'datetime', 'feature', 1.2, 412, [0.1, 0.12, 0.14, 0.2, 0.28, 0.4, 0.5, 0.62, 0.8, 0.95, 1, 0.7]),
  c('tenure_months', 'int', 'feature', 0, 72, [1, 0.86, 0.7, 0.58, 0.5, 0.42, 0.34, 0.28, 0.2, 0.16, 0.1, 0.08]),
  c('plan_tier', 'category', 'feature', 0, 4, [0.35, 0, 0, 1, 0, 0, 0.6, 0, 0, 0.22, 0, 0]),
  c('monthly_spend', 'float', 'feature', 0.4, 5120, [0.2, 0.5, 0.9, 1, 0.8, 0.55, 0.36, 0.22, 0.14, 0.08, 0.04, 0.02]),
  c('support_tickets_90d', 'int', 'feature', 0, 19, [1, 0.6, 0.34, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01, 0]),
  c('avg_session_min', 'float', 'feature', 2.6, 3810, [0.1, 0.4, 0.86, 1, 0.72, 0.46, 0.28, 0.16, 0.1, 0.06, 0.03, 0.01]),
  c('payment_failures', 'int', 'feature', 0, 9, [1, 0.14, 0.06, 0.03, 0.02, 0.01, 0, 0, 0, 0, 0, 0]),
  c('region', 'category', 'feature', 0.2, 12, [0.8, 1, 0.6, 0.5, 0.45, 0.4, 0.3, 0.26, 0.2, 0.14, 0.1, 0.06]),
  c('device_type', 'category', 'feature', 0.9, 5, [1, 0, 0, 0.55, 0, 0, 0.3, 0, 0, 0.12, 0, 0.05]),
  c('nps_score', 'int', 'feature', 18.4, 11, [0.1, 0.06, 0.08, 0.12, 0.18, 0.3, 0.42, 0.66, 0.9, 1, 0.8, 0.4]),
  c('referral_source', 'category', 'feature', 9.1, 8, [1, 0.5, 0, 0.36, 0, 0.24, 0, 0.16, 0, 0.1, 0, 0.06]),
  c('churn_date', 'datetime', 'feature', 87.4, 18760, [0.2, 0.3, 0.42, 0.5, 0.6, 0.7, 0.8, 0.86, 0.9, 0.96, 1, 0.9], 'leak'),
  c('account_closed_flag', 'bool', 'feature', 0, 2, [0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3], 'leak'),
  c('last_ticket_resolution_ts', 'datetime', 'feature', 31.5, 40211, [0.2, 0.26, 0.4, 0.5, 0.58, 0.6, 0.72, 0.8, 0.86, 1, 0.9, 0.6], 'warn'),
];

export const mockSuggestions: FeatureSuggestion[] = [
  {
    name: 'days_since_active',
    formula: 'current_date - last_login',
    reason: 'Recent inactivity strongly correlates with churn.',
    risk: 'Ensure last_login is available before prediction time.',
    impact: 0.031,
  },
  {
    name: 'support_ticket_velocity',
    formula: 'count(tickets, 30d) / count(tickets, 90d)',
    reason: 'A rising short-term ticket rate signals frustration that precedes cancellation.',
    risk: 'Sparse for low-engagement accounts; expect NaNs to impute.',
    impact: 0.024,
  },
  {
    name: 'plan_downgrade_recency',
    formula: 'months_since(last_plan_change where delta < 0)',
    reason: 'Downgrades are a leading indicator; recency captures momentum better than a flag.',
    risk: 'Only defined for accounts with at least one plan change.',
    impact: 0.018,
  },
  {
    name: 'payment_failure_streak',
    formula: 'max_consecutive(payment_failures > 0)',
    reason: 'Involuntary churn clusters around consecutive failed charges.',
    risk: 'Overlaps with payment_failures; watch collinearity in linear models.',
    impact: 0.021,
  },
];

export const mockExperiments: Experiment[] = [
  {
    id: 'exp_001',
    parent_id: null,
    model_name: 'LogisticRegression',
    status: 'completed',
    metrics: { f1: 0.72, accuracy: 0.81, precision: 0.7, recall: 0.74 },
    runtime_seconds: 3.2,
    created_at: ago(96),
    title: 'Baseline',
    params: defaultParams('LogisticRegression'),
    decision: 'baseline',
  },
  {
    id: 'exp_014',
    parent_id: 'exp_001',
    model_name: 'XGBClassifier',
    status: 'completed',
    metrics: { f1: 0.81, accuracy: 0.88, precision: 0.79, recall: 0.83 },
    runtime_seconds: 12.4,
    created_at: ago(71),
    title: '+ days_since_active',
    feature: 'days_since_active',
    params: defaultParams('XGBClassifier'),
    decision: 'keep',
  },
  {
    id: 'exp_021',
    parent_id: 'exp_014',
    model_name: 'XGBClassifier',
    status: 'completed',
    metrics: { f1: 0.84, accuracy: 0.91, precision: 0.82, recall: 0.87 },
    runtime_seconds: 14.1,
    created_at: ago(44),
    title: '+ support_ticket_velocity',
    feature: 'support_ticket_velocity',
    params: { ...defaultParams('XGBClassifier'), max_depth: 7 },
    decision: 'keep',
  },
  {
    id: 'exp_018',
    parent_id: 'exp_001',
    model_name: 'RandomForestClassifier',
    status: 'failed',
    metrics: {},
    runtime_seconds: 2.1,
    created_at: ago(58),
    title: 'Unbounded depth',
    params: { ...defaultParams('RandomForestClassifier'), max_depth: 'None' },
    error: 'MemoryError: unable to allocate 4.2 GiB for an array with shape (148920, 3800)',
    decision: 'reject',
  },
  {
    id: 'exp_022',
    parent_id: 'exp_014',
    model_name: 'LGBMClassifier',
    status: 'running',
    metrics: {},
    runtime_seconds: 6.7,
    created_at: ago(2),
    title: 'Swap to LightGBM',
    params: defaultParams('LGBMClassifier'),
    progress: 58,
    decision: 'none',
  },
  {
    id: 'exp_023',
    parent_id: 'exp_021',
    model_name: 'CatBoostClassifier',
    status: 'queued',
    metrics: {},
    runtime_seconds: 0,
    created_at: ago(1),
    title: 'Native categoricals',
    params: defaultParams('CatBoostClassifier'),
    progress: 0,
    decision: 'none',
  },
];

export const mockEndpoints: Endpoint[] = [
  {
    id: 'ep_live',
    experiment_id: 'exp_021',
    model_name: 'XGBClassifier',
    url: 'https://api.mlpilot.dev/v1/predict/churn-v3',
    status: 'healthy',
    rpm: 1840,
    p95_ms: 42,
    created_at: ago(180),
  },
];

export const mockDrift = Array.from({ length: 30 }, (_, i) => {
  const t = Date.now() - (29 - i) * 86400000;
  const wave = 0.08 + Math.sin(i / 4) * 0.03 + (i > 24 ? (i - 24) * 0.035 : 0);
  return { t, psi: Math.round(wave * 1000) / 1000 };
});

export const mockVolume = Array.from({ length: 24 }, (_, h) => ({
  h,
  n: Math.round(800 + Math.sin((h - 8) / 3.2) * 520 + (h % 3) * 40),
}));

const IDEAS: { keys: RegExp; intro: string; s: FeatureSuggestion }[] = [
  {
    keys: /session|usage|engage|trend/i,
    intro: 'Engagement momentum is usually more predictive than level. This one measures whether sessions are shrinking.',
    s: {
      name: 'session_gap_trend',
      formula: 'slope(avg_session_min, window=8w)',
      reason: 'A downward usage slope precedes cancellation by 3–5 weeks in comparable datasets.',
      risk: 'Needs at least 4 weekly observations; new accounts will be null.',
      impact: 0.019,
    },
  },
  {
    keys: /ratio|rate|spend|price|value/i,
    intro: 'Normalizing spend by tenure separates loyal high-value users from new, expensive trials.',
    s: {
      name: 'spend_to_tenure_ratio',
      formula: 'monthly_spend / (tenure_months + 1)',
      reason: 'Captures perceived value; heavy early spend with little history is a churn-prone cohort.',
      risk: 'Skewed distribution — apply log1p before linear models.',
      impact: 0.016,
    },
  },
  {
    keys: /time|recen|date|last|window|decay|temporal/i,
    intro: 'Steering toward temporal structure. Exponential decay weights recent behavior far more than old activity.',
    s: {
      name: 'recency_weighted_activity',
      formula: 'sum(logins_30d * exp(-age_days / 14))',
      reason: 'Smoother than a hard 30-day cutoff and less sensitive to one-off spikes.',
      risk: 'Requires a per-login event table; confirm it is snapshotted at cutoff.',
      impact: 0.027,
    },
  },
  {
    keys: /tenure|age|cohort|bucket|bin/i,
    intro: 'Tenure has a non-linear hazard curve. Bucketing lets simpler models capture it.',
    s: {
      name: 'tenure_bucket',
      formula: 'qcut(tenure_months, q=6)',
      reason: 'Churn risk peaks around months 2–3 and again at renewal boundaries.',
      risk: 'Bucket edges must be fitted on train only to avoid leakage.',
      impact: 0.014,
    },
  },
];

export function craftReply(prompt: string, used: string[]): AgentReply {
  const free = IDEAS.filter((i) => !used.includes(i.s.name));
  const hit = free.find((i) => i.keys.test(prompt)) ?? free[0];
  if (!hit) {
    return {
      text: "I've exhausted the ideas I can justify from the current schema. Try excluding leaky columns or joining another source.",
    };
  }
  return { text: hit.intro, suggestion: hit.s };
}
