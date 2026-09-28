import {
  DataMetrics,
  LeakageWarning,
  FeatureSuggestion,
  Experiment,
} from '../types';

export const mockMetrics: DataMetrics = {
  total_rows: 148920,
  total_columns: 42,
  missing_data_percent: 0.0317,
  duplicate_rows: 214,
};

export const mockWarnings: LeakageWarning[] = [
  {
    id: 'lk_01',
    column: 'churn_date',
    message:
      'Directly encodes the target. Present only for churned customers — leaks label into features.',
    severity: 'high',
  },
  {
    id: 'lk_02',
    column: 'account_closed_flag',
    message:
      'Populated post-outcome. High correlation (0.94) with target suggests future information.',
    severity: 'high',
  },
  {
    id: 'lk_03',
    column: 'last_ticket_resolution_ts',
    message:
      'May occur after the prediction cutoff for a subset of rows. Verify temporal ordering.',
    severity: 'medium',
  },
];

export const mockSuggestions: FeatureSuggestion[] = [
  {
    name: 'days_since_active',
    formula: 'current_date - last_login',
    reason: 'Recent inactivity strongly correlates with churn.',
    risk: 'Ensure last_login is available before prediction time.',
  },
  {
    name: 'support_ticket_velocity',
    formula: 'count(tickets, 30d) / count(tickets, 90d)',
    reason:
      'A rising short-term ticket rate signals frustration that precedes cancellation.',
    risk: 'Sparse for low-engagement accounts; expect NaNs to impute.',
  },
  {
    name: 'plan_downgrade_recency',
    formula: 'months_since(last_plan_change where delta < 0)',
    reason:
      'Downgrades are a leading indicator; recency captures momentum better than a flag.',
    risk: 'Only defined for accounts with at least one plan change.',
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
    created_at: '2026-02-11T09:14:00Z',
  },
  {
    id: 'exp_014',
    parent_id: 'exp_001',
    model_name: 'XGBClassifier',
    status: 'completed',
    metrics: { f1: 0.81, accuracy: 0.88, precision: 0.79, recall: 0.83 },
    runtime_seconds: 12.4,
    created_at: '2026-02-11T09:41:00Z',
  },
  {
    id: 'exp_021',
    parent_id: 'exp_014',
    model_name: 'XGBClassifier',
    status: 'completed',
    metrics: { f1: 0.84, accuracy: 0.91, precision: 0.82, recall: 0.87 },
    runtime_seconds: 14.1,
    created_at: '2026-02-11T10:02:00Z',
  },
  {
    id: 'exp_022',
    parent_id: 'exp_014',
    model_name: 'LightGBM',
    status: 'running',
    metrics: {},
    runtime_seconds: 6.7,
    created_at: '2026-02-11T10:05:00Z',
  },
  {
    id: 'exp_023',
    parent_id: 'exp_021',
    model_name: 'CatBoost',
    status: 'queued',
    metrics: {},
    runtime_seconds: 0,
    created_at: '2026-02-11T10:08:00Z',
  },
  {
    id: 'exp_018',
    parent_id: 'exp_001',
    model_name: 'RandomForest',
    status: 'failed',
    metrics: {},
    runtime_seconds: 2.1,
    created_at: '2026-02-11T09:52:00Z',
  },
];
