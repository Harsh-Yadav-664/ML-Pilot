export interface DataMetrics {
  total_rows: number;
  total_columns: number;
  missing_data_percent: number;
  duplicate_rows: number;
}

export type LeakageCategory =
  | 'target'
  | 'temporal'
  | 'missingness'
  | 'contamination'
  | 'preprocessing'
  | 'aggregate';

export interface LeakageWarning {
  id: string;
  column: string;
  message: string;
  severity: 'low' | 'medium' | 'high';
  category: LeakageCategory;
}

export interface FeatureSuggestion {
  name: string;
  formula: string;
  reason: string;
  risk: string;
  impact?: number;
}

export type ExperimentStatus = 'running' | 'completed' | 'failed' | 'queued';
export type Decision = 'keep' | 'reject' | 'baseline' | 'none';

export interface Experiment {
  id: string;
  parent_id: string | null;
  model_name: string;
  status: ExperimentStatus;
  metrics: {
    f1?: number;
    accuracy?: number;
    precision?: number;
    recall?: number;
  };
  runtime_seconds: number;
  created_at: string;
  title?: string;
  feature?: string;
  params?: Record<string, string | number>;
  progress?: number;
  error?: string;
  decision?: Decision;
  /** Set by the backend: the best measured model so far, and the lineage that leads to it. */
  champion?: boolean;
  on_champion_path?: boolean;
}

export interface ColumnProfile {
  name: string;
  dtype: 'int' | 'float' | 'category' | 'datetime' | 'bool' | 'string';
  role: 'feature' | 'target' | 'id';
  missing_pct: number;
  unique: number;
  dist: number[];
  flag?: 'leak' | 'warn';
}

export interface AgentReply {
  text: string;
  suggestion?: FeatureSuggestion;
}

export interface Endpoint {
  id: string;
  experiment_id: string;
  model_name: string;
  url: string;
  status: 'healthy' | 'warming' | 'degraded';
  rpm: number;
  p95_ms: number;
  created_at: string;
}

export interface FeatureWeight {
  name: string;
  weight: number;
}

export interface CleanStep {
  id: string;
  label: string;
  detail: string;
  done: boolean;
}

export type View = 'dashboard' | 'experiments' | 'deployments' | 'settings';
export type AgentStatus = 'idle' | 'thinking' | 'autopilot';
export type Provider = 'openai' | 'anthropic' | 'local';
