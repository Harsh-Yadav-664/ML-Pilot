export interface DataMetrics {
  total_rows: number;
  total_columns: number;
  missing_data_percent: number;
  duplicate_rows: number;
}

export interface LeakageWarning {
  id: string;
  column: string;
  message: string;
  severity: 'low' | 'medium' | 'high';
}

export interface FeatureSuggestion {
  name: string;
  formula: string;
  reason: string;
  risk: string;
}

export interface Experiment {
  id: string;
  parent_id: string | null;
  model_name: string;
  status: 'running' | 'completed' | 'failed' | 'queued';
  metrics: {
    f1?: number;
    accuracy?: number;
    precision?: number;
    recall?: number;
  };
  runtime_seconds: number;
  created_at: string;
}

export interface ExperimentTreeNode extends Experiment {
  children: ExperimentTreeNode[];
}
