// TypeScript interfaces matching backend Pydantic schemas

export type TaskType =
  | 'binary_classification'
  | 'multiclass_classification'
  | 'regression';

export type ProjectStatus = 'active' | 'archived' | 'completed';

export interface Project {
  id: string;
  name: string;
  description?: string;
  task_type: TaskType;
  status: ProjectStatus;
  owner_id: string;
  created_at: string;
  updated_at: string;
}

export interface ProjectCreate {
  name: string;
  description?: string;
  task_type: TaskType;
}

export interface Dataset {
  id: string;
  project_id: string;
  name: string;
  version: string;
  format: string;
  target_column?: string;
  num_rows?: number;
  num_columns?: number;
  profile?: Record<string, unknown>;
  status: string;
  file_path?: string;
  created_at: string;
  updated_at: string;
}

export type ExperimentStatus =
  | 'created'
  | 'validated'
  | 'queued'
  | 'running'
  | 'completed'
  | 'evaluated'
  | 'failed';

export type ExperimentDecision = 'keep' | 'reject' | 'inconclusive' | 'pending';

export interface Experiment {
  id: string;
  parent_id?: string;
  project_id: string;
  dataset_version: string;
  hypothesis: string;
  change_description: string;
  model_name: string;
  parameters: Record<string, unknown>;
  validation_config: Record<string, unknown>;
  feature_set: string[];
  preprocessing_config: Record<string, unknown>;
  budget: Record<string, number>;
  metrics?: Record<string, number>;
  artifacts: string[];
  runtime_seconds?: number;
  cost_usd?: number;
  status: ExperimentStatus;
  decision: ExperimentDecision;
  decision_reason?: string;
  agent_model?: string;
  created_at: string;
  updated_at: string;
}

export interface Hypothesis {
  id: string;
  experiment_id: string;
  project_id: string;
  text: string;
  rationale?: string;
  source: string;
  status: string;
  agent_model?: string;
  tags: string[];
  created_at: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}
