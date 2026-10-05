import axios from 'axios';
import {
  ColumnProfile,
  DataMetrics,
  Experiment,
  FeatureSuggestion,
  LeakageWarning,
} from '../types';
import {
  mockColumns,
  mockExperiments,
  mockMetrics,
  mockSuggestions,
  mockWarnings,
} from './mock';

export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1/ui';

const DEMO_KEY = 'mlpilot.demo';

function readDemo(): boolean {
  try {
    return localStorage.getItem(DEMO_KEY) === '1';
  } catch {
    return false;
  }
}

/** Demo mode is the only path that may show sample data. The user turns it on explicitly. */
export function setDemoMode(on: boolean) {
  try {
    if (on) localStorage.setItem(DEMO_KEY, '1');
    else localStorage.removeItem(DEMO_KEY);
  } catch {
    /* storage unavailable: demo mode can't be persisted */
  }
  window.location.reload();
}

export const connection = { live: false, demo: readDemo() };

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: { 'Content-Type': 'application/json' },
});

const delay = <T,>(value: T, ms = 280) => new Promise<T>((resolve) => setTimeout(() => resolve(value), ms));

/** Real backend call, or sample data when Demo mode is on. Errors propagate to the caller. */
async function call<T>(request: () => Promise<T>, demoData: T): Promise<T> {
  if (connection.demo) return delay(demoData);
  const res = await request();
  connection.live = true;
  return res;
}

/** The dataset the user is working on (set by sample data, upload or SQL). */
export interface ActiveDataset {
  dataset_path: string;
  filename: string;
  target_column: string;
}

export interface DatasetInfo {
  dataset_path: string;
  filename: string;
  columns: string[];
  total_rows: number;
  default_target?: string;
}

const ds = (d: ActiveDataset) => ({ dataset_path: d.dataset_path, target_column: d.target_column });

export const loadSampleDataset = () =>
  api.post<DatasetInfo>('/data/sample', { dataset_name: 'telecom_churn' }).then((r) => r.data);

export const uploadDataset = (file: File) => {
  const form = new FormData();
  form.append('file', file);
  return api
    .post<DatasetInfo>('/data/upload', form, { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 60000 })
    .then((r) => r.data);
};

export const getDataMetrics = (d: ActiveDataset) =>
  call(() => api.get<DataMetrics>('/data/metrics', { params: ds(d) }).then((r) => r.data), mockMetrics);

export const getLeakageWarnings = (d: ActiveDataset) =>
  call(() => api.get<LeakageWarning[]>('/data/leakage-warnings', { params: ds(d) }).then((r) => r.data), mockWarnings);

export const getColumns = (d: ActiveDataset) =>
  call(() => api.get<ColumnProfile[]>('/data/columns', { params: ds(d) }).then((r) => r.data), mockColumns);

export const getFeatureSuggestions = (d: ActiveDataset) =>
  call(
    () => api.get<FeatureSuggestion[]>('/agent/suggestions', { params: ds(d), timeout: 120000 }).then((r) => r.data),
    mockSuggestions
  );

type BackendStatus = Experiment['status'] | 'created';
type BackendExperiment = Omit<Experiment, 'status'> & { status: BackendStatus };

export const getExperimentTree = (d: ActiveDataset | null) =>
  call(
    () =>
      api
        .get<BackendExperiment[]>('/experiments/tree', { params: d ? { dataset_path: d.dataset_path } : {} })
        .then((r) => r.data.map((e): Experiment => ({ ...e, status: e.status === 'created' ? 'queued' : e.status }))),
    mockExperiments
  );

export const runExperiment = (d: ActiveDataset, suggestion: FeatureSuggestion, model_name: string, parent_id: string | null) =>
  api
    .post<Experiment>('/experiments/run', { ...ds(d), feature_suggestion: suggestion, model_name, parent_id })
    .then((r) => r.data);

export const runBaseline = (d: ActiveDataset) =>
  api.post<Experiment>('/experiments/baseline', ds(d)).then((r) => r.data);

export const runAutoClean = (d: ActiveDataset) =>
  api.post<Experiment>('/agent/auto-clean', ds(d), { timeout: 120000 }).then((r) => r.data);

export interface AutoOptimizeJob {
  status: 'running' | 'completed' | 'failed';
  error?: string;
  result?: { summary: string; best_f1: number; winner_features: string[] };
}

export const startAutoOptimize = (d: ActiveDataset, n_hypotheses: number) =>
  api.post<{ job_id: string }>('/agent/auto-optimize', { ...ds(d), n_hypotheses }).then((r) => r.data.job_id);

export const getAutoOptimize = (jobId: string) =>
  api.get<AutoOptimizeJob>(`/agent/auto-optimize/${jobId}`).then((r) => r.data);

/** Grounded Q&A over the recorded experiment history (POST /chat/ask). */
export const chatAsk = (query: string) =>
  axios
    .post<{ answer: string }>(`${API_BASE_URL.replace(/\/ui\/?$/, '')}/chat/ask`, { query }, { timeout: 120000 })
    .then((r) => r.data.answer);

/** True when the backend answered with an HTTP error (it is reachable). */
export const isHttpError = (e: unknown) => axios.isAxiosError(e) && !!e.response;

export function describeError(e: unknown): string {
  if (axios.isAxiosError(e)) {
    if (e.response) return `HTTP ${e.response.status}: ${JSON.stringify(e.response.data?.detail ?? e.response.statusText)}`;
    return e.message || 'network error';
  }
  return e instanceof Error ? e.message : String(e);
}

export default api;
