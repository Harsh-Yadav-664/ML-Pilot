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

export const getDataMetrics = () =>
  call(() => api.get<DataMetrics>('/data/metrics').then((r) => r.data), mockMetrics);

export const getLeakageWarnings = () =>
  call(() => api.get<LeakageWarning[]>('/data/leakage-warnings').then((r) => r.data), mockWarnings);

export const getColumns = () =>
  call(() => api.get<ColumnProfile[]>('/data/columns').then((r) => r.data), mockColumns);

export const getFeatureSuggestions = () =>
  call(() => api.get<FeatureSuggestion[]>('/agent/suggestions').then((r) => r.data), mockSuggestions);

export const runExperiment = (
  suggestion: FeatureSuggestion,
  model_name: string,
  parent_id: string | null
) =>
  call(
    () =>
      api
        .post<Experiment>('/experiments/run', { feature_suggestion: suggestion, model_name, parent_id })
        .then((r) => r.data),
    null as Experiment | null
  );

export const getExperimentTree = () =>
  call(() => api.get<Experiment[]>('/experiments/tree').then((r) => r.data), mockExperiments);

export function describeError(e: unknown): string {
  if (axios.isAxiosError(e)) {
    if (e.response) return `HTTP ${e.response.status}: ${JSON.stringify(e.response.data?.detail ?? e.response.statusText)}`;
    return e.message || 'network error';
  }
  return e instanceof Error ? e.message : String(e);
}

export default api;
