import axios from 'axios';
import {
  AgentReply,
  ColumnProfile,
  DataMetrics,
  Experiment,
  FeatureSuggestion,
  LeakageWarning,
} from '../types';
import {
  craftReply,
  mockColumns,
  mockExperiments,
  mockMetrics,
  mockSuggestions,
  mockWarnings,
} from './mock';

export const API_BASE_URL = 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 2500,
  headers: { 'Content-Type': 'application/json' },
});

export const connection = { live: false };

async function withFallback<T>(request: () => Promise<T>, fallback: T, delay = 280): Promise<T> {
  try {
    const res = await request();
    connection.live = true;
    return res;
  } catch {
    return new Promise((resolve) => setTimeout(() => resolve(fallback), delay));
  }
}

export const getDataMetrics = () =>
  withFallback(() => api.get<DataMetrics>('/data/metrics').then((r) => r.data), mockMetrics);

export const getLeakageWarnings = () =>
  withFallback(
    () => api.get<LeakageWarning[]>('/data/leakage-warnings').then((r) => r.data),
    mockWarnings
  );

export const getColumns = () =>
  withFallback(() => api.get<ColumnProfile[]>('/data/columns').then((r) => r.data), mockColumns);

export const getFeatureSuggestions = () =>
  withFallback(
    () => api.get<FeatureSuggestion[]>('/agent/suggestions').then((r) => r.data),
    mockSuggestions
  );

export const askAgent = (prompt: string, used: string[]) =>
  withFallback(
    () => api.post<AgentReply>('/agent/ask', { prompt }).then((r) => r.data),
    craftReply(prompt, used),
    1100
  );

export const runExperiment = (
  suggestion: FeatureSuggestion,
  model_name: string,
  parent_id: string | null
) =>
  withFallback(
    () =>
      api
        .post<Experiment>('/experiments/run', { feature_suggestion: suggestion, model_name, parent_id })
        .then((r) => r.data),
    null as Experiment | null
  );

export const getExperimentTree = () =>
  withFallback(
    () => api.get<Experiment[]>('/experiments/tree').then((r) => r.data),
    mockExperiments
  );

export default api;
