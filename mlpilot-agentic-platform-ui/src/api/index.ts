import axios from 'axios';
import {
  DataMetrics,
  LeakageWarning,
  FeatureSuggestion,
  Experiment,
} from '../types';
import {
  mockMetrics,
  mockWarnings,
  mockSuggestions,
  mockExperiments,
} from './mock';

const API_BASE_URL = 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000, // Increased timeout for AI calls
  headers: { 'Content-Type': 'application/json' },
});

/**
 * Wraps a live request so the UI stays populated with representative data
 * when the FastAPI backend isn't reachable during local development.
 */
async function withFallback<T>(request: () => Promise<T>, fallback: T): Promise<T> {
  try {
    return await request();
  } catch (error) {
    console.warn("Backend request failed, using fallback data:", error);
    return new Promise((resolve) => setTimeout(() => resolve(fallback), 300));
  }
}

// Data Readiness
export const getDataMetrics = (datasetPath: string, targetColumn: string) =>
  withFallback(() => api.get<DataMetrics>(`/ui/data/metrics?dataset_path=${encodeURIComponent(datasetPath)}&target_column=${encodeURIComponent(targetColumn)}`).then((r) => r.data), mockMetrics);

export const getLeakageWarnings = (datasetPath: string, targetColumn: string) =>
  withFallback(
    () => api.get<LeakageWarning[]>(`/ui/data/leakage-warnings?dataset_path=${encodeURIComponent(datasetPath)}&target_column=${encodeURIComponent(targetColumn)}`).then((r) => r.data),
    mockWarnings
  );

// AI Agent
export const getFeatureSuggestions = (datasetPath: string, targetColumn: string) =>
  withFallback(
    () => api.get<FeatureSuggestion[]>(`/ui/agent/suggestions?dataset_path=${encodeURIComponent(datasetPath)}&target_column=${encodeURIComponent(targetColumn)}`).then((r) => r.data),
    mockSuggestions
  );

export const runExperiment = (suggestion: FeatureSuggestion, datasetPath: string, targetColumn: string) =>
  withFallback(
    () =>
      api
        .post<Experiment>('/ui/experiments/run', { 
          feature_suggestion: suggestion, 
          dataset_path: datasetPath,
          target_column: targetColumn
        })
        .then((r) => r.data),
    {
      id: `exp_${Math.floor(Math.random() * 900 + 100)}`,
      parent_id: 'exp_021',
      model_name: 'XGBClassifier',
      status: 'queued' as const,
      metrics: {},
      runtime_seconds: 0,
      created_at: new Date().toISOString(),
    }
  );

export const runBaselineExperiment = (datasetPath: string, targetColumn: string) =>
  withFallback(
    () =>
      api
        .post<Experiment>('/ui/experiments/baseline', { 
          dataset_path: datasetPath,
          target_column: targetColumn
        })
        .then((r) => r.data),
    {
      id: `baseline_${Math.floor(Math.random() * 900 + 100)}`,
      parent_id: null as any,
      model_name: 'RandomForestClassifier',
      status: 'queued' as const,
      metrics: {},
      runtime_seconds: 0,
      created_at: new Date().toISOString(),
    }
  );

export const runAutoClean = (datasetPath: string, targetColumn: string) =>
  api.post<Experiment>('/ui/agent/auto-clean', { 
    dataset_path: datasetPath,
    target_column: targetColumn
  }).then(r => r.data);

// Experiments
export const getExperiments = () =>
  withFallback(() => api.get<Experiment[]>('/ui/experiments/tree').then((r) => r.data), mockExperiments);

export const getExperimentTree = () =>
  withFallback(
    () => api.get<Experiment[]>('/ui/experiments/tree').then((r) => r.data),
    mockExperiments
  );

export const getExperimentById = (id: string) =>
  withFallback(
    () => api.get<Experiment>(`/ui/experiments/${id}`).then((r) => r.data),
    mockExperiments.find((e) => e.id === id) ?? mockExperiments[0]
  );

export const runAutoOptimize = (datasetPath: string, targetColumn: string, nHypotheses: number = 5) =>
  api.post('/ui/agent/auto-optimize', { dataset_path: datasetPath, target_column: targetColumn, n_hypotheses: nHypotheses }).then(r => r.data);

export const getOptimizeStatus = (jobId: string) =>
  api.get(`/ui/agent/auto-optimize/${jobId}`).then(r => r.data);

export const exportExperimentScript = async (id: string) => {
  const res = await api.get(`/ui/experiments/${id}/export`);
  return res.data as { script: string; filename: string };
};

export default api;
