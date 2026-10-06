import axios from 'axios';
import type { components } from './schema';
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

export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1';

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

/** The local access token; `python start.py` writes it to frontend/.env.local. */
const API_TOKEN: string | undefined = import.meta.env.VITE_MLPILOT_TOKEN;

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
    ...(API_TOKEN ? { Authorization: `Bearer ${API_TOKEN}` } : {}),
  },
});

const delay = <T,>(value: T, ms = 280) => new Promise<T>((resolve) => setTimeout(() => resolve(value), ms));

/** Real backend call, or sample data when Demo mode is on. Errors propagate to the caller. */
async function call<T>(request: () => Promise<T>, demoData: T): Promise<T> {
  if (connection.demo) return delay(demoData);
  const res = await request();
  connection.live = true;
  return res;
}

/** Request and response shapes generated from the backend's OpenAPI schema (npm run gen:api). */
type S = components['schemas'];
export type DatasetInfo = S['DatasetInfo'];
export type Job = S['JobRead'];
export type JobEvent = S['JobEventRead'];
type ExperimentNode = S['ExperimentNode'];
type ExperimentRead = S['ExperimentRead'];

/** The dataset the user is working on: an immutable data version of the active project. */
export interface ActiveDataset {
  data_version_id: string;
  filename: string;
  target_column: string;
  /** First 12 hex chars of the data version's SHA-256 (absent in Demo mode). */
  short_hash?: string;
}

/* ---------------- the active project ---------------- */

const PROJECT_KEY = 'mlpilot.project';
let projectId: string | null = null;

function rememberProject(id: string) {
  projectId = id;
  try {
    localStorage.setItem(PROJECT_KEY, id);
  } catch {
    /* storage unavailable: the latest project is picked again next time */
  }
}

/**
 * Pick the project to work in: the one used last, else the most recent one, else a new one.
 * Also the first call to the backend, so it doubles as the connection check.
 */
let pending: Promise<S['ProjectRead']> | null = null;

/** One call at a time: React can mount twice, and two creates would race. */
export function ensureProject(): Promise<S['ProjectRead']> {
  pending ??= pickProject().finally(() => {
    pending = null;
  });
  return pending;
}

async function pickProject(): Promise<S['ProjectRead']> {
  let remembered: string | null = null;
  try {
    remembered = localStorage.getItem(PROJECT_KEY);
  } catch {
    remembered = null;
  }
  const list = (await api.get<S['PaginatedResponse_ProjectRead_']>('/projects/')).data.items;
  const chosen = list.find((p) => p.id === remembered) ?? list[0];
  const project =
    chosen ??
    (
      await api.post<S['ProjectRead']>('/projects/', {
        name: 'My first project',
        task_type: 'binary_classification',
      } satisfies S['ProjectCreate'])
    ).data;
  rememberProject(project.id);
  connection.live = true;
  return project;
}

function project(): string {
  if (!projectId) throw new Error('No project selected yet');
  return `/projects/${projectId}`;
}

const version = (d: ActiveDataset) => `${project()}/datasets/${d.data_version_id}`;
const target = (d: ActiveDataset) => ({ target_column: d.target_column });
const versionBody = (d: ActiveDataset): S['DatasetTargetRequest'] => ({
  data_version_id: d.data_version_id,
  target_column: d.target_column,
});

export const loadSampleDataset = () =>
  api
    .post<DatasetInfo>(`${project()}/datasets/sample`, { dataset_name: 'telecom_churn' } satisfies S['SampleDatasetRequest'])
    .then((r) => r.data);

export const uploadDataset = (file: File) => {
  const form = new FormData();
  form.append('file', file);
  return api
    .post<DatasetInfo>(`${project()}/datasets/upload`, form, { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 60000 })
    .then((r) => r.data);
};

export const getDataMetrics = (d: ActiveDataset) =>
  call(() => api.get<S['DataMetrics']>(`${version(d)}/metrics`, { params: target(d) }).then((r): DataMetrics => r.data), mockMetrics);

export const getLeakageWarnings = (d: ActiveDataset) =>
  call(
    () =>
      api
        .get<S['LeakageFinding'][]>(`${version(d)}/leakage`, { params: target(d) })
        .then((r) => r.data.map((w): LeakageWarning => ({ ...w, category: w.category as LeakageWarning['category'] }))),
    mockWarnings
  );

export const getColumns = (d: ActiveDataset) =>
  call(() => api.get<S['ColumnProfile'][]>(`${version(d)}/columns`, { params: target(d) }).then((r): ColumnProfile[] => r.data), mockColumns);

export const getFeatureSuggestions = (d: ActiveDataset) =>
  call(
    () =>
      api
        .get<S['FeatureSuggestion'][]>(`${version(d)}/suggestions`, { params: target(d), timeout: 120000 })
        .then((r) => r.data.map((s): FeatureSuggestion => ({ ...s, risk: s.risk ?? '' }))),
    mockSuggestions
  );

const toExperiment = (e: ExperimentNode): Experiment => ({
  ...e,
  status: e.status === 'created' ? 'queued' : (e.status as Experiment['status']),
  metrics: Object.fromEntries(Object.entries(e.metrics).filter(([, v]) => v != null)) as Experiment['metrics'],
  feature: e.feature ?? undefined,
  error: e.error ?? undefined,
  hypothesis_mode: (e.hypothesis_mode as Experiment['hypothesis_mode']) ?? null,
});

const queued = (e: ExperimentRead): Experiment => ({
  id: e.id,
  parent_id: e.parent_id ?? null,
  model_name: e.model_name,
  status: 'queued',
  metrics: {},
  runtime_seconds: 0,
  created_at: e.created_at,
  title: e.change_description,
});

export const getExperimentTree = (d: ActiveDataset | null) =>
  call(
    () =>
      api
        .get<ExperimentNode[]>(`${project()}/experiments`, { params: d ? { data_version_id: d.data_version_id } : {} })
        .then((r) => r.data.map(toExperiment)),
    mockExperiments
  );

export const runExperiment = (d: ActiveDataset, suggestion: FeatureSuggestion, model_name: string, parent_id: string | null) =>
  api
    .post<ExperimentRead>(`${project()}/experiments`, {
      ...versionBody(d),
      feature_suggestion: { name: suggestion.name, formula: suggestion.formula, reason: suggestion.reason },
      model_name,
      parent_id,
    } satisfies S['RunExperimentRequest'])
    .then((r) => queued(r.data));

export const runBaseline = (d: ActiveDataset) =>
  api.post<ExperimentRead>(`${project()}/experiments/baseline`, versionBody(d)).then((r) => queued(r.data));

export const runAutoClean = (d: ActiveDataset) =>
  api.post<ExperimentRead>(`${project()}/experiments/auto-clean`, versionBody(d), { timeout: 120000 }).then((r) => queued(r.data));

/** What a finished auto-optimize job reports (DecisionAgent.run_optimization_loop). */
export interface AutoOptimizeResult {
  summary: string;
  best_f1: number;
  winner_features: string[];
}

export const startAutoOptimize = (d: ActiveDataset, n_hypotheses: number) =>
  api
    .post<Job>(`${project()}/agent/auto-optimize`, { ...versionBody(d), n_hypotheses } satisfies S['AutoOptimizeRequest'])
    .then((r) => r.data.id);

/** A background job (auto-optimize, a training run): status, progress and result. */
export const getJob = (jobId: string) =>
  api.get<Job>(`${project()}/jobs/${jobId}`).then((r) => ({
    ...r.data,
    result: r.data.result as AutoOptimizeResult | null | undefined,
  }));

/** The job's events after `after` (the last seq already seen), in order. */
export const getJobEvents = (jobId: string, after: number) =>
  api.get<JobEvent[]>(`${project()}/jobs/${jobId}/events`, { params: { after } }).then((r) => r.data);

export const cancelJob = (jobId: string) => api.post<Job>(`${project()}/jobs/${jobId}/cancel`).then((r) => r.data);

/** Grounded Q&A over the project's recorded experiment history. */
export const chatAsk = (query: string) =>
  api
    .post<S['AskResponse']>(`${project()}/chat/ask`, { query } satisfies S['AskRequest'], { timeout: 120000 })
    .then((r) => r.data.answer);

/** True when the backend answered with an HTTP error (it is reachable). */
export const isHttpError = (e: unknown) => axios.isAxiosError(e) && !!e.response;

export function describeError(e: unknown): string {
  if (axios.isAxiosError(e)) {
    if (e.response?.status === 401)
      return 'The backend refused the access token. Start MLPilot with `python start.py` (it passes the token to the UI).';
    if (e.response) return `HTTP ${e.response.status}: ${JSON.stringify(e.response.data?.detail ?? e.response.statusText)}`;
    return e.message || 'network error';
  }
  return e instanceof Error ? e.message : String(e);
}

export default api;
