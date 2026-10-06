import {
  ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import * as backend from './api';
import type { ActiveDataset } from './api';
import { craftReply, mockDrift, mockEndpoints, mockVolume } from './api/mock';
import {
  AgentStatus,
  CleanStep,
  ColumnProfile,
  DataMetrics,
  Decision,
  Endpoint,
  Experiment,
  FeatureSuggestion,
  LeakageWarning,
  Provider,
  View,
} from './types';
import { MODELS, algoLabel, clamp, defaultParams, f3, importances, inferDtype, parseCsv, rand } from './lib';

export const WORKERS = 2;
export type Tone = 'ok' | 'warn' | 'err' | 'info';
export type MetricKey = 'f1' | 'recall' | 'accuracy';

export interface Toast {
  id: number;
  text: string;
  tone: Tone;
  action?: { label: string; run: () => void };
}
export interface Activity {
  id: number;
  t: number;
  text: string;
  tone: Tone;
}
export type FeedItem =
  | { id: string; kind: 'text'; from: 'agent' | 'user'; text: string; stream?: boolean }
  | { id: string; kind: 'suggestion'; s: FeatureSuggestion; expId?: string }
  | { id: string; kind: 'thinking' }
  | { id: string; kind: 'checkpoint'; question: string; detail: string; options: string[]; choice?: number };

interface Store {
  loading: boolean;
  live: boolean;
  demo: boolean;
  connectionError: string | null;
  started: boolean;
  startApp: (source?: 'sample' | 'upload') => void;
  metrics: DataMetrics | null;
  warnings: LeakageWarning[];
  columns: ColumnProfile[];
  excluded: Set<string>;
  experiments: Experiment[];
  feed: FeedItem[];
  activity: Activity[];
  toasts: Toast[];
  champion: Experiment | null;
  baseline: Experiment | null;
  readiness: number;
  selectedId: string | null;
  view: View;
  agentOpen: boolean;
  paletteOpen: boolean;
  uploadOpen: boolean;
  datasetName: string;
  fileName: string | null;
  target: string;
  cleaning: boolean;
  cleaned: boolean;
  cleanSteps: CleanStep[];
  hypotheses: number;
  agentStatus: AgentStatus;
  compareBaseline: boolean;
  optimizeFor: MetricKey;
  treeOnly: boolean;
  budgetMin: number | null;
  endpoints: Endpoint[];
  provider: Provider;
  apiKey: string;
  providerModel: string;
  drift: { t: number; psi: number }[];
  volume: { h: number; n: number }[];
  setView: (v: View) => void;
  setAgentOpen: (b: boolean) => void;
  setPaletteOpen: (b: boolean) => void;
  setUploadOpen: (b: boolean) => void;
  setTarget: (t: string) => void;
  setHypotheses: (n: number) => void;
  setCompareBaseline: (b: boolean) => void;
  setProvider: (p: Provider) => void;
  setApiKey: (k: string) => void;
  setProviderModel: (m: string) => void;
  select: (id: string | null) => void;
  toggleColumn: (name: string) => void;
  excludeHighRisk: () => void;
  ingestCsv: (file: File) => Promise<void>;
  autoClean: () => void;
  answerCheckpoint: (id: string, choice: number) => void;
  runSuggestion: (feedId: string, modelKey?: string) => void;
  runNext: () => void;
  runAutopilot: () => void;
  dismissSuggestion: (feedId: string) => void;
  askAgent: (prompt: string) => Promise<void>;
  fork: (id: string, modelKey: string) => void;
  retry: (id: string) => void;
  cancel: (id: string) => void;
  promote: (id: string) => void;
  remove: (id: string) => void;
  setDecision: (id: string, d: Decision) => void;
  deploy: (id: string) => void;
  retireEndpoint: (id: string) => void;
  toast: (text: string, tone?: Tone, action?: Toast['action']) => void;
  dismissToast: (id: number) => void;
}

const Ctx = createContext<Store>(null as unknown as Store);
export const useStore = () => useContext(Ctx);

type DistOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const r3 = (n: number) => Math.round(n * 1000) / 1000;

function bestOf(list: Experiment[], metric: MetricKey) {
  return (
    list
      .filter((e) => e.status === 'completed' && e.metrics[metric] !== undefined && e.decision !== 'reject')
      .sort((a, b) => (b.metrics[metric] ?? 0) - (a.metrics[metric] ?? 0))[0] ?? null
  );
}
function pickChampion(list: Experiment[], id: string | null, metric: MetricKey) {
  // Live mode: the backend decides the champion (accepted by rule, not by best test score).
  const decided = list.find((e) => e.champion);
  if (decided) return decided;
  return list.find((e) => e.id === id && e.status === 'completed' && e.decision !== 'reject') ?? bestOf(list, metric);
}

// Demo mode only: sample metrics for simulated runs.
function simulateMetrics(e: Experiment, all: Experiment[]) {
  const parent = all.find((p) => p.id === e.parent_id)?.metrics.f1 ?? 0.72;
  const boost = /xgb|lgbm|cat/i.test(e.model_name) ? 0.004 : 0;
  const delta = e.feature ? rand(-0.006, 0.028) : rand(-0.015, 0.015);
  const f1 = clamp(parent + delta + boost, 0.55, 0.965);
  return {
    f1: r3(f1),
    accuracy: r3(clamp(f1 + rand(0.06, 0.09), 0, 0.985)),
    precision: r3(clamp(f1 + rand(-0.03, 0.02), 0, 0.99)),
    recall: r3(clamp(f1 + rand(-0.02, 0.04), 0, 0.99)),
  };
}

const CLEAN: Omit<CleanStep, 'done'>[] = [
  { id: 'leak', label: 'Drop leaky columns', detail: 'Remove features flagged by the leakage report' },
  { id: 'dup', label: 'Deduplicate rows', detail: 'Exact-match collapse' },
  { id: 'miss', label: 'Impute missing', detail: 'Median numeric · mode categorical' },
  { id: 'scale', label: 'Scale numerics', detail: 'RobustScaler fit on train fold only' },
  { id: 'enc', label: 'Encode categoricals', detail: 'Target encoding with CV folds' },
];

const DATASET_KEY = 'mlpilot.dataset';

function readDataset(): ActiveDataset | null {
  try {
    const raw = sessionStorage.getItem(DATASET_KEY);
    const d = raw ? (JSON.parse(raw) as Partial<ActiveDataset>) : null;
    // Saved by an older version (path-based): drop it rather than send a path.
    return d?.data_version_id ? (d as ActiveDataset) : null;
  } catch {
    return null;
  }
}

function saveDataset(d: ActiveDataset) {
  try {
    sessionStorage.setItem(DATASET_KEY, JSON.stringify(d));
  } catch {
    /* private mode: dataset choice not remembered across reloads */
  }
}

/** Models the backend can train (MODEL_REGISTRY in ml/experiments/executor.py). */
const BACKEND_MODELS = new Set(['XGBClassifier', 'LGBMClassifier', 'RandomForestClassifier', 'LogisticRegression', 'GradientBoostingClassifier']);

function pickTarget(columns: string[], suggested?: string): string {
  if (suggested && columns.includes(suggested)) return suggested;
  return columns.find((h) => /churn|label|target|y$/i.test(h)) ?? columns[columns.length - 1] ?? '';
}

function readFlag(): boolean {
  try {
    return sessionStorage.getItem('mlpilot.started') === '1';
  } catch {
    return false;
  }
}

export function StoreProvider({ children }: { children: ReactNode }) {
  const [loading, setLoading] = useState(true);
  const [live, setLive] = useState(false);
  const demo = backend.connection.demo;
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [dataset, setDatasetState] = useState<ActiveDataset | null>(() => (backend.connection.demo ? null : readDataset()));
  const datasetRef = useRef<ActiveDataset | null>(dataset);
  const [started, setStarted] = useState(readFlag);
  const [metrics, setMetrics] = useState<DataMetrics | null>(null);
  const [warnings, setWarnings] = useState<LeakageWarning[]>([]);
  const [columns, setColumns] = useState<ColumnProfile[]>([]);
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [feed, setFeed] = useState<FeedItem[]>([]);
  const [activity, setActivity] = useState<Activity[]>([]);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [championId, setChampionId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [view, setView] = useState<View>('dashboard');
  const [agentOpen, setAgentOpen] = useState(true);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [datasetName, setDatasetName] = useState(backend.connection.demo ? 'churn_v3' : 'no dataset');
  const [fileName, setFileName] = useState<string | null>(backend.connection.demo ? 'churn_customers.csv' : null);
  const [target, setTargetState] = useState(backend.connection.demo ? 'is_churned' : '—');
  const [cleaning, setCleaning] = useState(false);
  const [cleaned, setCleaned] = useState(false);
  const [cleanSteps, setCleanSteps] = useState<CleanStep[]>(CLEAN.map((s) => ({ ...s, done: false })));
  const [hypotheses, setHypotheses] = useState(3);
  const [agentStatus, setAgentStatus] = useState<AgentStatus>('idle');
  const [compareBaseline, setCompareBaseline] = useState(true);
  const [optimizeFor, setOptimizeFor] = useState<MetricKey>('f1');
  const [treeOnly, setTreeOnly] = useState(false);
  const [budgetMin, setBudgetMin] = useState<number | null>(null);
  const [endpoints, setEndpoints] = useState<Endpoint[]>(mockEndpoints);
  const [provider, setProvider] = useState<Provider>('anthropic');
  const [apiKey, setApiKey] = useState('');
  const [providerModel, setProviderModel] = useState('claude-sonnet-4-5');
  const [drift] = useState(mockDrift);
  const [volume] = useState(mockVolume);

  const expRef = useRef<Experiment[]>([]);
  const feedRef = useRef<FeedItem[]>([]);
  const columnsRef = useRef<ColumnProfile[]>([]);
  const warningsRef = useRef<LeakageWarning[]>([]);
  const excludedRef = useRef<Set<string>>(new Set());
  const championRef = useRef<string | null>(null);
  const metricRef = useRef<MetricKey>('f1');
  const treeOnlyRef = useRef(false);
  const midNarrated = useRef<Set<string>>(new Set());
  const pendingCheckpoint = useRef<{ id: string; names: string[] } | null>(null);
  const budgetTimer = useRef<number | null>(null);
  const autoBusy = useRef(false);
  const uid = useRef(100);
  const expCounter = useRef(30);

  useEffect(() => {
    feedRef.current = feed;
  }, [feed]);
  useEffect(() => {
    columnsRef.current = columns;
  }, [columns]);
  useEffect(() => {
    warningsRef.current = warnings;
  }, [warnings]);
  useEffect(() => {
    excludedRef.current = excluded;
  }, [excluded]);
  useEffect(() => {
    metricRef.current = optimizeFor;
  }, [optimizeFor]);
  useEffect(() => {
    treeOnlyRef.current = treeOnly;
  }, [treeOnly]);

  const commit = useCallback((next: Experiment[]) => {
    expRef.current = next;
    setExperiments(next);
  }, []);

  const pushFeed = useCallback((item: DistOmit<FeedItem, 'id'> & { id?: string }) => {
    const full = { ...item, id: item.id ?? `m${++uid.current}` } as FeedItem;
    setFeed((f) => [...f, full]);
    return full.id;
  }, []);

  const log = useCallback((text: string, tone: Tone = 'info') => {
    setActivity((a) => [{ id: ++uid.current, t: Date.now(), text, tone }, ...a].slice(0, 40));
  }, []);

  const dismissToast = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), []);
  const toast = useCallback((text: string, tone: Tone = 'info', action?: Toast['action']) => {
    const id = ++uid.current;
    setToasts((t) => [...t.slice(-3), { id, text, tone, action }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), action ? 6500 : 3600);
  }, []);

  /** Actions that would produce numbers need the backend, unless Demo mode is on. */
  const requireBackend = useCallback(() => {
    if (demo || live) return true;
    toast('Not connected to the backend. Start it and reload, or turn on Demo mode.', 'err');
    return false;
  }, [demo, live, toast]);

  /* ---------------- real backend: load a dataset ---------------- */
  const setDataset = useCallback((d: ActiveDataset) => {
    datasetRef.current = d;
    setDatasetState(d);
    saveDataset(d);
  }, []);

  const loadDataset = useCallback(
    async (d: ActiveDataset) => {
      setLoading(true);
      setDatasetName(d.filename.replace(/\.csv$/i, ''));
      setFileName(d.filename);
      setTargetState(d.target_column);
      let loaded;
      try {
        loaded = await Promise.all([
          backend.getDataMetrics(d),
          backend.getLeakageWarnings(d),
          backend.getColumns(d),
          backend.getExperimentTree(d),
        ]);
      } catch (err) {
        const msg = backend.describeError(err);
        if (!backend.isHttpError(err)) setConnectionError(msg);
        setFeed([{ id: 'intro', kind: 'text', from: 'agent', text: `Could not load ${d.filename}: ${msg}` }]);
        setLoading(false);
        return;
      }
      const [m, w, c, e] = loaded;
      setLive(true);
      setConnectionError(null);
      setMetrics(m);
      setWarnings(w);
      setColumns(c);
      commit(e);
      setFeed([
        {
          id: 'intro',
          kind: 'text',
          from: 'agent',
          text: `Loaded ${d.filename}${d.short_hash ? ` (data version ${d.short_hash})` : ''}: ${m.total_rows.toLocaleString('en-US')} rows, ${m.total_columns} columns, target ${d.target_column}. ${w.length} leakage findings (${w.filter((x) => x.severity === 'high').length} high-risk). ${e.length ? `${e.length} recorded runs on this dataset.` : 'Queuing a baseline run now.'} Asking the planner for feature ideas…`,
        },
      ]);
      setLoading(false);
      if (!e.length) {
        try {
          const b = await backend.runBaseline(d);
          log(`Baseline queued · ${algoLabel(b.model_name)}`, 'info');
        } catch (err) {
          toast(`Baseline failed to start: ${backend.describeError(err)}`, 'err');
        }
      }
      try {
        const sug = await backend.getFeatureSuggestions(d);
        const stamp = Date.now();
        setFeed((f) => [...f, ...sug.map((x, i): FeedItem => ({ id: `s${stamp}_${i}`, kind: 'suggestion', s: x }))]);
      } catch (err) {
        setFeed((f) => [...f, { id: `m${++uid.current}`, kind: 'text', from: 'agent', text: `Feature suggestions failed: ${backend.describeError(err)}` }]);
      }
    },
    [commit, log, toast]
  );

  /* ---------------- real backend: initial connection ---------------- */
  useEffect(() => {
    if (demo) return;
    (async () => {
      try {
        await backend.ensureProject();
      } catch (err) {
        // No sample data outside Demo mode: show the failure, not numbers.
        setConnectionError(backend.describeError(err));
        setLive(false);
        setFeed([
          {
            id: 'intro',
            kind: 'text',
            from: 'agent',
            text: `I can't reach the MLPilot backend at ${backend.API_BASE_URL}, so there is nothing real to show yet. Start the backend and retry, or turn on Demo mode to explore with sample data.`,
          },
        ]);
        setLoading(false);
        return;
      }
      setLive(true);
      const d = datasetRef.current;
      if (d) await loadDataset(d);
      else {
        setFeed([{ id: 'intro', kind: 'text', from: 'agent', text: 'Connected to the backend. Load the sample dataset or upload a CSV to start.' }]);
        setLoading(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ---------------- Demo mode: sample data only ---------------- */
  useEffect(() => {
    if (!demo) return;
    let dead = false;
    (async () => {
      let loaded;
      const sample: ActiveDataset = { data_version_id: 'demo', filename: 'churn_customers.csv', target_column: 'is_churned' };
      try {
        loaded = await Promise.all([
          backend.getDataMetrics(sample),
          backend.getLeakageWarnings(sample),
          backend.getColumns(sample),
          backend.getFeatureSuggestions(sample),
          backend.getExperimentTree(sample),
        ]);
      } catch (err) {
        if (dead) return;
        // No sample data outside Demo mode: show the failure, not numbers.
        setConnectionError(backend.describeError(err));
        setLive(false);
        setFeed([
          {
            id: 'intro',
            kind: 'text',
            from: 'agent',
            text: `I can't reach the MLPilot backend at ${backend.API_BASE_URL}, so there is nothing real to show yet. Start the backend and retry, or turn on Demo mode to explore with sample data.`,
          },
        ]);
        setLoading(false);
        return;
      }
      if (dead) return;
      const [m, w, c, s, e] = loaded;
      setMetrics(m);
      setWarnings(w);
      setColumns(c);
      const tgt = c.find((x) => x.role === 'target')?.name ?? 'is_churned';
      setTargetState(tgt);
      commit(e);
      setLive(backend.connection.live);
      const bestF1 = Math.max(0, ...e.map((y) => y.metrics.f1 ?? 0));
      const bestId = e.find((x) => x.metrics.f1 === bestF1)?.id ?? '—';
      const intro: FeedItem[] = [
        {
          id: 'intro',
          kind: 'text',
          from: 'agent',
          text: `Profiled ${m.total_rows.toLocaleString('en-US')} rows against ${tgt}. ${w.length} leakage findings (${w.filter((x) => x.severity === 'high').length} high-risk), ${s.length} feature hypotheses ready. Champion so far: ${bestId} at ${bestF1.toFixed(3)} F1. Ask me about any of it — status, leakage, or why a run was rejected.`,
        },
        ...s.map(
          (x, i): FeedItem => ({
            id: `s${i}`,
            kind: 'suggestion',
            s: x,
            expId: e.find((ex) => ex.feature === x.name)?.id,
          })
        ),
      ];
      setFeed(intro);
      const now = Date.now();
      // Sample activity only in Demo mode; real activity comes from real runs.
      if (demo) setActivity([
        { id: 1, t: now - 60_000, text: 'exp_023 queued · CatBoost native categoricals', tone: 'info' },
        { id: 2, t: now - 2 * 60_000, text: 'exp_022 started · LightGBM', tone: 'info' },
        { id: 3, t: now - 44 * 60_000, text: 'exp_021 finished · F1 0.840', tone: 'ok' },
        { id: 4, t: now - 58 * 60_000, text: 'exp_018 failed · out of memory', tone: 'err' },
        { id: 5, t: now - 96 * 60_000, text: 'Baseline logged · F1 0.720', tone: 'ok' },
      ]);
      setLoading(false);
    })();
    return () => {
      dead = true;
    };
  }, [commit, demo]);

  /* ---------------- live polling (real backend) ---------------- */
  useEffect(() => {
    if (!live) return;
    const t = setInterval(() => {
      const d = datasetRef.current;
      if (!d) return;
      backend
        .getExperimentTree(d)
        .then(commit)
        .catch((err) => setConnectionError(backend.isHttpError(err) ? null : backend.describeError(err)));
    }, 4000);
    return () => clearInterval(t);
  }, [live, commit]);

  /* ---------------- local worker-pool simulation + live narration ---------------- */
  useEffect(() => {
    // Simulated runs are sample data: only in Demo mode.
    if (loading || live || !demo) return;
    const t = setInterval(() => {
      const prev = expRef.current;
      let running = prev.filter((e) => e.status === 'running').length;
      const mid: string[] = [];
      const finished: Experiment[] = [];
      let changed = false;
      const mk = metricRef.current;
      const next = prev.map((e): Experiment => {
        if (e.status === 'running') {
          changed = true;
          const progress = Math.min(100, (e.progress ?? 0) + rand(5, 12));
          const runtime = +(e.runtime_seconds + 0.8).toFixed(1);
          if (progress < 100) {
            if (progress >= 50 && !midNarrated.current.has(e.id)) {
              midNarrated.current.add(e.id);
              mid.push(e.id);
            }
            return { ...e, progress, runtime_seconds: runtime };
          }
          // Demo mode only (this effect returns early unless demo).
          if (Math.random() < 0.06) {
            const f: Experiment = {
              ...e,
              status: 'failed',
              progress,
              runtime_seconds: runtime,
              error: 'MemoryError: unable to allocate 3.1 GiB for an array with shape (148920, 2800)',
              decision: 'reject',
            };
            finished.push(f);
            return f;
          }
          const m = simulateMetrics(e, prev);
          const f: Experiment = { ...e, status: 'completed', progress: 100, runtime_seconds: runtime, metrics: m, decision: 'keep' };
          finished.push(f);
          return f;
        }
        if (e.status === 'queued' && running < WORKERS) {
          running++;
          changed = true;
          return { ...e, status: 'running', progress: 0 };
        }
        return e;
      });
      if (!changed) return;
      commit(next);

      finished.forEach((f) => {
        const mv = f.metrics[mk];
        const parent = f.parent_id ? next.find((e) => e.id === f.parent_id) : undefined;
        const pm = parent?.metrics[mk];
        const ch = next.find((e) => e.id === championRef.current);
        const cm = ch?.metrics[mk];
        log(`${f.id} ${f.status} · ${mk} ${mv !== undefined ? f3(mv) : 'n/a'}`, f.status === 'failed' ? 'err' : 'ok');
        if (f.status === 'failed') {
          pushFeed({ kind: 'text', from: 'agent', text: `${f.id} failed — ${f.error}. Marking reject; you can retry it from the inspector.` });
          return;
        }
        if (mv === undefined) return;
        const newChampion = f.champion !== undefined ? f.champion : cm !== undefined && mv > cm;
        if (newChampion) {
          championRef.current = f.id;
          setChampionId(f.id);
          pushFeed({
            kind: 'text',
            from: 'agent',
            text: `${f.id} done — new champion. ${mk} ${mv.toFixed(3)}${cm !== undefined ? `, previous champion ${ch?.id} had ${cm.toFixed(3)}` : ''}. Keeping ${f.feature ? `${f.feature} ` : ''}in the working set.`,
          });
        } else if (f.champion === undefined && f.feature && pm !== undefined && mv < pm) {
          commit(expRef.current.map((e) => (e.id === f.id ? { ...e, decision: 'reject' as Decision } : e)));
          pushFeed({
            kind: 'text',
            from: 'agent',
            text: `${f.id} done — ${f.feature} dropped ${mk} from ${pm.toFixed(3)} to ${mv.toFixed(3)}. Below its parent, so I'm marking it reject.`,
          });
        } else {
          pushFeed({
            kind: 'text',
            from: 'agent',
            text: `${f.id} done — ${mk} ${mv.toFixed(3)}${pm !== undefined ? ` (parent ${pm.toFixed(3)})` : ''}. Holds position${cm !== undefined && mv < cm ? ' under the champion' : ''}.`,
          });
        }
      });

      mid.forEach((id) => {
        const e = next.find((x) => x.id === id);
        if (e) pushFeed({ kind: 'text', from: 'agent', text: `${id} at 50% — fold 3/5, ${e.runtime_seconds.toFixed(1)}s in.` });
      });

      // autopilot finished everything → grounded debrief
      const anyLeft = next.some((e) => e.status === 'running' || e.status === 'queued');
      if (!anyLeft && finished.length && agentStatusRef.current === 'autopilot') {
        autoBusy.current = false;
        setAgentStatus('idle');
        pushFeed({ kind: 'text', from: 'agent', text: debrief() });
      }
    }, 800);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading, live, demo, commit, log, pushFeed]);

  const agentStatusRef = useRef<AgentStatus>('idle');
  useEffect(() => {
    agentStatusRef.current = agentStatus;
  }, [agentStatus]);

  /* ---------------- grounded debrief, built only from live state ---------------- */
  const debrief = useCallback(() => {
    const list = expRef.current;
    const ch = pickChampion(list, championRef.current, metricRef.current);
    const base = list.find((e) => e.decision === 'baseline') ?? list.find((e) => !e.parent_id);
    const mk = metricRef.current;
    if (!ch || ch.metrics[mk] === undefined) return 'No completed runs yet, so there is nothing to debrief. Queue a hypothesis first.';
    const done = list.filter((e) => e.status === 'completed').length;
    const tested = list.filter((e) => e.feature).length;
    const baseM = base?.metrics[mk];
    if (!backend.connection.demo) {
      return `Debrief. ${algoLabel(ch.model_name)} (${ch.id}) leads at ${mk} ${f3(ch.metrics[mk])} on ${list.length} total runs, ${done} completed, ${tested} feature hypotheses tested since ${base?.id ?? 'the baseline'}${baseM !== undefined ? ` (${f3(baseM)})` : ''}. Feature importances are not available yet.`;
    }
    const imps = importances(ch, columnsRef.current);
    const drivers = imps.slice(0, 3).map((i, n) => `${['first', 'second', 'third'][n]} ${i.name} (${(i.weight * 100).toFixed(0)}%)`).join(', ');
    return `Debrief. ${algoLabel(ch.model_name)} (${ch.id}) leads at ${mk} ${f3(ch.metrics[mk])} on ${list.length} total runs, ${done} completed, ${tested} feature hypotheses tested since ${base?.id ?? 'the baseline'}${baseM !== undefined ? ` (${f3(baseM)})` : ''}. What drives the model, in order: ${drivers}. Nothing else cleared the keep line. The full ranked explanation is in the champion's Importance tab.`;
  }, []);

  /* ---------------- session / app start ---------------- */
  const startApp = useCallback((source?: 'sample' | 'upload') => {
    setStarted(true);
    try {
      sessionStorage.setItem('mlpilot.started', '1');
    } catch {
      /* private mode */
    }
    if (source === 'upload') setUploadOpen(true);
    if (source === 'sample' && !backend.connection.demo) {
      backend
        .loadSampleDataset()
        .then((info) => {
          const d = { data_version_id: info.data_version_id, filename: info.filename, target_column: pickTarget(info.columns, info.default_target ?? undefined), short_hash: info.short_hash };
          setDataset(d);
          return loadDataset(d);
        })
        .catch((err) => {
          if (!backend.isHttpError(err)) setConnectionError(backend.describeError(err));
          toast(`Could not load the sample dataset: ${backend.describeError(err)}`, 'err');
        });
    }
  }, [setDataset, loadDataset, toast]);

  /* ---------------- actions ---------------- */
  const select = useCallback((id: string | null) => setSelectedId(id), []);

  const flip = useCallback((name: string) => {
    setExcluded((p) => {
      const n = new Set(p);
      if (n.has(name)) n.delete(name);
      else n.add(name);
      return n;
    });
  }, []);

  const toggleColumn = useCallback(
    (name: string) => {
      const was = excluded.has(name);
      flip(name);
      log(`${was ? 'Restored' : 'Excluded'} ${name}`, was ? 'info' : 'ok');
      toast(`${was ? 'Restored' : 'Excluded'} ${name}`, was ? 'info' : 'ok', {
        label: 'Undo',
        run: () => flip(name),
      });
    },
    [excluded, flip, log, toast]
  );

  const excludeHighRisk = useCallback(() => {
    const names = warnings.filter((w) => w.severity === 'high' && !excluded.has(w.column)).map((w) => w.column);
    if (!names.length) return;
    setExcluded((p) => new Set([...p, ...names]));
    log(`Excluded ${names.length} high-risk columns`, 'ok');
    toast(`Excluded ${names.join(', ')}`, 'ok', {
      label: 'Undo',
      run: () =>
        setExcluded((p) => {
          const n = new Set(p);
          names.forEach((x) => n.delete(x));
          return n;
        }),
    });
  }, [warnings, excluded, log, toast]);

  const setTarget = useCallback(
    (t: string) => {
      const d = datasetRef.current;
      if (!demo && d) {
        const next = { ...d, target_column: t };
        setDataset(next);
        void loadDataset(next);
        log(`Target set to ${t}`, 'info');
        return;
      }
      setTargetState(t);
      setColumns((cols) => cols.map((c) => ({ ...c, role: c.name === t ? 'target' : c.role === 'target' ? 'feature' : c.role })));
      log(`Target set to ${t}`, 'info');
    },
    [log, demo, setDataset, loadDataset]
  );

  const ingestCsv = useCallback(
    async (file: File) => {
      if (!requireBackend()) return;
      if (!demo) {
        try {
          const info = await backend.uploadDataset(file);
          const d = { data_version_id: info.data_version_id, filename: info.filename, target_column: pickTarget(info.columns), short_hash: info.short_hash };
          setDataset(d);
          setUploadOpen(false);
          toast(`Uploaded ${info.filename}`, 'ok');
          await loadDataset(d);
        } catch (err) {
          toast(`Upload failed: ${backend.describeError(err)}`, 'err');
        }
        return;
      }
      const text = await file.text();
      const { headers, rows, samples } = parseCsv(text);
      if (!headers.length) {
        toast('Could not parse CSV headers', 'err');
        return;
      }
      const guessTarget = headers.find((h) => /churn|label|target|y$/i.test(h)) ?? headers[headers.length - 1];
      const cols: ColumnProfile[] = headers.map((name, i) => {
        const dtype = inferDtype(name, samples[i] ?? '');
        const role: ColumnProfile['role'] = /id$/i.test(name) ? 'id' : name === guessTarget ? 'target' : 'feature';
        // Demo mode only: real uploads return above with backend stats.
        const dist = Array.from({ length: 12 }, () => Math.random());
        const mx = Math.max(...dist) || 1;
        return {
          name,
          dtype,
          role,
          missing_pct: Math.round(Math.random() * 8 * 10) / 10,
          unique: Math.min(rows, Math.round(rows * (0.2 + Math.random() * 0.8))),
          dist: dist.map((d) => d / mx),
        };
      });
      setColumns(cols);
      setMetrics({
        total_rows: rows,
        total_columns: headers.length,
        missing_data_percent: 0.02 + Math.random() * 0.04,
        duplicate_rows: Math.round(rows * 0.0015),
      });
      setTargetState(guessTarget);
      setDatasetName(file.name.replace(/\.(csv|tsv)$/i, ''));
      setFileName(file.name);
      setWarnings([]);
      setCleaned(false);
      setCleanSteps(CLEAN.map((s) => ({ ...s, done: false })));
      setUploadOpen(false);
      log(`Ingested ${file.name} · ${rows.toLocaleString()} rows`, 'ok');
      toast(`Loaded ${file.name}`, 'ok');
    },
    [log, toast, requireBackend, demo, setDataset, loadDataset]
  );

  /* ---------------- auto-clean with a human-in-the-loop checkpoint ---------------- */
  const runCleanSteps = useCallback(
    (startIdx: number) => {
      CLEAN.slice(startIdx).forEach((step, i) => {
        setTimeout(() => {
          setCleanSteps((prev) => prev.map((s, j) => (j === startIdx + i ? { ...s, done: true } : s)));
          if (step.id === 'dup') setMetrics((m) => (m ? { ...m, duplicate_rows: 0 } : m));
          if (step.id === 'miss') {
            setMetrics((m) => (m ? { ...m, missing_data_percent: 0 } : m));
            setColumns((cols) => cols.map((c) => ({ ...c, missing_pct: excludedRef.current.has(c.name) ? c.missing_pct : 0 })));
          }
          if (i === CLEAN.length - 1 - startIdx) {
            setCleaning(false);
            setCleaned(true);
            setAgentStatus('idle');
            log('Autoclean finished · feature set ready', 'ok');
            toast('Pipeline applied', 'ok');
          }
        }, 700 * (i + 1));
      });
    },
    [log, toast]
  );

  const autoClean = useCallback(() => {
    if (cleaning || !requireBackend()) return;
    if (!demo) {
      const d = datasetRef.current;
      if (!d) return toast('Load a dataset first.', 'warn');
      setCleaning(true);
      backend
        .runAutoClean(d)
        .then((exp) => {
          log(`Auto-clean run queued · ${exp.id}`, 'info');
          toast('AI cleaning recipe accepted; training a cleaned baseline', 'ok');
          setCleaned(true);
        })
        .catch((err) => toast(`Auto-clean failed: ${backend.describeError(err)}`, 'err'))
        .finally(() => setCleaning(false));
      return;
    }
    setCleaning(true);
    setCleaned(false);
    setCleanSteps(CLEAN.map((s) => ({ ...s, done: false })));
    setAgentStatus('thinking');
    const names = warningsRef.current.filter((w) => w.severity === 'high' && !excludedRef.current.has(w.column)).map((w) => w.column);
    if (names.length) {
      const id = `cp${++uid.current}`;
      pushFeed({ kind: 'thinking' });
      setTimeout(() => {
        pushFeed({
          kind: 'checkpoint',
          id,
          question: `Drop ${names.join(' + ')} from the feature matrix?`,
          detail: `Both carry ${names.map((n) => warningsRef.current.find((w) => w.column === n)?.category).join(' / ')} leakage flags. Dropping them changes the feature set for every future run. This is a decision I don't make unilaterally.`,
          options: ['Drop them, then build the pipeline', 'Leave them in for now'],
        });
        pendingCheckpoint.current = { id, names };
      }, 600);
    } else {
      runCleanSteps(0);
    }
  }, [cleaning, pushFeed, runCleanSteps, requireBackend, demo, log, toast]);

  const answerCheckpoint = useCallback(
    (id: string, choice: number) => {
      const pc = pendingCheckpoint.current;
      if (!pc || pc.id !== id) return;
      pendingCheckpoint.current = null;
      setFeed((f) => f.map((x) => (x.id === id && x.kind === 'checkpoint' ? { ...x, choice } : x)));
      const item = feedRef.current.find((x) => x.id === id);
      const opt = item && item.kind === 'checkpoint' ? item.options[choice] : '';
      pushFeed({ kind: 'text', from: 'user', text: opt });
      if (choice === 0) {
        setExcluded((p) => new Set([...p, ...pc.names]));
        log(`Excluded ${pc.names.length} high-risk columns (agent checkpoint)`, 'ok');
      }
      setAgentStatus('thinking');
      setTimeout(() => {
        pushFeed({
          kind: 'text',
          from: 'agent',
          text: choice === 0 ? `Understood. Dropped ${pc.names.join(', ')} — ${pc.names.length} fewer columns to leak. Building the pipeline.` : 'Noted — leaving them in. I will keep flagging them in every leakage report until you decide.',
        });
        setAgentStatus('idle');
        runCleanSteps(0);
      }, 500);
    },
    [pushFeed, runCleanSteps, log]
  );

  /* ---------------- experiment actions ---------------- */
  const enqueue = useCallback(
    (exp: Omit<Experiment, 'id' | 'status' | 'metrics' | 'runtime_seconds' | 'created_at' | 'progress' | 'decision'>) => {
      const id = `exp_${String(++expCounter.current).padStart(3, '0')}`;
      const full: Experiment = { ...exp, id, status: 'queued', metrics: {}, runtime_seconds: 0, created_at: new Date().toISOString(), progress: 0, decision: 'none' };
      commit([...expRef.current, full]);
      return full;
    },
    [commit]
  );

  const pickModelKey = useCallback(() => {
    const available = backend.connection.demo ? MODELS : MODELS.filter((m) => BACKEND_MODELS.has(m.cls));
    const pool = treeOnlyRef.current ? available.filter((m) => m.tree) : available;
    // Rotate models; real runs don't go through enqueue, so advance the counter here.
    return pool[(backend.connection.demo ? expCounter.current : expCounter.current++) % pool.length].key;
  }, []);

  const runSuggestion = useCallback(
    (feedId: string, modelKey?: string) => {
      if (!requireBackend()) return;
      const item = feedRef.current.find((f) => f.id === feedId);
      if (!item || item.kind !== 'suggestion' || item.expId) return;
      const key = modelKey ?? pickModelKey();
      const m = MODELS.find((x) => x.key === key) ?? MODELS[0];
      const parent = pickChampion(expRef.current, championRef.current, metricRef.current);
      if (!demo) {
        const d = datasetRef.current;
        if (!d) return toast('Load a dataset first.', 'warn');
        if (!BACKEND_MODELS.has(m.cls)) return toast(`${m.label} isn't available in the backend yet.`, 'warn');
        backend
          .runExperiment(d, item.s, m.cls, parent?.id ?? null)
          .then((exp) => {
            setFeed((f) => f.map((x) => (x.id === feedId && x.kind === 'suggestion' ? { ...x, expId: exp.id } : x)));
            log(`${exp.id} queued · ${m.label} + ${item.s.name}`, 'info');
            toast(`Queued ${exp.id}`, 'info', {
              label: 'View',
              run: () => {
                setView('experiments');
                setSelectedId(exp.id);
              },
            });
          })
          .catch((err) => toast(`Run failed to start: ${backend.describeError(err)}`, 'err'));
        return;
      }
      const exp = enqueue({
        parent_id: parent?.id ?? null,
        model_name: m.cls,
        title: `+ ${item.s.name}`,
        feature: item.s.name,
        params: defaultParams(m.cls),
      });
      setFeed((f) => f.map((x) => (x.id === feedId && x.kind === 'suggestion' ? { ...x, expId: exp.id } : x)));
      log(`${exp.id} queued · ${m.label} + ${item.s.name}`, 'info');
      toast(`Queued ${exp.id}`, 'info', {
        label: 'View',
        run: () => {
          setView('experiments');
          setSelectedId(exp.id);
        },
      });
    },
    [enqueue, pickModelKey, log, toast, requireBackend, demo]
  );

  const runNext = useCallback(() => {
    const next = feedRef.current.find((f) => f.kind === 'suggestion' && !f.expId);
    if (!next) return toast('No pending ideas — steer the agent for more.', 'warn');
    runSuggestion(next.id);
  }, [runSuggestion, toast]);

  const runAutopilot = useCallback(() => {
    if (autoBusy.current || !requireBackend()) return;
    if (!demo) {
      const d = datasetRef.current;
      if (!d) return toast('Load a dataset first.', 'warn');
      autoBusy.current = true;
      setAgentStatus('autopilot');
      log(`Autopilot · ${hypotheses} hypotheses`, 'info');
      const done = (text: string, tone: Tone) => {
        autoBusy.current = false;
        setAgentStatus('idle');
        pushFeed({ kind: 'text', from: 'agent', text });
        log(text, tone);
      };
      backend
        .startAutoOptimize(d, hypotheses)
        .then((jobId) => {
          let seen = 0;
          const poll = window.setInterval(() => {
            backend
              .getJobEvents(jobId, seen)
              .then((events) => {
                // Narrate what the job recorded, in order (decisions come from the rule).
                for (const e of events) {
                  seen = e.seq;
                  const p = e.payload as Record<string, unknown>;
                  if (e.type === 'proposal' && p.name) log(`Autopilot proposed ${String(p.name)} = ${String(p.formula)}`, 'info');
                  if (e.type === 'decision') log(`Autopilot ${p.decision === 'keep' ? 'kept' : 'rejected'} ${String(p.name)}`, p.decision === 'keep' ? 'ok' : 'warn');
                }
                return backend.getJob(jobId);
              })
              .then((job) => {
                if (job.status === 'queued' || job.status === 'running') return;
                window.clearInterval(poll);
                if (job.status === 'succeeded') done(`Autopilot finished. ${job.result?.summary ?? ''}`, 'ok');
                else if (job.status === 'cancelled') done('Autopilot was cancelled.', 'warn');
                else done(`Autopilot failed: ${job.error ?? 'unknown error'}`, 'err');
              })
              .catch((err) => {
                window.clearInterval(poll);
                done(`Lost track of the autopilot job: ${backend.describeError(err)}`, 'err');
              });
          }, 2000);
        })
        .catch((err) => done(`Autopilot failed to start: ${backend.describeError(err)}`, 'err'));
      return;
    }
    const pending = feedRef.current.filter((f) => f.kind === 'suggestion' && !f.expId).slice(0, hypotheses);
    if (!pending.length) {
      toast('No unused hypotheses. Steer the agent first.', 'warn');
      return;
    }
    autoBusy.current = true;
    setAgentStatus('autopilot');
    log(`Autopilot · ${pending.length} hypotheses`, 'info');
    toast(`Autopilot running ${pending.length} experiments`, 'info');
    pending.forEach((p, i) => {
      setTimeout(() => runSuggestion(p.id), 420 * i);
    });
    if (budgetMin && budgetTimer.current) window.clearTimeout(budgetTimer.current);
    if (budgetMin) {
      budgetTimer.current = window.setTimeout(() => {
        if (agentStatusRef.current === 'autopilot') {
          setAgentStatus('idle');
          autoBusy.current = false;
          pushFeed({ kind: 'text', from: 'agent', text: `Time budget hit (${budgetMin} min). Autopilot is paused; in-flight runs will still finish and report in.` });
        }
      }, budgetMin * 60_000);
    }
  }, [hypotheses, runSuggestion, log, toast, budgetMin, pushFeed, requireBackend, demo]);

  const dismissSuggestion = useCallback((feedId: string) => {
    setFeed((f) => f.filter((x) => x.id !== feedId));
  }, []);

  /* ---------------- the grounded agent ----------------
     Every branch answers only from live state (experiments, leakage
     report, champion, columns). Unknown intent → it says what it can
     and cannot do instead of inventing an answer. */
  const askAgent = useCallback(
    async (prompt: string) => {
      const p = prompt.trim();
      if (!p) return;
      const lower = p.toLowerCase();
      const thinkId = `th${++uid.current}`;
      setAgentStatus('thinking');
      pushFeed({ kind: 'text', from: 'user', text: p });
      pushFeed({ id: thinkId, kind: 'thinking' });
      if (!backend.connection.demo) {
        // Real grounded Q&A over the recorded experiment history.
        let text: string;
        try {
          text = await backend.chatAsk(p);
        } catch (err) {
          text = `Chat failed: ${backend.describeError(err)}`;
        }
        setFeed((f) => f.filter((x) => x.id !== thinkId));
        pushFeed({ kind: 'text', from: 'agent', text, stream: true });
        setAgentStatus('idle');
        return;
      }
      await sleep(550);
      setFeed((f) => f.filter((x) => x.id !== thinkId));

      const list = expRef.current;
      const mk = metricRef.current;
      const ch = pickChampion(list, championRef.current, mk);
      const base = list.find((e) => e.decision === 'baseline') ?? list.find((e) => !e.parent_id);

      const say = (text: string, stream = true) => pushFeed({ kind: 'text', from: 'agent', text, stream });

      // 1 · grounded: why a run was rejected / what happened to a feature
      if (/\bwhy\b|\breject/.test(lower)) {
        const rejected = list.filter((e) => e.decision === 'reject');
        if (!rejected.length) {
          say('No rejected runs in this lineage yet — every completed run is kept or still pending a decision. Ask me again once a run lands, or point me at a feature: "why did days_since_active move F1?".');
          setAgentStatus('idle');
          return;
        }
        const lines = rejected.map((e) => {
          if (e.status === 'failed' && e.error) return `${e.id} (${e.title ?? algoLabel(e.model_name)}) — the run itself failed: ${e.error}`;
          const parent = e.parent_id ? list.find((x) => x.id === e.parent_id) : undefined;
          const pm = parent?.metrics[mk];
          if (e.metrics[mk] !== undefined && pm !== undefined) {
            return `${e.id} (+${e.feature ?? 'fork'}) — landed at ${mk} ${e.metrics[mk]!.toFixed(3)} vs ${pm.toFixed(3)} on parent ${parent?.id}. Below the keep line, so I rejected it.`;
          }
          return `${e.id} — rejected during the auto-clean stage (see the leakage report).`;
        });
        say(lines.join('\n\n'));
        setAgentStatus('idle');
        return;
      }

      // 2 · grounded: current champion
      if (/\b(champion|best)\b/.test(lower)) {
        if (ch && ch.metrics[mk] !== undefined) {
          const depth = (() => {
            let n = 0;
            let c: Experiment | undefined = ch;
            while (c) {
              n++;
              c = c.parent_id ? list.find((e) => e.id === c!.parent_id) : undefined;
            }
            return n;
          })();
          const delta = base?.metrics[mk] !== undefined ? (ch.metrics[mk]! - base.metrics[mk]!).toFixed(3) : null;
          say(
            `Current best is ${ch.id} · ${algoLabel(ch.model_name)} at ${mk} ${ch.metrics[mk]!.toFixed(3)} (acc ${f3(ch.metrics.accuracy)}). ${depth}-node lineage from ${base?.id ?? 'the root'}${delta ? `, +${delta} vs baseline on ${mk}` : ''}. Full explanation in its Importance tab, or type "debrief".`
          );
        } else {
          say('No champion yet — no completed runs in the workspace. Queue a hypothesis and I will name a champion the moment one finishes.');
        }
        setAgentStatus('idle');
        return;
      }

      // 3 · grounded: live status
      if (/status|running|progress|what'?s (going|happening)|queue/.test(lower)) {
        const run = list.filter((e) => e.status === 'running');
        const queued = list.filter((e) => e.status === 'queued').length;
        const done = list.filter((e) => e.status === 'completed').length;
        const failed = list.filter((e) => e.status === 'failed').length;
        const parts = run.map((e) => `${e.id} at ${Math.round(e.progress ?? 0)}% (${e.runtime_seconds.toFixed(1)}s in)`);
        say(
          parts.length
            ? `In flight: ${parts.join('; ')}.${queued ? ` ${queued} queued behind them.` : ''}`
            : `Queue is clear — ${done} completed, ${failed} failed, nothing in flight.`
        );
        setAgentStatus('idle');
        return;
      }

      // 4 · grounded: leakage report
      if (/leak/.test(lower)) {
        const w = warningsRef.current;
        if (!w.length) {
          say('The leakage report is empty for this dataset — no flagged columns on file. If you expected findings, run the Auto-Clean scan first.');
        } else {
          say(
            `${w.length} findings on file: ${w
              .map((x) => `${x.column} — ${x.category}, ${x.severity}${excludedRef.current.has(x.column) ? ', excluded' : ', still in the matrix'}`)
              .join('; ')}. I categorize by mechanism, not vibes — ask about any column for the evidence.`
          );
        }
        setAgentStatus('idle');
        return;
      }

      // 5 · grounded: counts
      if (/how many|count|number of/.test(lower)) {
        say(
          `${list.length} total runs: ${list.filter((e) => e.status === 'completed').length} completed, ${list.filter((e) => e.status === 'running').length} running, ${list.filter((e) => e.status === 'queued').length} queued, ${list.filter((e) => e.status === 'failed').length} failed. ${list.filter((e) => e.feature).length} feature hypotheses tested since the baseline.`
        );
        setAgentStatus('idle');
        return;
      }

      // 6 · grounded debrief
      if (/debrief|summar|explain (the |my )?model|what does it use/.test(lower)) {
        say(debrief());
        setAgentStatus('idle');
        return;
      }

      // 7 · natural-language configuration
      const configParts: string[] = [];
      if (/\bonly\b.*tree|tree-?based|no logistic|skip logistic/.test(lower)) {
        setTreeOnly(true);
        treeOnlyRef.current = true;
        configParts.push('tree-based models only — Logistic Reg. is out of the rotation');
      }
      if (/\brecall\b/.test(lower)) {
        setOptimizeFor('recall');
        metricRef.current = 'recall';
        configParts.push('optimizing for recall — keep/reject calls and the champion now rank by it');
      } else if (/\baccuracy\b/.test(lower)) {
        setOptimizeFor('accuracy');
        metricRef.current = 'accuracy';
        configParts.push('optimizing for accuracy');
      } else if (/\bf1\b|f-?score/.test(lower)) {
        setOptimizeFor('f1');
        metricRef.current = 'f1';
        configParts.push('optimizing for F1');
      }
      const stop = lower.match(/stop after (\d+)\s*(min|minute)/);
      if (stop) {
        const n = Number(stop[1]);
        setBudgetMin(n);
        configParts.push(`autopilot will stop after ${n} min`);
      }
      if (configParts.length) {
        say(`Applied: ${configParts.join('; ')}. The chip above shows the live configuration; every future run uses it.`);
        log(`Config: ${configParts.join(', ')}`, 'info');
        setAgentStatus('idle');
        return;
      }

      // 8 · steering → a real queued experiment, not a logged note
      const used = [
        ...feedRef.current.flatMap((x) => (x.kind === 'suggestion' ? [x.s.name] : [])),
        ...expRef.current.flatMap((e) => (e.feature ? [e.feature] : [])),
      ];
      // Demo mode only: real chat returns above via /chat/ask.
      const reply = craftReply(lower, used);
      if (!reply.suggestion) {
        say(reply.text);
        setAgentStatus('idle');
        return;
      }
      const fid = pushFeed({ kind: 'suggestion', s: reply.suggestion });
      say(reply.text);
      setTimeout(() => {
        runSuggestion(fid, pickModelKey());
        pushFeed({
          kind: 'text',
          from: 'agent',
          text: `Steering accepted. That is queued as a real run, branching from ${pickChampion(expRef.current, championRef.current, metricRef.current)?.id ?? 'the baseline'} — you will see it move in the lineage and I will report the result here.`,
        });
      }, 650);
      setAgentStatus('idle');
    },
    [pushFeed, debrief, runSuggestion, pickModelKey, log]
  );

  const fork = useCallback(
    (id: string, modelKey: string) => {
      const src = expRef.current.find((e) => e.id === id);
      if (!src) return;
      const m = MODELS.find((x) => x.key === modelKey) ?? MODELS[0];
      const exp = enqueue({
        parent_id: id,
        model_name: m.cls,
        title: `${m.label} fork`,
        params: defaultParams(m.cls),
      });
      log(`${exp.id} forked from ${id} · ${m.label}`, 'info');
      toast(`Forked ${id} → ${exp.id}`, 'info');
      setSelectedId(exp.id);
    },
    [enqueue, log, toast]
  );

  const patch = useCallback(
    (id: string, fn: (e: Experiment) => Experiment) => commit(expRef.current.map((e) => (e.id === id ? fn(e) : e))),
    [commit]
  );

  const retry = useCallback(
    (id: string) => {
      patch(id, (e) => ({ ...e, status: 'queued', progress: 0, runtime_seconds: 0, error: undefined, decision: 'none' }));
      log(`${id} re-queued`, 'info');
      toast(`Re-queued ${id}`, 'info');
    },
    [patch, log, toast]
  );

  const cancel = useCallback(
    (id: string) => {
      patch(id, (e) => ({ ...e, status: 'failed', error: 'Cancelled by user', decision: 'reject' }));
      log(`${id} cancelled`, 'warn');
      toast(`Cancelled ${id}`, 'warn');
    },
    [patch, log, toast]
  );

  const promote = useCallback(
    (id: string) => {
      championRef.current = id;
      setChampionId(id);
      patch(id, (e) => ({ ...e, decision: 'keep' }));
      log(`${id} promoted to champion`, 'ok');
      toast(`${id} is now the champion`, 'ok');
    },
    [patch, log, toast]
  );

  const remove = useCallback(
    (id: string) => {
      if (expRef.current.some((e) => e.parent_id === id)) return toast('Delete child runs first.', 'warn');
      commit(expRef.current.filter((e) => e.id !== id));
      setSelectedId((s) => (s === id ? null : s));
      log(`${id} deleted`, 'warn');
      toast(`Deleted ${id}`, 'warn');
    },
    [commit, log, toast]
  );

  const setDecision = useCallback(
    (id: string, d: Decision) => {
      if (d === 'baseline') {
        commit(expRef.current.map((e) => ({ ...e, decision: e.id === id ? 'baseline' : e.decision === 'baseline' ? 'keep' : e.decision })));
      } else {
        patch(id, (e) => ({ ...e, decision: d }));
      }
      log(`${id} marked ${d}`, d === 'reject' ? 'warn' : 'ok');
    },
    [commit, patch, log]
  );

  const deploy = useCallback(
    (id: string) => {
      const exp = expRef.current.find((e) => e.id === id);
      if (!exp || exp.status !== 'completed') return toast('Only completed runs can be deployed', 'warn');
      const ep: Endpoint = {
        id: `ep_${exp.id}`,
        experiment_id: exp.id,
        model_name: exp.model_name,
        url: `https://api.mlpilot.dev/v1/predict/${datasetName}`,
        status: 'warming',
        rpm: 0,
        p95_ms: 0,
        created_at: new Date().toISOString(),
      };
      setEndpoints((list) => [ep, ...list.filter((x) => x.experiment_id !== id)]);
      setView('deployments');
      log(`Deploying ${id} → ${ep.url}`, 'info');
      toast(`Deploying ${id}`, 'ok');
      setTimeout(() => {
        setEndpoints((list) => list.map((x) => (x.id === ep.id ? { ...x, status: 'healthy', rpm: 120, p95_ms: 38 } : x)));
        toast(`${id} is live`, 'ok');
      }, 1600);
    },
    [datasetName, log, toast]
  );

  const retireEndpoint = useCallback(
    (id: string) => {
      setEndpoints((e) => e.filter((x) => x.id !== id));
      toast('Endpoint retired', 'warn');
    },
    [toast]
  );

  /* ---------------- derived ---------------- */
  const champion = pickChampion(experiments, championId, optimizeFor);
  const baseline = experiments.find((e) => e.decision === 'baseline') ?? experiments.find((e) => !e.parent_id) ?? null;

  const readiness = useMemo(() => {
    if (!metrics) return 0;
    let s = 100;
    warnings.forEach((w) => {
      if (excluded.has(w.column)) return;
      s -= w.severity === 'high' ? 18 : w.severity === 'medium' ? 8 : 3;
    });
    s -= metrics.missing_data_percent * 100 * 2;
    s -= Math.min(6, (metrics.duplicate_rows / Math.max(1, metrics.total_rows)) * 100 * 20);
    if (cleaned) s = Math.max(s, 92);
    return Math.round(clamp(s, 0, 100));
  }, [metrics, warnings, excluded, cleaned]);

  const value: Store = {
    loading,
    live,
    demo,
    connectionError,
    started,
    startApp,
    metrics,
    warnings,
    columns,
    excluded,
    experiments,
    feed,
    activity,
    toasts,
    champion,
    baseline,
    readiness,
    selectedId,
    view,
    agentOpen,
    paletteOpen,
    uploadOpen,
    datasetName,
    fileName,
    target,
    cleaning,
    cleaned,
    cleanSteps,
    hypotheses,
    agentStatus,
    compareBaseline,
    optimizeFor,
    treeOnly,
    budgetMin,
    endpoints,
    provider,
    apiKey,
    providerModel,
    drift,
    volume,
    setView,
    setAgentOpen,
    setPaletteOpen,
    setUploadOpen,
    setTarget,
    setHypotheses,
    setCompareBaseline,
    setProvider,
    setApiKey,
    setProviderModel,
    select,
    toggleColumn,
    excludeHighRisk,
    ingestCsv,
    autoClean,
    answerCheckpoint,
    runSuggestion,
    runNext,
    runAutopilot,
    dismissSuggestion,
    askAgent,
    fork,
    retry,
    cancel,
    promote,
    remove,
    setDecision,
    deploy,
    retireEndpoint,
    toast,
    dismissToast,
  };

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
