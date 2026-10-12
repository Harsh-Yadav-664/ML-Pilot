import {
  ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';
import * as backend from './api';
import type { ActiveDataset } from './api';
import { craftReply, mockDrift, mockEndpoints, mockVolume } from './api/mock';
import {
  AgentStatus,
  ColumnProfile,
  DataMetrics,
  Endpoint,
  Experiment,
  FeatureSuggestion,
  LeakageWarning,
  View,
} from './types';
import { MODELS, algoLabel, clamp, defaultParams, f3, importances, inferDtype, parseCsv, rand } from './lib';

/** Demo mode only: how many simulated runs train at once. */
const DEMO_WORKERS = 2;
type Tone = 'ok' | 'warn' | 'err' | 'info';
type MetricKey = 'f1' | 'recall' | 'accuracy';
/** The metric the workspace ranks runs by (the backend's champion rule uses F1 too). */
const METRIC: MetricKey = 'f1';

interface Toast {
  id: number;
  text: string;
  tone: Tone;
  action?: { label: string; run: () => void };
}
export type FeedItem =
  | { id: string; kind: 'text'; from: 'agent' | 'user'; text: string; stream?: boolean }
  | { id: string; kind: 'suggestion'; s: FeatureSuggestion; expId?: string }
  | { id: string; kind: 'thinking' };

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
  experiments: Experiment[];
  feed: FeedItem[];
  toasts: Toast[];
  champion: Experiment | null;
  baseline: Experiment | null;
  selectedId: string | null;
  view: View;
  agentOpen: boolean;
  paletteOpen: boolean;
  uploadOpen: boolean;
  datasetName: string;
  fileName: string | null;
  target: string;
  cleaning: boolean;
  hypotheses: number;
  agentStatus: AgentStatus;
  compareBaseline: boolean;
  endpoints: Endpoint[];
  drift: { t: number; psi: number }[];
  volume: { h: number; n: number }[];
  setView: (v: View) => void;
  setAgentOpen: (b: boolean) => void;
  setPaletteOpen: (b: boolean) => void;
  setUploadOpen: (b: boolean) => void;
  setTarget: (t: string) => void;
  setHypotheses: (n: number) => void;
  setCompareBaseline: (b: boolean) => void;
  select: (id: string | null) => void;
  ingestCsv: (file: File) => Promise<void>;
  autoClean: () => void;
  runSuggestion: (feedId: string, modelKey?: string) => void;
  runNext: () => void;
  runAutopilot: () => void;
  dismissSuggestion: (feedId: string) => void;
  askAgent: (prompt: string) => Promise<void>;
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
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [feed, setFeed] = useState<FeedItem[]>([]);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [championId, setChampionId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [view, setView] = useState<View>('dashboard');
  // Below the xl breakpoint the panel covers the page, so it starts closed there.
  const [agentOpen, setAgentOpen] = useState(() => window.innerWidth >= 1280);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [datasetName, setDatasetName] = useState(backend.connection.demo ? 'churn_v3' : 'no dataset');
  const [fileName, setFileName] = useState<string | null>(backend.connection.demo ? 'churn_customers.csv' : null);
  const [target, setTargetState] = useState(backend.connection.demo ? 'is_churned' : '—');
  const [cleaning, setCleaning] = useState(false);
  const [hypotheses, setHypotheses] = useState(3);
  const [agentStatus, setAgentStatus] = useState<AgentStatus>('idle');
  const [compareBaseline, setCompareBaseline] = useState(true);
  // Deployments are sample data: shown only in Demo mode.
  const [endpoints, setEndpoints] = useState<Endpoint[]>(demo ? mockEndpoints : []);
  const [drift] = useState(mockDrift);
  const [volume] = useState(mockVolume);

  const expRef = useRef<Experiment[]>([]);
  const feedRef = useRef<FeedItem[]>([]);
  const columnsRef = useRef<ColumnProfile[]>([]);
  const warningsRef = useRef<LeakageWarning[]>([]);
  const championRef = useRef<string | null>(null);
  const midNarrated = useRef<Set<string>>(new Set());
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

  const commit = useCallback((next: Experiment[]) => {
    expRef.current = next;
    setExperiments(next);
  }, []);

  const pushFeed = useCallback((item: DistOmit<FeedItem, 'id'> & { id?: string }) => {
    const full = { ...item, id: item.id ?? `m${++uid.current}` } as FeedItem;
    setFeed((f) => [...f, full]);
    return full.id;
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
          await backend.runBaseline(d);
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
    [commit, toast]
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
      const mk = METRIC;
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
        if (e.status === 'queued' && running < DEMO_WORKERS) {
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
        if (f.status === 'failed') {
          pushFeed({ kind: 'text', from: 'agent', text: `${f.id} failed — ${f.error}. Marking reject.` });
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
          commit(expRef.current.map((e) => (e.id === f.id ? { ...e, decision: 'reject' as const } : e)));
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
  }, [loading, live, demo, commit, pushFeed]);

  const agentStatusRef = useRef<AgentStatus>('idle');
  useEffect(() => {
    agentStatusRef.current = agentStatus;
  }, [agentStatus]);

  /* ---------------- grounded debrief, built only from live state ---------------- */
  const debrief = useCallback(() => {
    const list = expRef.current;
    const ch = pickChampion(list, championRef.current, METRIC);
    const base = list.find((e) => e.decision === 'baseline') ?? list.find((e) => !e.parent_id);
    const mk = METRIC;
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

  const setTarget = useCallback(
    (t: string) => {
      const d = datasetRef.current;
      if (!demo && d) {
        const next = { ...d, target_column: t };
        setDataset(next);
        void loadDataset(next);
        return;
      }
      setTargetState(t);
      setColumns((cols) => cols.map((c) => ({ ...c, role: c.name === t ? 'target' : c.role === 'target' ? 'feature' : c.role })));
    },
    [demo, setDataset, loadDataset]
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
      setUploadOpen(false);
      toast(`Loaded ${file.name}`, 'ok');
    },
    [toast, requireBackend, demo, setDataset, loadDataset]
  );

  /* ---------------- auto-clean: the backend asks the LLM for a cleaning recipe and queues it as a run ---------------- */
  const autoClean = useCallback(() => {
    if (cleaning || !requireBackend()) return;
    if (demo) return toast('Demo mode: auto-clean needs the backend.', 'warn');
    const d = datasetRef.current;
    if (!d) return toast('Load a dataset first.', 'warn');
    setCleaning(true);
    backend
      .runAutoClean(d)
      .then(() => toast('AI cleaning recipe accepted; training a cleaned baseline', 'ok'))
      .catch((err) => toast(`Auto-clean failed: ${backend.describeError(err)}`, 'err'))
      .finally(() => setCleaning(false));
  }, [cleaning, requireBackend, demo, toast]);

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
    // Rotate models; real runs don't go through enqueue, so advance the counter here.
    return available[(backend.connection.demo ? expCounter.current : expCounter.current++) % available.length].key;
  }, []);

  const runSuggestion = useCallback(
    (feedId: string, modelKey?: string) => {
      if (!requireBackend()) return;
      const item = feedRef.current.find((f) => f.id === feedId);
      if (!item || item.kind !== 'suggestion' || item.expId) return;
      const key = modelKey ?? pickModelKey();
      const m = MODELS.find((x) => x.key === key) ?? MODELS[0];
      const parent = pickChampion(expRef.current, championRef.current, METRIC);
      if (!demo) {
        const d = datasetRef.current;
        if (!d) return toast('Load a dataset first.', 'warn');
        if (!BACKEND_MODELS.has(m.cls)) return toast(`${m.label} isn't available in the backend yet.`, 'warn');
        backend
          .runExperiment(d, item.s, m.cls, parent?.id ?? null)
          .then((exp) => {
            setFeed((f) => f.map((x) => (x.id === feedId && x.kind === 'suggestion' ? { ...x, expId: exp.id } : x)));
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
      toast(`Queued ${exp.id}`, 'info', {
        label: 'View',
        run: () => {
          setView('experiments');
          setSelectedId(exp.id);
        },
      });
    },
    [enqueue, pickModelKey, toast, requireBackend, demo]
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
      const done = (text: string) => {
        autoBusy.current = false;
        setAgentStatus('idle');
        pushFeed({ kind: 'text', from: 'agent', text });
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
                  if (e.type === 'proposal' && p.name) pushFeed({ kind: 'text', from: 'agent', text: `Autopilot proposed ${String(p.name)} = ${String(p.formula)}` });
                  if (e.type === 'decision')
                    pushFeed({ kind: 'text', from: 'agent', text: `Autopilot ${p.decision === 'keep' ? 'kept' : 'rejected'} ${String(p.name)}` });
                }
                return backend.getJob(jobId);
              })
              .then((job) => {
                if (job.status === 'queued' || job.status === 'running') return;
                window.clearInterval(poll);
                if (job.status === 'succeeded') done(`Autopilot finished. ${job.result?.summary ?? ''}`);
                else if (job.status === 'cancelled') done('Autopilot was cancelled.');
                else done(`Autopilot failed: ${job.error ?? 'unknown error'}`);
              })
              .catch((err) => {
                window.clearInterval(poll);
                done(`Lost track of the autopilot job: ${backend.describeError(err)}`);
              });
          }, 2000);
        })
        .catch((err) => done(`Autopilot failed to start: ${backend.describeError(err)}`));
      return;
    }
    const pending = feedRef.current.filter((f) => f.kind === 'suggestion' && !f.expId).slice(0, hypotheses);
    if (!pending.length) {
      toast('No unused hypotheses. Steer the agent first.', 'warn');
      return;
    }
    autoBusy.current = true;
    setAgentStatus('autopilot');
    toast(`Autopilot running ${pending.length} experiments`, 'info');
    pending.forEach((p, i) => {
      setTimeout(() => runSuggestion(p.id), 420 * i);
    });
  }, [hypotheses, runSuggestion, toast, pushFeed, requireBackend, demo]);

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
      const mk = METRIC;
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
              .map((x) => `${x.column} — ${x.category}, ${x.severity}`)
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

      // 7 · steering → a real queued experiment, not a logged note
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
          text: `Steering accepted. That is queued as a real run, branching from ${pickChampion(expRef.current, championRef.current, METRIC)?.id ?? 'the baseline'} — you will see it move in the lineage and I will report the result here.`,
        });
      }, 650);
      setAgentStatus('idle');
    },
    [pushFeed, debrief, runSuggestion, pickModelKey]
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
      toast(`Deploying ${id}`, 'ok');
      setTimeout(() => {
        setEndpoints((list) => list.map((x) => (x.id === ep.id ? { ...x, status: 'healthy', rpm: 120, p95_ms: 38 } : x)));
        toast(`${id} is live`, 'ok');
      }, 1600);
    },
    [datasetName, toast]
  );

  const retireEndpoint = useCallback(
    (id: string) => {
      setEndpoints((e) => e.filter((x) => x.id !== id));
      toast('Endpoint retired', 'warn');
    },
    [toast]
  );

  /* ---------------- derived ---------------- */
  const champion = pickChampion(experiments, championId, METRIC);
  const baseline = experiments.find((e) => e.decision === 'baseline') ?? experiments.find((e) => !e.parent_id) ?? null;

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
    experiments,
    feed,
    toasts,
    champion,
    baseline,
    selectedId,
    view,
    agentOpen,
    paletteOpen,
    uploadOpen,
    datasetName,
    fileName,
    target,
    cleaning,
    hypotheses,
    agentStatus,
    compareBaseline,
    endpoints,
    drift,
    volume,
    setView,
    setAgentOpen,
    setPaletteOpen,
    setUploadOpen,
    setTarget,
    setHypotheses,
    setCompareBaseline,
    select,
    ingestCsv,
    autoClean,
    runSuggestion,
    runNext,
    runAutopilot,
    dismissSuggestion,
    askAgent,
    deploy,
    retireEndpoint,
    toast,
    dismissToast,
  };

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
