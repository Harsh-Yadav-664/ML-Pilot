import { ColumnProfile, Experiment, FeatureWeight } from './types';

export const MODELS = [
  { key: 'xgboost', label: 'XGBoost', cls: 'XGBClassifier', tree: true },
  { key: 'lightgbm', label: 'LightGBM', cls: 'LGBMClassifier', tree: true },
  { key: 'catboost', label: 'CatBoost', cls: 'CatBoostClassifier', tree: true },
  { key: 'rf', label: 'Random Forest', cls: 'RandomForestClassifier', tree: true },
  { key: 'logistic', label: 'Logistic Reg.', cls: 'LogisticRegression', tree: false },
] as const;

export const clamp = (n: number, a: number, b: number) => Math.min(b, Math.max(a, n));
export const rand = (a: number, b: number) => a + Math.random() * (b - a);

export function algoLabel(name: string): string {
  if (/xgb/i.test(name)) return 'XGBoost';
  if (/lgbm|lightgbm/i.test(name)) return 'LightGBM';
  if (/cat/i.test(name)) return 'CatBoost';
  if (/forest/i.test(name)) return 'Random Forest';
  if (/logistic/i.test(name)) return 'Logistic Reg.';
  return name;
}

export function defaultParams(cls: string): Record<string, string | number> {
  if (/xgb/i.test(cls)) return { n_estimators: 400, max_depth: 6, learning_rate: 0.08, subsample: 0.85 };
  if (/lgbm/i.test(cls)) return { n_estimators: 600, num_leaves: 63, learning_rate: 0.05, colsample_bytree: 0.8 };
  if (/cat/i.test(cls)) return { iterations: 800, depth: 6, learning_rate: 0.06, l2_leaf_reg: 3 };
  if (/forest/i.test(cls)) return { n_estimators: 300, max_depth: 12, min_samples_leaf: 3 };
  return { C: 1.0, penalty: 'l2', solver: 'lbfgs', max_iter: 500 };
}

export function hash(seed: string): () => number {
  let h = 2166136261;
  for (const ch of seed) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return () => {
    h += 0x6d2b79f5;
    let t = Math.imul(h ^ (h >>> 15), 1 | h);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeCurve(seed: string, final: number, n = 28): number[] {
  const rnd = hash(seed);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const base = final - (final - 0.42) * Math.exp(-4.4 * t);
    out.push(base + (rnd() - 0.5) * 0.024 * (1 - t * 0.7));
  }
  out[n - 1] = final;
  return out;
}

export function estLift(name: string): number {
  const rnd = hash(name);
  return Math.round((0.012 + rnd() * 0.038) * 1000) / 1000;
}

export function timeAgo(ts: number | string): string {
  const t = typeof ts === 'number' ? ts : new Date(ts).getTime();
  const s = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (s < 5) return 'just now';
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export const f3 = (n?: number) => (n === undefined ? '—' : n.toFixed(3));
export const pct = (n?: number, d = 1) => (n === undefined ? '—' : `${(n * 100).toFixed(d)}%`);

export function importances(exp: Experiment, cols: ColumnProfile[]): FeatureWeight[] {
  const rnd = hash(exp.id + (exp.feature ?? ''));
  const feats = cols.filter((c) => c.role === 'feature' && !c.flag);
  const extra = exp.feature ? [{ name: exp.feature, boost: 0.22 }] : [];
  const raw = [
    ...extra.map((e) => ({ name: e.name, weight: 0.18 + rnd() * 0.16 + e.boost })),
    ...feats.slice(0, 8).map((c) => ({ name: c.name, weight: 0.04 + rnd() * 0.22 })),
  ];
  const seen = new Set<string>();
  const uniq = raw.filter((r) => (seen.has(r.name) ? false : (seen.add(r.name), true)));
  const sum = uniq.reduce((s, x) => s + x.weight, 0) || 1;
  return uniq
    .map((x) => ({ name: x.name, weight: x.weight / sum }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 8);
}

export function trainingScript(exp: Experiment, target: string, excluded: string[]): string {
  const feats = exp.feature ? `# engineered: ${exp.feature}\n` : '';
  const drops = excluded.length ? `    .drop(columns=${JSON.stringify(excluded)}, errors='ignore')\n` : '';
  return `# MLPilot · ${exp.id} · ${algoLabel(exp.model_name)}
# Lineage parent: ${exp.parent_id ?? 'none'}
${feats}import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

df = pd.read_parquet("churn_v3.parquet")${drops ? "\ndf = (df\n" + drops + ")" : ""}
y = df["${target}"]
X = df.drop(columns=["${target}"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = ${exp.model_name}(${Object.entries(exp.params ?? defaultParams(exp.model_name))
    .map(([k, v]) => `${k}=${typeof v === 'string' ? `'${v}'` : v}`)
    .join(', ')})
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("f1", f1_score(y_test, pred), "acc", accuracy_score(y_test, pred))
`;
}

export function curlSnippet(url: string): string {
  return `curl -X POST ${url} \\
  -H "Authorization: Bearer $MLPILOT_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "tenure_months": 14,
    "monthly_spend": 89.0,
    "last_login": "2026-03-01",
    "support_tickets_90d": 3
  }'`;
}

export function inferDtype(name: string, sample: string): ColumnProfile['dtype'] {
  const n = name.toLowerCase();
  if (/date|time|_ts$|_at$/.test(n)) return 'datetime';
  if (/^is_|flag|churned/.test(n)) return 'bool';
  if (/id$/.test(n)) return 'string';
  if (sample !== '' && !Number.isNaN(Number(sample))) return sample.includes('.') ? 'float' : 'int';
  return 'category';
}

export function parseCsv(text: string): { headers: string[]; rows: number; samples: string[] } {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length);
  const headers = (lines[0] ?? '').split(',').map((h) => h.trim().replace(/^"|"$/g, ''));
  const samples = (lines[1] ?? '').split(',').map((h) => h.trim().replace(/^"|"$/g, ''));
  return { headers: headers.filter(Boolean), rows: Math.max(0, lines.length - 1), samples };
}
