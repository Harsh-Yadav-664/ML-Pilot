import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { describeError } from '../api';
import { ConnectionRead, SchemaGraphData, getSchema, listConnections } from '../api/connections';
import { DbVersion, listDbVersions } from '../api/snapshots';
import { SpecCheck, SpecIssue, TaskSpecBody, TaskSpecRead, checkSpec, confirmTask, saveTaskNamed } from '../api/tasks';
import { LabelPreview } from '../components/LabelPreview';
import { leaveRoute } from '../route';
import { takeEditorSeed, EditorSeed } from '../taskEditorSeed';
import { Btn, CardHeader, Eyebrow, Panel, Tag } from '../ui';
import { cn } from '../utils/cn';

const inputCls = 'h-9 w-full border border-line bg-ink px-2 font-mono text-[12px] text-bone placeholder:text-mute focus:border-copper focus:outline-none';
const AGGS = ['count', 'sum', 'avg', 'min', 'max', 'count_distinct'];
const METRICS: Record<string, string[]> = {
  binary: ['pr_auc', 'roc_auc', 'f1', 'log_loss'],
  regression: ['mae', 'rmse', 'r2'],
  multiclass: ['macro_f1', 'accuracy', 'log_loss'],
};
const DEBOUNCE_MS = 500;

type Exists = { table: string; where?: string };
type Rule = string | { exists?: Exists; not_exists?: Exists };
type Spec = {
  name: string;
  description?: string;
  entity: { table: string; key: string; created_at?: string };
  eligibility: Rule[];
  target: {
    type: string;
    window?: { start?: string; end?: string };
    expression?: { table: string; agg: string; column?: string; where?: string; compare?: string; via?: string };
    expression_sql?: string;
  };
  horizon: string;
  cutoffs: { start: string; end: string; every: string };
  split: { val_from: string; test_from: string };
  metric?: string;
};

const blank = (): Spec => ({
  name: '',
  entity: { table: '', key: '' },
  eligibility: [],
  target: { type: 'binary', expression: { table: '', agg: 'count', compare: '>= 1' } },
  horizon: '30d',
  cutoffs: { start: '', end: '', every: '1 month' },
  split: { val_from: '', test_from: '' },
});

/** Empty text fields mean "not set": the backend gets the field left out, not an empty string. */
function clean(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(clean);
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      if (v === '' || v === undefined || v === null) continue;
      out[k] = clean(v);
    }
    return out;
  }
  return value;
}

function Field({ label, children, span }: { label: string; children: React.ReactNode; span?: boolean }) {
  return (
    <label className={cn('block', span && 'col-span-2')}>
      <Eyebrow>{label}</Eyebrow>
      <div className="mt-1">{children}</div>
    </label>
  );
}

function Select({ label, value, options, onChange, blankLabel }: { label: string; value: string; options: string[]; onChange: (v: string) => void; blankLabel?: string }) {
  return (
    <select aria-label={label} className={inputCls} value={value} onChange={(e) => onChange(e.target.value)}>
      {blankLabel !== undefined && <option value="">{blankLabel}</option>}
      {value && !options.includes(value) && <option value={value}>{value}</option>}
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}

function Text({ label, value, onChange, placeholder, type }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string; type?: string }) {
  return <input aria-label={label} type={type ?? 'text'} className={inputCls} value={value} placeholder={placeholder} onChange={(e) => onChange(e.target.value)} autoComplete="off" spellCheck={false} />;
}

function Problems({ issues, prefixes }: { issues: SpecIssue[]; prefixes: string[] }) {
  const mine = issues.filter((i) => prefixes.some((p) => i.path === p || i.path.startsWith(`${p}.`) || i.path.startsWith(`${p}[`)));
  if (!mine.length) return null;
  return (
    <ul className="mt-2 space-y-1" data-testid="field-problems">
      {mine.map((i) => (
        <li key={`${i.path}:${i.message}`} className={cn('text-[12px]', i.severity === 'error' ? 'text-clay' : 'text-copper-2')}>
          {i.path}: {i.message}
        </li>
      ))}
    </ul>
  );
}

/**
 * Build or fix a prediction task by hand: a form over the spec's fields, a YAML tab, and on the right
 * what the spec means: its problems, the label counts per cutoff, the label SQL and the feasibility
 * checks. Every change is checked again (debounced) through POST /tasks/check, which saves nothing.
 * Confirm saves the spec as a version and locks it.
 */
export function TaskEditor() {
  const [seed] = useState<EditorSeed | null>(() => takeEditorSeed());
  const [connections, setConnections] = useState<ConnectionRead[]>([]);
  const [connectionId, setConnectionId] = useState(seed?.connectionId ?? '');
  const [versions, setVersions] = useState<DbVersion[]>([]);
  const [versionId, setVersionId] = useState(seed?.dataVersionId ?? '');
  const [graph, setGraph] = useState<SchemaGraphData | null>(null);
  const [spec, setSpec] = useState<Spec>(blank);
  const [yamlText, setYamlText] = useState(seed?.yaml ?? '');
  const [tab, setTab] = useState<'form' | 'yaml'>('form');
  const [edit, setEdit] = useState<{ source: 'form' | 'yaml'; n: number }>({ source: seed ? 'yaml' : 'form', n: seed ? 1 : 0 });
  const [result, setResult] = useState<SpecCheck | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [taskId, setTaskId] = useState<string | null>(seed?.taskId ?? null);
  const [confirmed, setConfirmed] = useState<TaskSpecRead | null>(null);
  const [busy, setBusy] = useState(false);
  const latest = useRef({ spec, yamlText });
  latest.current = { spec, yamlText };

  useEffect(() => {
    listConnections()
      .then((all) => {
        setConnections(all);
        setConnectionId((current) => current || all[0]?.id || '');
      })
      .catch((e) => setError(describeError(e)));
  }, []);

  useEffect(() => {
    if (!connectionId) return;
    setGraph(null);
    getSchema(connectionId).then(setGraph).catch((e) => setError(describeError(e)));
    listDbVersions()
      .then((all) => setVersions(all.filter((v) => v.connection.id === connectionId)))
      .catch(() => setVersions([]));
  }, [connectionId]);

  // Check on every change, after a pause; a newer change cancels the check still running.
  useEffect(() => {
    if (!connectionId || edit.n === 0) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      setChecking(true);
      setError(null);
      try {
        const body = edit.source === 'form' ? { spec: clean(latest.current.spec) as TaskSpecBody } : { yaml: latest.current.yamlText };
        const out = await checkSpec({ ...body, connectionId, dataVersionId: versionId || null }, controller.signal);
        if (controller.signal.aborted) return;
        setResult(out);
        // The other view follows the one being edited, once the spec parses.
        if (edit.source === 'form' && out.yaml) setYamlText(out.yaml);
        if (edit.source === 'yaml' && out.spec) setSpec({ ...blank(), ...(out.spec as unknown as Spec) });
      } catch (e) {
        if (!controller.signal.aborted) setError(describeError(e));
      } finally {
        if (!controller.signal.aborted) setChecking(false);
      }
    }, DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [edit, connectionId, versionId]);

  const change = useCallback((fn: (s: Spec) => void) => {
    setConfirmed(null);
    setSpec((prev) => {
      const next = structuredClone(prev);
      fn(next);
      return next;
    });
    setEdit((e) => ({ source: 'form', n: e.n + 1 }));
  }, []);

  const tables = useMemo(() => graph?.tables ?? [], [graph]);
  const tableNames = tables.map((t) => t.key);
  const columnsOf = (table: string) => tables.find((t) => t.key === table)?.columns.map((c) => c.name) ?? [];
  const timeColumnsOf = (table: string) => tables.find((t) => t.key === table)?.columns.filter((c) => /time|date/i.test(c.type)).map((c) => c.name) ?? [];
  const linked = (entity: string) => tableNames.filter((t) => graph?.edges.some((e) => e.from_table === t && e.to_table === entity));

  const issues = result?.issues ?? [];
  const errors = issues.filter((i) => i.severity === 'error');
  const preview = result?.preview ?? null;
  const ready = !!result?.yaml && errors.length === 0 && !!preview && !preview.feasibility.blocked && !checking;
  const ex = spec.target.expression;

  async function confirm() {
    if (!result?.yaml) return;
    setBusy(true);
    setError(null);
    try {
      const saved = await saveTaskNamed({ yaml: result.yaml, connectionId, dataVersionId: versionId || null, draftSource: seed?.draftSource ?? null }, String(result.spec?.name ?? ''), taskId);
      setTaskId(saved.id);
      setConfirmed(await confirmTask(saved.id, versionId || null));
    } catch (e) {
      setError(describeError(e));
    } finally {
      setBusy(false);
    }
  }

  const metricOptions = METRICS[spec.target.type] ?? [];

  return (
    <div className="relative min-h-screen overflow-y-auto text-bone">
      <div className="guides" aria-hidden />
      <header className="mx-auto flex h-14 max-w-[1400px] items-center justify-between px-6">
        <Btn variant="ghost" size="sm" onClick={leaveRoute}>
          ← Back
        </Btn>
        <Tag tone="copper">nothing trains here</Tag>
      </header>
      <main className="mx-auto max-w-[1400px] space-y-6 px-6 pb-16">
        <section>
          <Eyebrow className="text-copper">Prediction task</Eyebrow>
          <h1 className="mt-1 font-display text-[40px] font-extrabold uppercase leading-none tracking-wide text-bone">Task editor</h1>
          <p className="mt-3 max-w-2xl text-[14px] leading-relaxed text-mute">
            Define what to predict without a language model, or fix a drafted task. The right side shows what the spec means on your data and updates as you type.
          </p>
        </section>

        <div className="flex flex-wrap items-end gap-3">
          <Field label="Database">
            <select aria-label="Connection" className={cn(inputCls, 'w-64')} value={connectionId} onChange={(e) => setConnectionId(e.target.value)}>
              {connections.length === 0 && <option value="">No connection saved yet</option>}
              {connections.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} ({c.dialect})
                </option>
              ))}
            </select>
          </Field>
          <Field label="Data version">
            <select aria-label="Data version" className={cn(inputCls, 'w-64')} value={versionId} onChange={(e) => setVersionId(e.target.value)}>
              <option value="">Live database, as of now</option>
              {versions.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.mode} · as of {v.as_of.slice(0, 10)}
                </option>
              ))}
            </select>
          </Field>
          <div className="flex gap-1" role="tablist" aria-label="Editor view">
            {(['form', 'yaml'] as const).map((t) => (
              <button
                key={t}
                role="tab"
                aria-selected={tab === t}
                onClick={() => setTab(t)}
                className={cn('h-9 border px-4 font-mono text-[11px] uppercase tracking-[0.1em]', tab === t ? 'border-copper bg-copper/10 text-copper-2' : 'border-line text-mute hover:text-bone')}
              >
                {t === 'form' ? 'Form' : 'YAML'}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div role="alert" data-testid="editor-error" className="border border-clay/50 bg-clay/5 p-4 text-[13px] text-bone">
            {error}
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-[minmax(0,560px)_minmax(0,1fr)]">
          <div className="space-y-4" data-testid="task-editor">
            {tab === 'yaml' ? (
              <Panel ticks>
                <CardHeader title="Task spec (YAML)" sub="Checked as you type; the form follows once it parses" />
                <div className="p-4">
                  <textarea
                    aria-label="Task spec YAML"
                    className="h-[520px] w-full border border-line bg-ink p-3 font-mono text-[12px] text-bone focus:border-copper focus:outline-none"
                    value={yamlText}
                    spellCheck={false}
                    onChange={(e) => {
                      setConfirmed(null);
                      setYamlText(e.target.value);
                      setEdit((s) => ({ source: 'yaml', n: s.n + 1 }));
                    }}
                  />
                </div>
              </Panel>
            ) : (
              <>
                <Panel ticks>
                  <CardHeader title="Who and when" />
                  <div className="grid grid-cols-2 gap-3 p-4">
                    <Field label="Name" span>
                      <Text label="Task name" value={spec.name} onChange={(v) => change((s) => void (s.name = v))} placeholder="churn_30d" />
                    </Field>
                    <Field label="Entity table">
                      <Select
                        label="Entity table"
                        value={spec.entity.table}
                        options={tableNames}
                        blankLabel="Choose a table"
                        onChange={(v) =>
                          change((s) => {
                            const t = tables.find((x) => x.key === v);
                            s.entity.table = v;
                            s.entity.key = t?.primary_key[0] ?? '';
                            s.entity.created_at = timeColumnsOf(v)[0] ?? '';
                          })
                        }
                      />
                    </Field>
                    <Field label="Entity key">
                      <Select label="Entity key" value={spec.entity.key} options={columnsOf(spec.entity.table)} blankLabel="Choose a column" onChange={(v) => change((s) => void (s.entity.key = v))} />
                    </Field>
                    <Field label="Exists from (time column)">
                      <Select label="Entity created at" value={spec.entity.created_at ?? ''} options={timeColumnsOf(spec.entity.table)} blankLabel="Not set" onChange={(v) => change((s) => void (s.entity.created_at = v))} />
                    </Field>
                    <Field label="Horizon">
                      <Text label="Horizon" value={spec.horizon} onChange={(v) => change((s) => void (s.horizon = v))} placeholder="30d" />
                    </Field>
                    <Field label="First cutoff">
                      <Text label="Cutoffs start" type="date" value={spec.cutoffs.start} onChange={(v) => change((s) => void (s.cutoffs.start = v))} />
                    </Field>
                    <Field label="Last cutoff">
                      <Text label="Cutoffs end" type="date" value={spec.cutoffs.end} onChange={(v) => change((s) => void (s.cutoffs.end = v))} />
                    </Field>
                    <Field label="Every">
                      <Text label="Cutoffs every" value={spec.cutoffs.every} onChange={(v) => change((s) => void (s.cutoffs.every = v))} placeholder="1 month" />
                    </Field>
                    <Field label="Validation from">
                      <Text label="Validation from" type="date" value={spec.split.val_from} onChange={(v) => change((s) => void (s.split.val_from = v))} />
                    </Field>
                    <Field label="Test from">
                      <Text label="Test from" type="date" value={spec.split.test_from} onChange={(v) => change((s) => void (s.split.test_from = v))} />
                    </Field>
                  </div>
                  <div className="px-4 pb-4">
                    <Problems issues={issues} prefixes={['name', 'entity', 'horizon', 'cutoffs', 'split']} />
                  </div>
                </Panel>

                <Panel ticks>
                  <CardHeader title="What to predict" sub="The label is read only from the window after each cutoff" />
                  <div className="grid grid-cols-2 gap-3 p-4">
                    <Field label="Kind">
                      <Select
                        label="Task type"
                        value={spec.target.type}
                        options={['binary', 'regression']}
                        onChange={(v) =>
                          change((s) => {
                            s.target.type = v;
                            if (!s.target.expression) s.target.expression = { table: '', agg: v === 'binary' ? 'count' : 'sum' };
                            if (v === 'regression') s.target.expression.compare = '';
                            s.metric = '';
                          })
                        }
                      />
                    </Field>
                    <Field label="Judged by">
                      <Select label="Metric" value={spec.metric ?? ''} options={metricOptions} blankLabel="Default" onChange={(v) => change((s) => void (s.metric = v))} />
                    </Field>
                    <Field label="Table the label reads">
                      <Select
                        label="Target table"
                        value={ex?.table ?? ''}
                        options={linked(spec.entity.table)}
                        blankLabel="Choose a table"
                        onChange={(v) => change((s) => void (s.target.expression = { ...(s.target.expression ?? { agg: 'count' }), table: v, column: '' }))}
                      />
                    </Field>
                    <Field label="Aggregate">
                      <Select label="Aggregate" value={ex?.agg ?? 'count'} options={AGGS} onChange={(v) => change((s) => void (s.target.expression = { ...(s.target.expression ?? { table: '' }), agg: v }))} />
                    </Field>
                    <Field label="Column (not needed for count)">
                      <Select label="Target column" value={ex?.column ?? ''} options={columnsOf(ex?.table ?? '')} blankLabel="None" onChange={(v) => change((s) => void (s.target.expression = { ...(s.target.expression ?? { table: '', agg: 'count' }), column: v }))} />
                    </Field>
                    <Field label="Only rows where">
                      <Text label="Target filter" value={ex?.where ?? ''} placeholder="status != 'cancelled'" onChange={(v) => change((s) => void (s.target.expression = { ...(s.target.expression ?? { table: '', agg: 'count' }), where: v }))} />
                    </Field>
                    {spec.target.type === 'binary' && (
                      <Field label="Label is 1 when the aggregate is" span>
                        <Text label="Label is 1 when" value={ex?.compare ?? ''} placeholder="= 0" onChange={(v) => change((s) => void (s.target.expression = { ...(s.target.expression ?? { table: '', agg: 'count' }), compare: v }))} />
                      </Field>
                    )}
                  </div>
                  {spec.target.window && (
                    <p className="px-4 pb-3 text-[12px] text-copper-2" data-testid="window-note">
                      This spec sets its own label window ({spec.target.window.start ?? ':cutoff'} to {spec.target.window.end ?? 'the horizon'}). The form keeps it as it is; change it in the YAML tab.
                    </p>
                  )}
                  <div className="px-4 pb-4">
                    <Problems issues={issues} prefixes={['target', 'metric']} />
                  </div>
                </Panel>

                <Panel ticks>
                  <CardHeader title="Who is scored at each cutoff" sub="Rules read only data from before the cutoff" />
                  <div className="space-y-3 p-4">
                    {spec.eligibility.length === 0 && <p className="text-[12px] text-mute">No rules: every entity that exists at the cutoff is scored.</p>}
                    {spec.eligibility.map((rule, i) => {
                      const kind = typeof rule === 'string' ? 'condition' : rule.exists ? 'exists' : 'not_exists';
                      const test = typeof rule === 'string' ? undefined : (rule.exists ?? rule.not_exists);
                      return (
                        <div key={i} className="grid grid-cols-[150px_1fr_auto] gap-2" data-testid={`rule-${i}`}>
                          <Select
                            label={`Rule ${i + 1} kind`}
                            value={kind}
                            options={['condition', 'exists', 'not_exists']}
                            onChange={(v) =>
                              change((s) => {
                                s.eligibility[i] = v === 'condition' ? '' : { [v]: { table: '', where: '' } };
                              })
                            }
                          />
                          {kind === 'condition' ? (
                            <Text label={`Rule ${i + 1} condition`} value={rule as string} placeholder="signup_at < :cutoff" onChange={(v) => change((s) => void (s.eligibility[i] = v))} />
                          ) : (
                            <div className="grid grid-cols-2 gap-2">
                              <Select
                                label={`Rule ${i + 1} table`}
                                value={test?.table ?? ''}
                                options={linked(spec.entity.table)}
                                blankLabel="Table"
                                onChange={(v) => change((s) => void (s.eligibility[i] = { [kind]: { table: v, where: test?.where ?? '' } }))}
                              />
                              <Text
                                label={`Rule ${i + 1} where`}
                                value={test?.where ?? ''}
                                placeholder="ordered_at < :cutoff"
                                onChange={(v) => change((s) => void (s.eligibility[i] = { [kind]: { table: test?.table ?? '', where: v } }))}
                              />
                            </div>
                          )}
                          <Btn size="sm" variant="danger" aria-label={`Remove rule ${i + 1}`} onClick={() => change((s) => void s.eligibility.splice(i, 1))}>
                            ×
                          </Btn>
                        </div>
                      );
                    })}
                    <Btn size="sm" onClick={() => change((s) => void s.eligibility.push(''))}>
                      Add a rule
                    </Btn>
                    <Problems issues={issues} prefixes={['eligibility']} />
                  </div>
                </Panel>
              </>
            )}
          </div>

          <div className="min-w-0 space-y-4">
            <Panel ticks>
              <CardHeader title="What this means" sub={checking ? 'Checking…' : result ? 'Checked just now' : 'Fill in the spec to see its labels'} />
              <div className="space-y-3 p-4" data-testid="editor-preview">
                {errors.length > 0 && (
                  <div data-testid="editor-errors" className="border border-clay/50 bg-clay/5 p-3 text-[13px] text-bone">
                    <div className="font-mono text-[10px] uppercase tracking-[0.1em] text-mute">{errors.length} problem{errors.length === 1 ? '' : 's'} to fix before the labels can be built</div>
                    <ul className="mt-1 list-disc pl-5">
                      {errors.map((i) => (
                        <li key={`${i.path}:${i.message}`}>
                          {i.path ? `${i.path}: ` : ''}
                          {i.message}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {result?.preview_error && <div className="border border-clay/50 bg-clay/5 p-3 text-[13px] text-bone">The labels could not be built: {result.preview_error}</div>}
                {preview && <LabelPreview preview={preview} />}
                {!result && !checking && <p className="text-[12px] text-mute">Nothing checked yet.</p>}
              </div>
            </Panel>

            {confirmed ? (
              <Panel ticks>
                <CardHeader title="Confirmed" sub={`${confirmed.name}, version ${confirmed.version}, by ${confirmed.confirmed_by}`} />
                <div className="p-4" data-testid="task-confirmed">
                  <p className="text-[12px] text-sage">This version is locked. Nothing has been trained: running it comes with the training step.</p>
                  <pre data-testid="task-stored-yaml" className="mt-2 overflow-x-auto border border-line p-3 font-mono text-[11px] text-bone-dim">
                    {confirmed.yaml}
                  </pre>
                </div>
              </Panel>
            ) : (
              <div className="flex flex-wrap items-center gap-3">
                <Btn onClick={() => void confirm()} disabled={!ready || busy}>
                  {busy ? 'Saving…' : 'Confirm this task'}
                </Btn>
                {!ready && <span className="text-[12px] text-mute">{errors.length ? 'Fix the problems first.' : preview?.feasibility.blocked ? 'Blocked by the feasibility checks.' : 'Waiting for a checked spec with labels.'}</span>}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
