import { useEffect, useState } from 'react';
import { describeError } from '../api';
import { DbVersion, listDbVersions } from '../api/snapshots';
import { LabelPreview, TaskDraft, TaskSpecRead, confirmTask, draftTask, previewLabels, saveTask } from '../api/tasks';
import { Btn, CardHeader, Panel, Tag } from '../ui';

const inputCls = 'w-full border border-line bg-ink px-3 py-2 font-mono text-[12px] text-bone placeholder:text-mute focus:border-copper focus:outline-none';

const pct = (x: number | null | undefined) => (x == null ? '–' : `${(x * 100).toFixed(1)}%`);
const day = (iso: string) => iso.slice(0, 10);

/**
 * Ask a prediction question in plain words. The backend drafts a task spec (a language model when a key is
 * configured, otherwise the rule-based drafter, and it says which), checks it against the schema and shows it
 * in sentences. Nothing runs until the user presses Confirm; labels are previewed first.
 */
export function TaskCard({ connectionId }: { connectionId: string }) {
  const [question, setQuestion] = useState('');
  const [versions, setVersions] = useState<DbVersion[]>([]);
  const [versionId, setVersionId] = useState<string>('');
  const [draft, setDraft] = useState<TaskDraft | null>(null);
  const [yamlText, setYamlText] = useState('');
  const [editing, setEditing] = useState(false);
  const [task, setTask] = useState<TaskSpecRead | null>(null);
  const [preview, setPreview] = useState<LabelPreview | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listDbVersions()
      .then((all) => setVersions(all.filter((v) => v.connection.id === connectionId)))
      .catch(() => setVersions([]));
  }, [connectionId]);

  const version = versionId || null;
  const reset = () => {
    setDraft(null);
    setTask(null);
    setPreview(null);
    setEditing(false);
    setError(null);
  };

  async function run<T>(what: string, fn: () => Promise<T>): Promise<T | undefined> {
    setBusy(what);
    setError(null);
    try {
      return await fn();
    } catch (e) {
      setError(describeError(e));
      return undefined;
    } finally {
      setBusy(null);
    }
  }

  async function ask() {
    reset();
    const d = await run('Drafting', () => draftTask(question.trim(), connectionId, version));
    if (d) {
      setDraft(d);
      setYamlText(d.yaml ?? '');
    }
  }

  async function check() {
    if (!draft) return;
    const out = await run('Building labels', async () => {
      const saved = await saveTask({ yaml: yamlText, connectionId, dataVersionId: version, draftSource: draft.source }, task?.id ?? null);
      setTask(saved);
      return { saved, preview: await previewLabels(saved.id, version) };
    });
    if (out) {
      setPreview(out.preview);
      setEditing(false);
    }
  }

  async function confirm() {
    if (!task) return;
    const done = await run('Confirming', () => confirmTask(task.id, version));
    if (done) setTask(done);
  }

  const errors = [...(draft?.issues ?? [])].filter((i) => i.severity === 'error');
  const blocked = preview?.feasibility.blocked ?? false;

  return (
    <Panel ticks>
      <CardHeader title="Ask a prediction question" sub="Plain words in, a task you confirm out. Nothing trains yet." />
      <div className="space-y-3 p-4" data-testid="task-card">
        <textarea
          className={inputCls}
          rows={2}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Which customers will stop ordering in the next 30 days?"
          aria-label="Prediction question"
          disabled={busy !== null}
        />
        <div className="flex flex-wrap items-center gap-3">
          <select
            aria-label="Data version"
            value={versionId}
            onChange={(e) => {
              setVersionId(e.target.value);
              reset();
            }}
            className="h-9 border border-line bg-ink px-2 font-mono text-[12px] text-bone"
            disabled={busy !== null}
          >
            <option value="">Live database, as of now</option>
            {versions.map((v) => (
              <option key={v.id} value={v.id}>
                {v.mode} · as of {day(v.as_of)}
              </option>
            ))}
          </select>
          <Btn onClick={() => void ask()} disabled={busy !== null || question.trim() === ''}>
            {busy === 'Drafting' ? 'Drafting…' : 'Draft the task'}
          </Btn>
        </div>

        {error && (
          <div role="alert" data-testid="task-error" className="border border-clay/50 bg-clay/5 p-3 text-[13px] text-bone">
            {error}
          </div>
        )}

        {draft?.status === 'clarify' && (
          <div data-testid="task-clarify" className="border border-copper/40 bg-copper/5 p-3 text-[13px] text-bone">
            {draft.clarifying_question}
          </div>
        )}

        {draft?.status === 'invalid' && (
          <div data-testid="task-invalid" className="border border-clay/50 bg-clay/5 p-3 text-[13px] text-bone">
            The draft still has problems:
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

        {draft?.status === 'spec' && draft.spec && (
          <div className="space-y-3" data-testid="task-draft">
            <div className="flex flex-wrap items-center gap-2">
              <Tag tone={draft.decision_mode === 'llm' ? 'copper' : 'clay'}>{draft.decision_mode === 'llm' ? 'drafted by a language model' : 'fallback: rule-based drafter, no language model answered'}</Tag>
              {draft.repaired && <Tag tone="copper">repaired after a check</Tag>}
              <span className="font-mono text-[10px] text-mute">data ends {day(draft.data_ends)}</span>
            </div>
            <p className="text-[14px] leading-relaxed text-bone" data-testid="task-description">
              {draft.description}
            </p>
            {draft.assumptions.length > 0 && (
              <div>
                <div className="font-mono text-[10px] uppercase tracking-[0.1em] text-mute">Assumptions made</div>
                <ul className="list-disc pl-5 text-[13px] text-bone-dim">
                  {draft.assumptions.map((a) => (
                    <li key={a}>{a}</li>
                  ))}
                </ul>
              </div>
            )}
            {draft.issues.filter((i) => i.severity === 'warning').map((i) => (
              <p key={`${i.path}:${i.message}`} className="text-[12px] text-copper-2">
                Warning{i.path ? ` (${i.path})` : ''}: {i.message}
              </p>
            ))}

            {editing ? (
              <textarea className={`${inputCls} h-64`} value={yamlText} onChange={(e) => setYamlText(e.target.value)} aria-label="Task spec YAML" spellCheck={false} />
            ) : (
              <details>
                <summary className="cursor-pointer font-mono text-[11px] uppercase tracking-[0.1em] text-mute">The spec as YAML</summary>
                <pre className="mt-2 overflow-x-auto border border-line p-3 font-mono text-[11px] text-bone-dim">{yamlText}</pre>
              </details>
            )}

            {preview && (
              <div data-testid="task-preview" className="space-y-2">
                <div className="flex flex-wrap items-center gap-2">
                  <div className="font-mono text-[10px] uppercase tracking-[0.1em] text-mute">
                    Labels: {preview.total_rows.toLocaleString()} rows, {preview.cutoffs.length} cutoffs
                  </div>
                  <Tag tone={preview.feasibility.status === 'block' ? 'clay' : preview.feasibility.status === 'warn' ? 'copper' : 'sage'}>
                    feasibility: {preview.feasibility.status}
                  </Tag>
                </div>
                {preview.feasibility.reasons.length > 0 && (
                  <ul className="list-disc pl-5 text-[12px] text-bone-dim">
                    {preview.feasibility.reasons.map((r) => (
                      <li key={r}>{r}</li>
                    ))}
                  </ul>
                )}
                <div className="max-h-56 overflow-auto border border-line">
                  <table className="w-full font-mono text-[11px]">
                    <thead className="sticky top-0 bg-ink text-left text-mute">
                      <tr>
                        <th className="px-2 py-1">cutoff</th>
                        <th className="px-2 py-1 text-right">eligible</th>
                        <th className="px-2 py-1 text-right">{preview.cutoffs[0]?.mean_label != null ? 'mean label' : 'positives'}</th>
                        <th className="px-2 py-1 text-right">{preview.cutoffs[0]?.mean_label != null ? '' : 'balance'}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {preview.cutoffs.map((c) => (
                        <tr key={c.cutoff} className="border-t border-line text-bone-dim">
                          <td className="px-2 py-1">{day(c.cutoff)}</td>
                          <td className="px-2 py-1 text-right">{c.eligible.toLocaleString()}</td>
                          <td className="px-2 py-1 text-right">{c.mean_label != null ? c.mean_label.toFixed(2) : (c.positives ?? 0).toLocaleString()}</td>
                          <td className="px-2 py-1 text-right">{c.mean_label != null ? '' : pct(c.base_rate)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {preview.dropped_cutoffs.length > 0 && (
                  <p className="text-[12px] text-copper-2">{preview.dropped_cutoffs.length} cutoffs left out: their label window is not complete in the data.</p>
                )}
                <details>
                  <summary className="cursor-pointer font-mono text-[11px] uppercase tracking-[0.1em] text-mute">The label SQL</summary>
                  <pre className="mt-2 overflow-x-auto border border-line p-3 font-mono text-[11px] text-bone-dim">{preview.sql}</pre>
                </details>
              </div>
            )}

            {task?.status === 'confirmed' ? (
              <p data-testid="task-confirmed" className="font-mono text-[12px] text-sage">
                Confirmed as {task.name} (version {task.version}) by {task.confirmed_by}. Nothing has been trained: running it comes with the training step.
              </p>
            ) : (
              <div className="flex flex-wrap gap-2">
                <Btn onClick={() => void check()} disabled={busy !== null}>
                  {busy === 'Building labels' ? 'Building labels…' : preview ? 'Rebuild labels' : 'Check the labels'}
                </Btn>
                <Btn variant="ghost" onClick={() => setEditing((e) => !e)} disabled={busy !== null}>
                  {editing ? 'Stop editing' : 'Edit the YAML'}
                </Btn>
                <Btn onClick={() => void confirm()} disabled={busy !== null || !task || !preview || blocked || errors.length > 0}>
                  Confirm
                </Btn>
                {blocked && <span className="self-center text-[12px] text-clay">Blocked by the feasibility checks above.</span>}
                {!preview && <span className="self-center text-[12px] text-mute">Check the labels first.</span>}
              </div>
            )}
          </div>
        )}
      </div>
    </Panel>
  );
}
