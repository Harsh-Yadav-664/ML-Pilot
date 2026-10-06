import { useCallback, useEffect, useState } from 'react';
import { describeError } from '../api';
import { LLMCall, LLMCallDetail, getLlmCall, listLlmCalls } from '../api/privacy';
import { Btn, CardHeader, Panel, Tag } from '../ui';

const MODE_TONE = { llm: 'sage', fallback: 'copper', failed: 'clay', pending: 'neutral' } as const;

function Detail({ call, onClose }: { call: LLMCall; onClose: () => void }) {
  const [detail, setDetail] = useState<LLMCallDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    getLlmCall(call.id)
      .then(setDetail)
      .catch((e) => setError(describeError(e)));
  }, [call.id]);

  const manifest = detail?.manifest ?? {};
  return (
    <Panel ticks className="mt-3">
      <CardHeader title={call.purpose} sub={`${call.provider} · ${call.model}`} right={<Btn size="sm" variant="ghost" onClick={onClose}>Close</Btn>} />
      <div className="space-y-4 p-4">
        {error && <p role="alert" className="text-[12px] text-clay">{error}</p>}
        {!detail && !error && <p className="font-mono text-[11px] text-mute">Loading…</p>}
        {detail && (
          <>
            <section>
              <h4 className="font-mono text-[10px] uppercase tracking-[0.12em] text-mute">Sent to the provider</h4>
              {detail.system && <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap border border-line bg-ink p-3 font-mono text-[11px] text-bone-dim">{detail.system}</pre>}
              <pre className="mt-1 max-h-80 overflow-auto whitespace-pre-wrap border border-line bg-ink p-3 font-mono text-[11px] text-bone">{detail.prompt}</pre>
            </section>
            <section>
              <h4 className="font-mono text-[10px] uppercase tracking-[0.12em] text-mute">What the prompt contains</h4>
              <p className="mt-1 font-mono text-[11px] text-bone-dim">
                level {String(manifest.level)} · statistics: {((manifest.stats_used as string[] | undefined) ?? []).join(', ') || 'none'} · category labels:{' '}
                {manifest.category_labels_included ? 'yes' : 'no'} · sample values: {manifest.sample_values_included ? 'yes' : 'no'} · columns left out:{' '}
                {String(manifest.excluded_columns ?? 0)}
              </p>
            </section>
            <section>
              <h4 className="font-mono text-[10px] uppercase tracking-[0.12em] text-mute">Response</h4>
              <pre className="mt-1 max-h-60 overflow-auto whitespace-pre-wrap border border-line bg-ink p-3 font-mono text-[11px] text-bone-dim">
                {detail.error ?? (typeof detail.response === 'string' ? detail.response : JSON.stringify(detail.response, null, 2))}
              </pre>
            </section>
          </>
        )}
      </div>
    </Panel>
  );
}

/** Every prompt sent for the active project: provider, model, size, cost, and the full text on click. */
export function PromptLog() {
  const [calls, setCalls] = useState<LLMCall[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<LLMCall | null>(null);

  const load = useCallback(() => {
    setError(null);
    listLlmCalls()
      .then(setCalls)
      .catch((e) => setError(describeError(e)));
  }, []);
  useEffect(load, [load]);

  return (
    <div data-testid="prompt-log">
      <Panel className="mt-3">
        <CardHeader title="Prompts sent" sub={calls ? `${calls.length} most recent` : undefined} right={<Btn size="sm" onClick={load}>Refresh</Btn>} />
        {error && <p role="alert" className="p-4 text-[12px] text-clay">{error}</p>}
        {calls && calls.length === 0 && <p className="p-4 font-mono text-[11px] text-mute">No prompt has been sent for this project yet.</p>}
        {calls && calls.length > 0 && (
          <table className="w-full text-left font-mono text-[11px]">
            <thead className="text-mute">
              <tr className="border-b border-line">
                <th className="px-4 py-2 font-medium">time</th>
                <th className="font-medium">purpose</th>
                <th className="font-medium">provider · model</th>
                <th className="text-right font-medium">size</th>
                <th className="text-right font-medium">cost</th>
                <th className="px-4 text-right font-medium">result</th>
              </tr>
            </thead>
            <tbody>
              {calls.map((c) => (
                <tr key={c.id} onClick={() => setOpen(c)} className="cursor-pointer border-b border-line text-bone-dim hover:bg-panel-2" data-testid={`prompt-${c.id}`}>
                  <td className="px-4 py-2">{new Date(c.created_at).toLocaleString()}</td>
                  <td className="text-bone">{c.purpose}</td>
                  <td>
                    {c.provider} · {c.model}
                  </td>
                  <td className="text-right">{c.prompt_chars.toLocaleString('en-US')} chars</td>
                  <td className="text-right">${c.cost_usd.toFixed(4)}</td>
                  <td className="px-4 text-right">
                    <Tag tone={MODE_TONE[c.decision_mode as keyof typeof MODE_TONE] ?? 'neutral'}>{c.decision_mode === 'fallback' ? 'offline stub' : c.decision_mode}</Tag>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
      {open && <Detail key={open.id} call={open} onClose={() => setOpen(null)} />}
    </div>
  );
}
