import { useMemo, useState } from 'react';
import { useStore } from '../store';
import { algoLabel, curlSnippet, timeAgo } from '../lib';
import { cn } from '../utils/cn';
import { Btn, CardHeader, Eyebrow, Led, Leader, LineChart, Panel, SectionHead, Tag, VolumeBars } from '../ui';

export function Deployments() {
  const { endpoints, retireEndpoint, drift, volume, experiments } = useStore();
  const [copied, setCopied] = useState<string | null>(null);
  const psiNow = drift[drift.length - 1]?.psi ?? 0;
  const drifted = psiNow > 0.2;
  const pts = useMemo(() => drift.map((d, i) => ({ id: String(i), y: d.psi, label: `d-${drift.length - i}` })), [drift]);

  const copy = (id: string, text: string) => {
    navigator.clipboard?.writeText(text).catch(() => undefined);
    setCopied(id);
    setTimeout(() => setCopied(null), 1200);
  };

  return (
    <div className="fade-up mx-auto max-w-[1100px] space-y-8 p-6">
      <div>
        <Eyebrow className="text-copper">Serving</Eyebrow>
        <h1 className="mt-1 font-display text-[44px] font-extrabold uppercase leading-none tracking-wide text-bone">Deployments</h1>
        <p className="mt-2 max-w-xl text-[13px] text-mute">Live inference endpoints, request volume, and a drift watch on the production feature distribution.</p>
      </div>

      {drifted && (
        <div className="relative border border-clay/40 bg-clay/5 py-3 pl-5 pr-4 text-[13px] text-clay">
          <span className="absolute inset-y-0 left-0 w-[3px] bg-clay" />
          <span className="font-mono text-[11px] font-semibold uppercase tracking-[0.12em]">Drift alert</span> · PSI is {psiNow.toFixed(3)}, above the 0.20 line. Consider retraining the champion.
        </div>
      )}

      <div>
        <SectionHead n="01" title="Traffic & drift" />
        <div className="mt-3 grid gap-4 lg:grid-cols-2">
          <Panel>
            <CardHeader title="Request volume" sub="last 24h · rpm" />
            <div className="p-4">
              <VolumeBars data={volume} />
            </div>
          </Panel>
          <Panel>
            <CardHeader title="Data drift" sub="PSI vs training window" right={<Tag tone={drifted ? 'clay' : 'sage'}>{psiNow.toFixed(3)}</Tag>} />
            <div className="p-4">
              <LineChart points={pts} baseline={0.2} color={drifted ? '#e0705a' : '#e0a24f'} height={140} />
            </div>
          </Panel>
        </div>
      </div>

      <div>
        <SectionHead n="02" title="Endpoints" right={<Tag>{endpoints.length} live</Tag>} />
        <div className="mt-3 space-y-3">
          {endpoints.map((ep) => {
            const exp = experiments.find((e) => e.id === ep.experiment_id);
            const curl = curlSnippet(ep.url);
            const col = ep.status === 'healthy' ? '#b2cd8b' : ep.status === 'warming' ? '#f7cd82' : '#e0705a';
            return (
              <Panel key={ep.id} ticks>
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
                  <div className="flex items-center gap-3">
                    <Led color={col} pulse={ep.status !== 'healthy'} />
                    <span className="font-display text-[24px] font-bold uppercase leading-none tracking-wide text-bone">{algoLabel(ep.model_name)}</span>
                    <code className="font-mono text-[11px] text-mute">{ep.experiment_id}</code>
                    <Tag tone={ep.status === 'healthy' ? 'sage' : ep.status === 'warming' ? 'copper' : 'clay'}>{ep.status}</Tag>
                  </div>
                  <div className="flex items-center gap-4 font-mono text-[11px] text-mute">
                    <span>{ep.rpm.toLocaleString()} rpm</span>
                    <span>p95 {ep.p95_ms}ms</span>
                    <span>{timeAgo(ep.created_at)}</span>
                    <Btn variant="danger" size="sm" onClick={() => retireEndpoint(ep.id)}>
                      Retire
                    </Btn>
                  </div>
                </div>
                <div className="grid gap-5 p-4 lg:grid-cols-2">
                  <div className="min-w-0">
                    <Eyebrow>Endpoint</Eyebrow>
                    <code className="mt-1.5 block break-all font-mono text-[12px] text-copper-2">{ep.url}</code>
                    {exp?.metrics.f1 !== undefined && (
                      <div className="mt-3">
                        <Leader k="Serving F1" v={exp.metrics.f1.toFixed(3)} />
                        <Leader k="Accuracy" v={`${((exp.metrics.accuracy ?? 0) * 100).toFixed(1)}%`} />
                      </div>
                    )}
                  </div>
                  <div className="min-w-0">
                    <div className="mb-1.5 flex items-center justify-between">
                      <Eyebrow>cURL · POST /predict</Eyebrow>
                      <button onClick={() => copy(ep.id, curl)} className={cn('font-mono text-[10px] uppercase tracking-[0.1em] transition-colors', copied === ep.id ? 'text-sage' : 'text-mute hover:text-bone')}>
                        {copied === ep.id ? 'copied' : 'copy'}
                      </button>
                    </div>
                    <pre className="overflow-x-auto border border-line bg-ink p-3 font-mono text-[11px] leading-relaxed text-bone-dim [border-left-color:#e0a24f] [border-left-width:3px]">{curl}</pre>
                  </div>
                </div>
              </Panel>
            );
          })}
          {!endpoints.length && (
            <Panel className="p-10 text-center text-[13px] text-mute">No endpoints yet. Promote a completed run and deploy it from Experiments.</Panel>
          )}
        </div>
      </div>
    </div>
  );
}
