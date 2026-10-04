import { useState } from 'react';
import { useStore } from '../store';
import { Provider } from '../types';
import { cn } from '../utils/cn';
import { Btn, Eyebrow, Panel, SectionHead } from '../ui';

const PROVIDERS: { id: Provider; name: string; hint: string; models: string[] }[] = [
  { id: 'anthropic', name: 'Anthropic', hint: 'Claude · long reasoning traces', models: ['claude-sonnet-4-5', 'claude-opus-4-6'] },
  { id: 'openai', name: 'OpenAI', hint: 'GPT family · structured JSON', models: ['gpt-4.1', 'gpt-4o'] },
  { id: 'local', name: 'Local', hint: 'Ollama or vLLM on your machine · key optional', models: ['llama-3.3-70b', 'qwen2.5-32b'] },
];

export function Settings() {
  const { provider, setProvider, apiKey, setApiKey, providerModel, setProviderModel, toast } = useStore();
  const [show, setShow] = useState(false);
  const models = PROVIDERS.find((p) => p.id === provider)?.models ?? [];
  const ready = !!apiKey || provider === 'local';

  return (
    <div className="fade-up mx-auto max-w-[720px] space-y-8 p-6">
      <div>
        <Eyebrow className="text-copper">Workspace</Eyebrow>
        <h1 className="mt-1 font-display text-[44px] font-extrabold uppercase leading-none tracking-wide text-bone">Settings</h1>
        <p className="mt-2 text-[13px] text-mute">Bring your own key. Keys stay in this browser session and are never sent anywhere but your chosen provider.</p>
      </div>

      <div>
        <SectionHead n="01" title="AI provider" />
        <Panel ticks className="mt-3">
          {PROVIDERS.map((p) => {
            const on = provider === p.id;
            return (
              <button
                key={p.id}
                onClick={() => {
                  setProvider(p.id);
                  setProviderModel(p.models[0]);
                }}
                className={cn('relative flex w-full items-center gap-4 border-b border-line px-5 py-4 text-left transition-colors last:border-b-0', on ? 'bg-panel-2' : 'hover:bg-panel-2/50')}
              >
                {on && <span className="absolute inset-y-0 left-0 w-[3px] bg-copper" />}
                <span className={cn('flex h-4 w-4 shrink-0 items-center justify-center border', on ? 'border-copper' : 'border-rule')}>
                  {on && <span className="h-2 w-2 bg-copper" />}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block font-display text-[22px] font-bold uppercase leading-none tracking-wide text-bone">{p.name}</span>
                  <span className="mt-1 block font-mono text-[11px] text-mute">{p.hint}</span>
                </span>
              </button>
            );
          })}
        </Panel>
      </div>

      <div>
        <SectionHead n="02" title="Credentials" />
        <Panel className="mt-3 space-y-5 p-5">
          <div>
            <Eyebrow className="mb-1.5 block">Model</Eyebrow>
            <select
              value={providerModel}
              onChange={(e) => setProviderModel(e.target.value)}
              className="h-10 w-full cursor-pointer border border-rule bg-ink px-3 font-mono text-[12.5px] text-bone focus:border-copper focus:outline-none"
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div>
            <Eyebrow className="mb-1.5 block">API key</Eyebrow>
            <div className="flex border border-rule bg-ink focus-within:border-copper">
              <input
                type={show ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={provider === 'local' ? 'optional' : 'sk-…'}
                className="h-10 min-w-0 flex-1 bg-transparent px-3 font-mono text-[12.5px] text-bone placeholder:text-mute/70 focus:outline-none"
              />
              <button type="button" onClick={() => setShow((s) => !s)} className="border-l border-rule px-3 font-mono text-[10px] uppercase tracking-[0.1em] text-mute transition-colors hover:text-bone">
                {show ? 'hide' : 'show'}
              </button>
            </div>
          </div>
          <div className="flex items-center justify-between border-t border-line pt-4">
            <span className="font-mono text-[11px] uppercase tracking-[0.1em]" style={{ color: ready ? '#b2cd8b' : '#a08d76' }}>
              {ready ? '■ ready' : '□ no key set'}
            </span>
            <Btn
              variant="primary"
              onClick={() => toast(ready ? 'Provider saved for this session' : 'Enter a key first', ready ? 'ok' : 'warn')}
            >
              Save credentials
            </Btn>
          </div>
        </Panel>
      </div>
    </div>
  );
}
