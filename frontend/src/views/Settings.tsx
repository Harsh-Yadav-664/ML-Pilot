import { Eyebrow, Panel, SectionHead } from '../ui';
import { PrivacyPanel } from '../components/PrivacyPanel';
import { PromptLog } from './PromptLog';

/** The variables the backend reads to enable a provider (PROVIDER_FACTORIES in backend/ai/gateway.py). */
const PROVIDER_ENV = [
  'GROQ_API_KEY',
  'GEMINI_API_KEY',
  'NVIDIA_API_KEY',
  'OPENROUTER_API_KEY',
  'CEREBRAS_API_KEY',
  'MISTRAL_API_KEY',
  'OPENAI_API_KEY',
  'ANTHROPIC_API_KEY',
  'OLLAMA_BASE_URL',
];

export function Settings() {
  return (
    <div className="fade-up mx-auto max-w-[720px] space-y-8 p-6">
      <div>
        <Eyebrow className="text-copper">Workspace</Eyebrow>
        <h1 className="mt-1 font-display text-[44px] font-extrabold uppercase leading-none tracking-wide text-bone">Settings</h1>
        <p className="mt-2 text-[13px] text-mute">What the LLM may see, and every prompt it was sent.</p>
      </div>

      <div>
        <SectionHead n="01" title="AI provider" />
        <Panel ticks className="mt-3 p-5 text-[13px] leading-relaxed text-bone-dim">
          <p>
            Providers are configured on the backend, not in the browser: set one or more of{' '}
            {PROVIDER_ENV.map((v, i) => (
              <span key={v}>
                {i > 0 && ', '}
                <code className="font-mono text-[12px] text-copper-2">{v}</code>
              </span>
            ))}{' '}
            as environment variables (or in <code className="font-mono text-[12px] text-copper-2">backend/.env</code>) and restart the backend.
          </p>
          <p className="mt-2 text-mute">
            With none set, MLPilot uses its offline stub provider and every call is recorded as a fallback. The provider and model of each call are in the
            prompt log below.
          </p>
        </Panel>
      </div>

      <div>
        <SectionHead n="02" title="What the LLM sees" />
        <PrivacyPanel />
      </div>

      <div>
        <SectionHead n="03" title="Prompt log" />
        <PromptLog />
      </div>
    </div>
  );
}
