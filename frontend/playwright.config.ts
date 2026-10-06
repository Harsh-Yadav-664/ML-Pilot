import { defineConfig, devices } from '@playwright/test';

/**
 * UI smoke test via `python start.py`: the real backend (offline stub LLM, throwaway SQLite DB) and the real UI.
 * PW_CHROMIUM lets a machine with a preinstalled Chromium skip `playwright install`.
 */
const python = process.env.PYTHON ?? 'python';
const db = `sqlite+aiosqlite:///${process.env.SMOKE_DB ?? '/tmp/mlpilot-smoke.db'}`;

export default defineConfig({
  testDir: 'e2e',
  timeout: 180_000,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'retain-on-failure',
    launchOptions: process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM } : {},
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  // The one command a user runs: it creates the access token on first start and hands
  // it to the UI, so the test proves there is no manual token step.
  webServer: {
    command: `${python} start.py`,
    cwd: '..',
    url: 'http://localhost:5173',
    env: {
      DATABASE_URL: db,
      MLPILOT_TOKEN_FILE: process.env.SMOKE_TOKEN_FILE ?? '/tmp/mlpilot-smoke/token',
    },
    timeout: 120_000,
    reuseExistingServer: false,
  },
});
