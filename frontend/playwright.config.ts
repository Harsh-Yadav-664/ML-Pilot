import { defineConfig, devices } from '@playwright/test';

/**
 * UI smoke test: the real backend (offline stub LLM, throwaway SQLite DB) and the real UI.
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
  webServer: [
    {
      command: `${python} -m uvicorn app.main:app --port 8000`,
      cwd: '../backend',
      url: 'http://localhost:8000/health',
      env: { DATABASE_URL: db },
      timeout: 120_000,
      reuseExistingServer: false,
    },
    {
      command: 'npm run dev -- --port 5173 --strictPort',
      url: 'http://localhost:5173',
      timeout: 60_000,
      reuseExistingServer: false,
    },
  ],
});
