import { randomBytes } from 'node:crypto';
import { resolve } from 'node:path';
import { defineConfig, devices } from '@playwright/test';

/**
 * The v1 acceptance test (issue #102): one spec walks the product from "connect" to "scored list" against the
 * demo Postgres, the way a user would. The server is `python start.py` with the offline stub LLM, plus a scripted
 * provider that replays four feature proposals (three sound ones and one that reads the label window), so the
 * feature loop and the guards have something real to decide. Trace and video are kept for every run.
 */
const python = process.env.PYTHON ?? 'python';
const secretKey = randomBytes(32).toString('base64').replace(/\+/g, '-').replace(/\//g, '_');
const proposals = resolve(process.cwd(), '..', 'backend', 'tests', 'fixtures', 'stub_feature_proposals.yaml');

export default defineConfig({
  testDir: 'e2e',
  testMatch: /v1\.spec\.ts/,
  timeout: 1_800_000,
  retries: 0,
  reporter: [['list'], ['html', { open: 'never', outputFolder: 'playwright-report-v1' }]],
  outputDir: 'test-results-v1',
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on',
    video: 'on',
    launchOptions: process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM } : {},
    ...devices['Desktop Chrome'],
  },
  webServer: {
    command: `${python} start.py`,
    cwd: '..',
    url: 'http://localhost:5173',
    env: {
      DATABASE_URL: `sqlite+aiosqlite:///${process.env.V1_DB ?? '/tmp/mlpilot-v1.db'}`,
      MLPILOT_TOKEN_FILE: process.env.V1_TOKEN_FILE ?? '/tmp/mlpilot-v1/token',
      MLPILOT_SECRET_KEY: secretKey,
      MLPILOT_TEST_SCRIPTED_LLM: `${proposals}:good_refunds_14d,good_cancelled_share,good_ticket_count_60d,leaky_sql`,
    },
    timeout: 120_000,
    reuseExistingServer: false,
  },
});
