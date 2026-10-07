import { expect, test } from '@playwright/test';
import { demoDuckdb } from './demoDb';

/**
 * A full relational run in the UI (backend issue #101): ask, confirm, start a run, watch it, read the result. Runs against
 * `python start.py` (offline stub provider: the model proposes nothing, so the baseline features and the champion path
 * are what there is to see) and a DuckDB copy of the demo database that the test generates itself.
 * Every number on the page is compared with the API response the page itself received.
 */

type State = {
  status: string;
  stop_reason: string | null;
  rounds: number;
  champion_val_metrics: Record<string, number> | null;
  test_metrics: Record<string, number> | null;
};
type Feature = { id: string; kind: string; status: string };

test('start a run from a confirmed task and read it: features, champion, test score only at the end', async ({ page }, testInfo) => {
  const path = demoDuckdb();
  const states: State[] = [];
  let features: Feature[] = [];
  page.on('response', async (r) => {
    if (r.request().method() !== 'GET') return;
    const pathname = new URL(r.url()).pathname;
    try {
      if (/\/runs\/[^/]+$/.test(pathname)) states.push((await r.json()) as State);
      if (/\/runs\/[^/]+\/features$/.test(pathname)) features = ((await r.json()) as { features: Feature[] }).features;
    } catch {
      // a response that was cancelled by navigation
    }
  });

  await expect.poll(async () => (await page.request.get('http://127.0.0.1:8000/health').catch(() => null))?.status(), { timeout: 120_000 }).toBe(200);
  await page.goto('/#/connect');
  await page.getByRole('radio', { name: 'DuckDB' }).click();
  await page.getByLabel('Name', { exact: true }).fill(`demo-run-${Date.now()}`);
  await page.getByLabel('File path').fill(path);
  await page.getByRole('button', { name: 'Save & test' }).click();
  await expect(page.getByTestId('connect-ok')).toBeVisible({ timeout: 60_000 });

  const card = page.getByTestId('task-card');
  await card.getByLabel('Prediction question').fill('Which customers will stop ordering in the next 30 days?');
  await card.getByRole('button', { name: 'Draft the task' }).click();
  const draft = card.getByTestId('task-draft');
  await expect(draft).toBeVisible({ timeout: 120_000 });
  await draft.getByRole('button', { name: 'Check the labels' }).click();
  await expect(card.getByTestId('task-preview')).toBeVisible({ timeout: 300_000 });
  await draft.getByRole('button', { name: 'Confirm' }).click();
  await expect(card.getByTestId('task-confirmed')).toBeVisible();

  // Start: snapshot, run, loop. The page moves to the run.
  await card.getByRole('button', { name: 'Start a run' }).click();
  await expect(page).toHaveURL(/#\/run\//, { timeout: 300_000 });
  const status = page.getByTestId('run-status');
  await expect(status).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId('run-task')).not.toHaveText('Run');

  // While the run is live the test rows are not shown; after it, they are.
  let sawLive = false;
  const deadline = Date.now() + 600_000;
  for (;;) {
    const s = (await status.getAttribute('data-status')) ?? '';
    if (['queued', 'running'].includes(s)) {
      sawLive = true;
      await expect(page.getByTestId('run-test-metrics')).toHaveCount(0);
    }
    if (['completed', 'stopped', 'failed', 'cancelled'].includes(s)) break;
    expect(Date.now()).toBeLessThan(deadline);
    await page.waitForTimeout(500);
  }
  await expect(status).toHaveAttribute('data-status', 'completed');
  console.log(`saw the run live before it ended: ${sawLive}`);

  // The page's numbers are the API's. Wait for the last poll after the end, then compare.
  await expect(page.getByTestId('run-test-metrics')).toBeVisible();
  await expect.poll(() => states.at(-1)?.status).toBe('completed');
  const last = states.at(-1) as State;
  expect(last.test_metrics).not.toBeNull();
  expect(last.stop_reason).toBe('no_llm'); // the offline stub proposes nothing
  expect(last.rounds).toBe(0);

  const test = page.getByTestId('run-test-metrics');
  const val = page.getByTestId('run-validation-metrics');
  for (const key of ['pr_auc', 'base_rate']) {
    await expect(test.locator(`[data-metric="${key}"]`)).toHaveText((last.test_metrics as Record<string, number>)[key].toFixed(4));
    await expect(val.locator(`[data-metric="${key}"]`)).toHaveText((last.champion_val_metrics as Record<string, number>)[key].toFixed(4));
  }
  await expect(test.locator('[data-metric="n_test"]')).toHaveText((last.test_metrics as Record<string, number>).n_test.toLocaleString());

  // Features: one row per feature of the API, all of them the baseline's.
  const rows = page.getByTestId('feature-row');
  await expect(rows).toHaveCount(features.length);
  expect(features.length).toBeGreaterThan(50);
  expect(features.every((f) => f.kind === 'dfs')).toBe(true);
  await expect(page.locator('[data-testid="feature-row"][data-status="baseline"]')).toHaveCount(features.length);
  await page.getByRole('button', { name: /^baseline \d+$/i }).click();
  await expect(rows).toHaveCount(features.length);

  // Champion path: the baseline, with the API's validation PR-AUC.
  const steps = page.getByTestId('champion-step');
  await expect(steps).toHaveCount(1);
  await expect(steps.first()).toContainText((last.champion_val_metrics as Record<string, number>).pr_auc.toFixed(4));

  // The split and the log.
  await expect(page.getByTestId('split-timeline')).toBeVisible({ timeout: 120_000 });
  expect(await page.getByTestId('split-timeline').getByRole('row').count()).toBeGreaterThan(5);
  await expect(page.getByTestId('run-log')).toContainText('Baseline:');
  await expect(page.getByTestId('run-log')).toContainText('No language model answered');

  const dir = process.env.MLPILOT_E2E_SCREENSHOT_DIR ?? testInfo.outputDir;
  await page.screenshot({ path: `${dir}/run-view.png`, fullPage: true });
});
