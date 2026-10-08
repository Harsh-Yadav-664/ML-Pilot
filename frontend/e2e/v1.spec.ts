import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, test } from '@playwright/test';

/**
 * v1 definition of done (issue #102): from "connect" to a ranked list of customers, through the UI, on the demo
 * Postgres as the read-only role. Each step below is one line of the README's "works today" list.
 * Needs the compose demo database; the CI job `v1-e2e` provides it (MLPILOT_E2E_PG_*).
 */
const host = process.env.MLPILOT_E2E_PG_HOST;
const port = process.env.MLPILOT_E2E_PG_PORT ?? '5433';
const database = process.env.MLPILOT_E2E_PG_DB ?? 'demo';
const roUser = process.env.MLPILOT_E2E_PG_RO_USER ?? 'mlpilot_ro';
const roPassword = process.env.MLPILOT_E2E_PG_RO_PASSWORD ?? '';
const python = process.env.PYTHON ?? 'python';

const TABLES = ['customers', 'customer_status_snapshot', 'marketing_emails', 'order_items', 'orders', 'products', 'refunds', 'sessions', 'support_tickets'];

test.skip(!host, 'MLPILOT_E2E_PG_HOST is not set: no demo database to connect to');

type Feature = { id: string; name: string; kind: string; status: string };

test('v1: connect, ask, run, report, export, score', async ({ page }) => {
  test.setTimeout(1_800_000);
  let features: Feature[] = [];
  page.on('response', async (r) => {
    if (r.request().method() !== 'GET') return;
    try {
      if (/\/runs\/[^/]+\/features$/.test(new URL(r.url()).pathname)) features = ((await r.json()) as { features: Feature[] }).features;
    } catch {
      // a response cancelled by navigation
    }
  });
  await expect
    .poll(async () => (await page.request.get('http://127.0.0.1:8000/health').catch(() => null))?.status(), { timeout: 120_000 })
    .toBe(200);

  await test.step('1. connect to the demo database as the read-only role', async () => {
    await page.goto('/#/connect');
    await page.getByRole('radio', { name: 'Postgres' }).click();
    await page.getByLabel('Name', { exact: true }).fill('v1-demo');
    await page.getByLabel('Host', { exact: true }).fill(host!);
    await page.getByLabel('Port', { exact: true }).fill(port);
    await page.getByLabel('Database', { exact: true }).fill(database);
    await page.getByLabel('User', { exact: true }).fill(roUser);
    await page.getByLabel('Password', { exact: true }).fill(roPassword);
    await page.getByRole('button', { name: 'Save & test' }).click();
    await expect(page.getByTestId('connect-ok')).toBeVisible({ timeout: 60_000 });
    await expect(page.getByTestId('read-only-ok')).toBeVisible();
    await expect(page.getByTestId('write-warning')).toHaveCount(0);
  });

  await test.step('2. the schema graph shows every demo table; products is marked static', async () => {
    const graph = page.getByTestId('schema-graph');
    for (const t of TABLES) await expect(graph.getByTestId(`table-${t}`)).toBeVisible({ timeout: 60_000 });
    await graph.getByTestId('table-products').click();
    const box = page.getByTestId('table-details').getByLabel(/Static table/);
    await box.click(); // a controlled checkbox: it changes when the choice is saved
    await expect(box).toBeChecked({ timeout: 30_000 });
  });

  const card = page.getByTestId('task-card');
  await test.step('3. ask the question; the task card shows the drafted task', async () => {
    await card.getByLabel('Prediction question').fill('Which customers will stop ordering in the next 30 days?');
    await card.getByRole('button', { name: 'Draft the task' }).click();
    await expect(card.getByTestId('task-draft')).toBeVisible({ timeout: 120_000 });
    await expect(card.getByTestId('task-description')).toContainText('labelled 1');
  });

  await test.step('4. the label preview shows the base rate per cutoff and feasibility; confirm the task', async () => {
    const draft = card.getByTestId('task-draft');
    await draft.getByRole('button', { name: 'Check the labels' }).click();
    const preview = card.getByTestId('task-preview');
    await expect(preview).toBeVisible({ timeout: 300_000 });
    await expect(preview.getByText(/feasibility: (ok|warn)/)).toBeVisible();
    expect(await preview.getByRole('row').count()).toBeGreaterThan(5);
    await draft.getByRole('button', { name: 'Confirm' }).click();
    await expect(card.getByTestId('task-confirmed')).toBeVisible({ timeout: 300_000 });
  });

  await test.step('5. start a run: baseline features, the scripted proposals, the leaky one refused by the guard', async () => {
    await card.getByRole('button', { name: 'Start a run' }).click();
    await expect(page).toHaveURL(/#\/run\//, { timeout: 600_000 });
    const status = page.getByTestId('run-status');
    await expect(status).toBeVisible({ timeout: 60_000 });
    await expect(status).toHaveAttribute('data-status', /^(completed|stopped)$/, { timeout: 1_200_000 });
    await expect(page.getByTestId('feature-row').first()).toBeVisible();
    expect(features.filter((f) => f.kind === 'dfs').length).toBeGreaterThan(50);
    const proposals = features.filter((f) => f.kind === 'llm_sql');
    expect(proposals.length).toBe(4);
    // the fourth proposal reads the label's own window: the guards refuse it before any gain is measured
    expect(proposals[3].status).toBe('rejected_guard');
    expect(proposals.slice(0, 3).every((f) => ['accepted', 'rejected_gain'].includes(f.status))).toBe(true);
  });

  await test.step('6. the test metric is shown once the run has ended; the report opens with its safety section', async () => {
    await expect(page.getByTestId('run-test-metrics')).toBeVisible();
    await expect(page.getByTestId('run-test-metrics').locator('[data-metric="pr_auc"]')).toHaveText(/^\d\.\d{4}$/);
    const [report] = await Promise.all([page.waitForEvent('popup'), page.getByTestId('report-html').click()]);
    await report.waitForLoadState();
    await expect(report.locator('h1')).toContainText('run report');
    await expect(report.locator('body')).toContainText('Leakage and safety');
  });

  const out = mkdtempSync(join(tmpdir(), 'mlpilot-v1-'));
  await test.step('7. export the bundle; in a clean virtualenv score.py reproduces the validation metric', async () => {
    const [download] = await Promise.all([page.waitForEvent('download'), page.getByTestId('export-bundle').click()]);
    const zip = join(out, 'bundle.zip');
    await download.saveAs(zip);
    const dir = join(out, 'bundle');
    mkdirSync(dir);
    execFileSync(python, ['-I', '-c', 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])', zip, dir]);
    const root = join(dir, execFileSync(python, ['-I', '-c', 'import os, sys; print(os.listdir(sys.argv[1])[0])', dir]).toString().trim());
    const venv = join(out, 'venv');
    execFileSync(python, ['-m', 'venv', venv], { stdio: 'inherit' });
    const bin = join(venv, 'bin');
    execFileSync(join(bin, 'pip'), ['install', '--quiet', '-r', join(root, 'requirements.txt')], { stdio: 'inherit' });
    const verified = execFileSync(join(bin, 'python'), ['-I', 'score.py', '--verify'], {
      cwd: root,
      env: { ...process.env, MLPILOT_DB_URL: `postgresql://${roUser}:${roPassword}@${host}:${port}/${database}` },
      timeout: 900_000,
    }).toString();
    console.log(verified);
    expect(verified).toContain('OK: reproduced within 1e-6');
  });

  await test.step('8. score the latest day; download the ranked list with reasons', async () => {
    await page.getByTestId('score-run').click();
    const result = page.getByTestId('score-result');
    await expect(result).toBeVisible({ timeout: 600_000 });
    await expect(page.getByTestId('score-error')).toHaveCount(0);
    const [download] = await Promise.all([page.waitForEvent('download'), page.getByTestId('score-csv').click()]);
    const csv = join(out, 'scores.csv');
    await download.saveAs(csv);
    const lines = readFileSync(csv, 'utf8').trim().split('\n');
    expect(lines[0]).toBe('entity_id,score,rank,decile,reason_1,reason_2,reason_3');
    expect(lines.length).toBeGreaterThan(100);
    const scores = lines.slice(1).map((l) => Number(l.split(',')[1]));
    expect(scores.every((s) => s >= 0 && s <= 1)).toBe(true);
    expect([...scores].sort((a, b) => b - a)).toEqual(scores);
    expect(lines[1]).toMatch(/\(\+|\(-/); // the strongest reasons carry their direction
  });
});
