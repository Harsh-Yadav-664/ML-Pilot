import { execFileSync } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, test } from '@playwright/test';

/**
 * Ask a question, read the drafted task, check the labels, confirm (backend issue #53). Runs against
 * `python start.py` (offline stub provider, so the rule-based drafter answers and says so) and a DuckDB
 * copy of the demo database that the test generates itself.
 */
const python = process.env.PYTHON ?? 'python';

function demoDuckdb(): string {
  const dir = mkdtempSync(join(tmpdir(), 'mlpilot-task-'));
  const path = join(dir, 'demo.duckdb');
  execFileSync(
    python,
    ['-c', 'import sys; from pathlib import Path; from tests.fixtures.demo_db import demo_duckdb; demo_duckdb(Path(sys.argv[1]))', path],
    { cwd: join(process.cwd(), '..', 'backend'), stdio: 'inherit' },
  );
  return path;
}

test('a question becomes a task: draft, check the labels, confirm', async ({ page }) => {
  const path = demoDuckdb();
  await expect
    .poll(async () => (await page.request.get('http://127.0.0.1:8000/health').catch(() => null))?.status(), { timeout: 120_000 })
    .toBe(200);
  await page.goto('/#/connect');
  await page.getByRole('radio', { name: 'DuckDB' }).click();
  await page.getByLabel('Name', { exact: true }).fill(`demo-duckdb-${Date.now()}`);
  await page.getByLabel('File path').fill(path);
  await page.getByRole('button', { name: 'Save & test' }).click();
  await expect(page.getByTestId('connect-ok')).toBeVisible({ timeout: 60_000 });

  const card = page.getByTestId('task-card');
  await expect(card).toBeVisible({ timeout: 60_000 });

  // Too vague: a question comes back, no task.
  await card.getByLabel('Prediction question').fill('predict customers');
  await card.getByRole('button', { name: 'Draft the task' }).click();
  await expect(card.getByTestId('task-clarify')).toBeVisible({ timeout: 120_000 });
  await expect(card.getByTestId('task-draft')).toHaveCount(0);

  // A clear question: plain words, assumptions, and the fallback is labelled.
  await card.getByLabel('Prediction question').fill('Which customers will stop ordering in the next 30 days?');
  await card.getByRole('button', { name: 'Draft the task' }).click();
  const draft = card.getByTestId('task-draft');
  await expect(draft).toBeVisible({ timeout: 120_000 });
  await expect(card.getByTestId('task-description')).toContainText('labelled 1');
  await expect(draft.getByText(/fallback: rule-based drafter/)).toBeVisible();
  await expect(draft.getByText('Assumptions made')).toBeVisible();

  // Confirm needs the label preview first.
  await expect(draft.getByRole('button', { name: 'Confirm' })).toBeDisabled();
  await draft.getByRole('button', { name: 'Check the labels' }).click();
  const preview = card.getByTestId('task-preview');
  await expect(preview).toBeVisible({ timeout: 300_000 });
  await expect(preview.getByText(/feasibility: (ok|warn)/)).toBeVisible();
  expect(await preview.getByRole('row').count()).toBeGreaterThan(5);

  await draft.getByRole('button', { name: 'Confirm' }).click();
  await expect(card.getByTestId('task-confirmed')).toContainText('Nothing has been trained');
});
