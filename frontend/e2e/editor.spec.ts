import { expect, test } from '@playwright/test';
import { demoDuckdb } from './demoDb';

/**
 * The task editor (backend issue #99): build the churn task by hand with no language model, change the
 * horizon, watch the preview follow, confirm, and compare the stored spec with the expected one.
 */
const ELIGIBLE_90 = "ordered_at >= :cutoff - interval '90 days' AND ordered_at < :cutoff";

test('build the churn task by hand, change the horizon, see the preview update, confirm', async ({ page }) => {
  const path = demoDuckdb();
  const name = `editor-duckdb-${Date.now()}`;
  await expect
    .poll(async () => (await page.request.get('http://127.0.0.1:8000/health').catch(() => null))?.status(), { timeout: 120_000 })
    .toBe(200);
  await page.goto('/#/connect');
  await page.getByRole('radio', { name: 'DuckDB' }).click();
  await page.getByLabel('Name', { exact: true }).fill(name);
  await page.getByLabel('File path').fill(path);
  await page.getByRole('button', { name: 'Save & test' }).click();
  await expect(page.getByTestId('connect-ok')).toBeVisible({ timeout: 60_000 });

  await page.getByRole('button', { name: 'Task editor' }).click();
  await expect(page.getByRole('heading', { name: 'Task editor' })).toBeVisible();
  await page.getByLabel('Connection').selectOption({ label: `${name} (duckdb)` });

  // Who and when. The schema graph fills the pickers.
  const editor = page.getByTestId('task-editor');
  await editor.getByLabel('Task name').fill('churn_30d');
  await expect(editor.getByLabel('Entity table').locator('option[value="customers"]')).toHaveCount(1, { timeout: 60_000 });
  await editor.getByLabel('Entity table').selectOption('customers');
  await expect(editor.getByLabel('Entity key')).toHaveValue('customer_id');
  await expect(editor.getByLabel('Entity created at')).toHaveValue('signup_at');
  await editor.getByLabel('Horizon').fill('30d');
  await editor.getByLabel('Cutoffs start').fill('2023-04-01');
  await editor.getByLabel('Cutoffs end').fill('2024-10-01');
  await editor.getByLabel('Validation from').fill('2024-04-01');
  await editor.getByLabel('Test from').fill('2024-07-01');

  // What to predict: no order that is not cancelled in the window.
  await editor.getByLabel('Target table').selectOption('orders');
  await editor.getByLabel('Aggregate').selectOption('count');
  await editor.getByLabel('Target filter').fill("status != 'cancelled'");
  await editor.getByLabel('Label is 1 when').fill('= 0');

  // Who is scored: signed up before the cutoff, and ordered in the 90 days before it.
  await editor.getByRole('button', { name: 'Add a rule' }).click();
  await editor.getByLabel('Rule 1 condition').fill('signup_at < :cutoff');
  await editor.getByRole('button', { name: 'Add a rule' }).click();
  await editor.getByLabel('Rule 2 kind').selectOption('exists');
  await editor.getByLabel('Rule 2 table').selectOption('orders');
  await editor.getByLabel('Rule 2 where').fill(ELIGIBLE_90);

  // The preview appears once the spec is complete, with labels for every cutoff.
  const preview = page.getByTestId('editor-preview').getByTestId('task-preview');
  await expect(preview).toBeVisible({ timeout: 120_000 });
  await expect(preview.getByText(/feasibility: (ok|warn)/)).toBeVisible();
  const firstBalance = async () => (await preview.locator('tbody tr').first().locator('td').last().innerText()).trim();
  const before = await firstBalance();
  expect(before).toMatch(/%$/);
  expect(await preview.locator('tbody tr').count()).toBe(19);

  // Change the horizon: the YAML tab and the labels follow (a longer window, so fewer churners).
  await editor.getByLabel('Horizon').fill('60d');
  await expect.poll(firstBalance, { timeout: 120_000 }).not.toBe(before);
  await page.getByRole('tab', { name: 'YAML' }).click();
  await expect(page.getByLabel('Task spec YAML')).toHaveValue(/horizon: 60d/);
  await page.getByRole('tab', { name: 'Form' }).click();

  // A broken spec shows its problem next to the field and in the panel; no preview, no confirm.
  await editor.getByLabel('Horizon').fill('soon');
  await expect(page.getByTestId('editor-errors')).toContainText('horizon', { timeout: 60_000 });
  await expect(page.getByRole('button', { name: 'Confirm this task' })).toBeDisabled();
  await editor.getByLabel('Horizon').fill('60d');
  await expect(preview).toBeVisible({ timeout: 120_000 });

  // Confirm: the stored spec is the one built here.
  const confirmed = page.waitForResponse((r) => r.url().endsWith('/confirm') && r.request().method() === 'POST');
  await expect(page.getByRole('button', { name: 'Confirm this task' })).toBeEnabled({ timeout: 120_000 });
  await page.getByRole('button', { name: 'Confirm this task' }).click();
  const body = await (await confirmed).json();
  expect(body.status).toBe('confirmed');
  expect(body.spec).toEqual({
    name: 'churn_30d',
    entity: { table: 'customers', key: 'customer_id', created_at: 'signup_at' },
    eligibility: ['signup_at < :cutoff', { exists: { table: 'orders', where: ELIGIBLE_90 } }],
    target: { type: 'binary', expression: { table: 'orders', agg: 'count', where: "status != 'cancelled'", compare: '= 0' } },
    horizon: '60d',
    cutoffs: { start: '2023-04-01', end: '2024-10-01', every: '1 month' },
    split: { val_from: '2024-04-01', test_from: '2024-07-01' },
  });
  await expect(page.getByTestId('task-confirmed')).toContainText('Nothing has been trained');
  await expect(page.getByTestId('task-stored-yaml')).toContainText('horizon: 60d');
});
