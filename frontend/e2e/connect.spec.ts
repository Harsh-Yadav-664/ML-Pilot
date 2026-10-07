import { expect, test, type Page } from '@playwright/test';

/**
 * "Connect a database" against the demo Postgres (docker/demo-db), as a user would use it.
 * Needs the demo database: the CI job `demo-db` provides it and sets MLPILOT_E2E_PG_*; without
 * them the spec is skipped (the ui-smoke job has no database).
 */
const host = process.env.MLPILOT_E2E_PG_HOST;
const port = process.env.MLPILOT_E2E_PG_PORT ?? '5433';
const database = process.env.MLPILOT_E2E_PG_DB ?? 'demo';
const roUser = process.env.MLPILOT_E2E_PG_RO_USER ?? 'mlpilot_ro';
const roPassword = process.env.MLPILOT_E2E_PG_RO_PASSWORD ?? '';
const rwUser = process.env.MLPILOT_E2E_PG_RW_USER ?? 'mlpilot_rw';
// The RW password reaches the backend as an environment variable (the "password variable" option).
const rwPasswordVar = 'DEMO_E2E_RW_PASSWORD';

const TABLES = [
  'customers',
  'customer_status_snapshot',
  'marketing_emails',
  'order_items',
  'orders',
  'products',
  'refunds',
  'sessions',
  'support_tickets',
];

test.skip(!host, 'MLPILOT_E2E_PG_HOST is not set: no demo database to connect to');

async function waitForBackend(page: Page) {
  await expect
    .poll(async () => (await page.request.get('http://127.0.0.1:8000/health').catch(() => null))?.status(), { timeout: 120_000 })
    .toBe(200);
}

async function fillPostgres(page: Page, name: string, user: string) {
  await page.getByRole('radio', { name: 'Postgres' }).click();
  await page.getByLabel('Name', { exact: true }).fill(name);
  await page.getByLabel('Host', { exact: true }).fill(host!);
  await page.getByLabel('Port', { exact: true }).fill(port);
  await page.getByLabel('Database', { exact: true }).fill(database);
  await page.getByLabel('User', { exact: true }).fill(user);
}

test('connect as the read-only role: graph, table details, time column', async ({ page }) => {
  await waitForBackend(page);
  await page.goto('/');
  await page.getByRole('button', { name: 'Connect a database' }).first().click();
  await expect(page.getByRole('heading', { name: 'Connect a database' })).toBeVisible();

  await fillPostgres(page, 'demo-readonly', roUser);
  await page.getByLabel('Password', { exact: true }).fill(roPassword);
  await page.getByRole('button', { name: 'Save & test' }).click();

  // Connected, and the role is recognised as read-only.
  await expect(page.getByTestId('connect-ok')).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId('read-only-ok')).toBeVisible();
  await expect(page.getByTestId('write-warning')).toHaveCount(0);

  // The graph has every demo table, and all nine relationships.
  const graph = page.getByTestId('schema-graph');
  for (const t of TABLES) await expect(graph.getByTestId(`table-${t}`)).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText(`${TABLES.length} tables · 9 relationships`)).toBeVisible();

  // Open a table: columns, statistics (computed in the database) and its relationships.
  await graph.getByTestId('table-orders').click();
  const details = page.getByTestId('table-details');
  await expect(details).toBeVisible();
  await expect(details.getByTestId('col-customer_id')).toBeVisible();
  await expect(details.getByTestId('stats-note')).toBeVisible({ timeout: 60_000 });
  await expect(details.getByText('orders.customer_id → customers.customer_id', { exact: false })).toBeVisible();
  // The column is guessed to be the event time; the user can change it.
  const timeColumn = details.getByLabel('Time column');
  await expect(timeColumn).toHaveValue('ordered_at');
  await expect(graph.getByTestId('table-orders').getByText('time: ordered_at')).toBeVisible();

  // Toggle it off: the node updates, and the choice is stored (it survives opening another table).
  await timeColumn.selectOption('');
  await expect(graph.getByTestId('table-orders').getByText('no time column')).toBeVisible();
  await graph.getByTestId('table-customers').click();
  await expect(details.getByLabel('Time column')).toHaveValue('signup_at');
  await graph.getByTestId('table-orders').click();
  await expect(details.getByLabel('Time column')).toHaveValue('');
  // And back on.
  await details.getByLabel('Time column').selectOption('ordered_at');
  await expect(graph.getByTestId('table-orders').getByText('time: ordered_at')).toBeVisible();

  // Privacy: the customers table holds e-mail addresses; none reaches the page.
  await graph.getByTestId('table-customers').click();
  await expect(details.getByTestId('stats-note')).toBeVisible({ timeout: 60_000 });
  await expect(details.getByTestId('col-email')).toBeVisible();
  expect(await page.locator('body').innerText()).not.toMatch(/[\w.+-]+@[\w-]+\.[\w.]+/);
});

test('a wrong password is reported without echoing it', async ({ page }) => {
  await waitForBackend(page);
  await page.goto('/#/connect');
  await fillPostgres(page, 'demo-wrong-password', roUser);
  await page.getByLabel('Password', { exact: true }).fill('definitely-not-the-password');
  await page.getByRole('button', { name: 'Save & test' }).click();
  const error = page.getByTestId('connect-error');
  await expect(error).toBeVisible({ timeout: 60_000 });
  await expect(error).toContainText(/auth_failed|password|authentication/i);
  expect(await page.locator('body').innerText()).not.toContain('definitely-not-the-password');
  await expect(page.getByTestId('schema-graph')).toHaveCount(0);
});

test('a role that can write gets a visible warning', async ({ page }, testInfo) => {
  await waitForBackend(page);
  await page.goto('/#/connect');
  await fillPostgres(page, 'demo-readwrite', rwUser);
  await page.getByLabel('Read the password from an environment variable instead').check();
  await page.getByLabel('Password variable').fill(rwPasswordVar);
  await page.getByRole('button', { name: 'Save & test' }).click();

  await expect(page.getByTestId('connect-ok')).toBeVisible({ timeout: 60_000 });
  const warning = page.getByTestId('write-warning');
  await expect(warning).toBeVisible();
  await expect(warning).toContainText('this role can write');
  await expect(page.getByTestId('read-only-ok')).toHaveCount(0);
  // The graph still loads: the warning is advice, MLPilot reads either way.
  await expect(page.getByTestId('schema-graph').getByTestId('table-orders')).toBeVisible({ timeout: 60_000 });

  await warning.getByText('How to create a read-only role').click();
  const dir = process.env.MLPILOT_E2E_SCREENSHOT_DIR ?? testInfo.outputDir;
  await page.screenshot({ path: `${dir}/connect-write-warning.png`, fullPage: true });
});

test('a snapshot gets a version id, the same data gets the same one, and live mode says it cannot be reproduced', async ({ page }) => {
  await waitForBackend(page);
  await page.goto('/#/connect');
  await fillPostgres(page, 'demo-snapshot', roUser);
  await page.getByLabel('Password', { exact: true }).fill(roPassword);
  await page.getByRole('button', { name: 'Save & test' }).click();
  await expect(page.getByTestId('connect-ok')).toBeVisible({ timeout: 60_000 });
  const panel = page.getByTestId('data-versions');
  await expect(panel).toBeVisible({ timeout: 60_000 });

  const versions = panel.getByTestId('db-version');
  await panel.getByRole('button', { name: 'Take snapshot' }).click();
  await expect(versions).toHaveCount(1, { timeout: 120_000 });
  await expect(versions.first()).toContainText(/[0-9a-f]{12}/);
  await expect(versions.first()).toContainText('snapshot');
  await expect(versions.first()).toContainText('reproducible');
  const first = (await versions.first().innerText()).match(/[0-9a-f]{12}/)![0];

  // Nothing changed in the database, so the second snapshot is the same version, listed once.
  await expect(panel.getByRole('button', { name: 'Take snapshot' })).toBeEnabled({ timeout: 120_000 });
  await panel.getByRole('button', { name: 'Take snapshot' }).click();
  await expect(panel.getByRole('button', { name: 'Take snapshot' })).toBeEnabled({ timeout: 120_000 });
  await expect(versions).toHaveCount(1);
  await expect(versions.first()).toContainText(first);

  // Live mode copies nothing and says so.
  await panel.getByLabel('Version mode').selectOption('live');
  await panel.getByRole('button', { name: 'Record live state' }).click();
  await expect(versions).toHaveCount(2, { timeout: 60_000 });
  await expect(panel.getByText('not reproducible')).toBeVisible();
});
