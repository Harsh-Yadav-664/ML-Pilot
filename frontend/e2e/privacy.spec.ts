import { expect, test } from '@playwright/test';

/**
 * What the LLM sees: pick a level, hide a column, then let the app send a prompt (AI auto-clean on
 * the sample, offline stub provider) and read it in the prompt log. Runs against `python start.py`.
 */
test('privacy settings shape the prompt, and the prompt log shows what was sent', async ({ page, request }) => {
  await expect
    .poll(async () => (await request.get('http://127.0.0.1:8000/health').catch(() => null))?.status(), { timeout: 120_000 })
    .toBe(200);
  await page.goto('/');
  await page.getByRole('button', { name: /Start with sample data/ }).first().click();
  await expect(page.getByText(/Loaded telecom_churn\.csv/)).toBeVisible({ timeout: 60_000 });

  // Settings: privacy level and never-send list.
  await page.getByRole('button', { name: /Settings/ }).first().click();
  const levels = page.getByRole('radiogroup', { name: 'What the LLM may see' });
  await expect(levels.getByRole('radio', { name: /Schema and statistics/ })).toHaveAttribute('aria-checked', 'true');
  await levels.getByRole('radio', { name: /Also sample values/ }).click();
  await expect(page.getByRole('alert').filter({ hasText: 'cell values' })).toBeVisible();
  await levels.getByRole('radio', { name: /Schema and statistics/ }).click();
  await expect(page.getByRole('alert').filter({ hasText: 'cell values' })).toHaveCount(0);

  await page.getByLabel('Column to never send').fill('Contract');
  await page.getByRole('button', { name: 'Add', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Remove Contract' })).toBeVisible();

  // The choice is stored by the backend: it is still there after a reload.
  await page.reload();
  await page.getByRole('button', { name: /Settings/ }).first().click();
  await expect(page.getByRole('button', { name: 'Remove Contract' })).toBeVisible();

  // Send a prompt: the AI auto-clean builds one from the dataset profile.
  await page.getByRole('button', { name: /Dashboard/ }).first().click();
  await page.getByRole('button', { name: 'Apply AI auto-clean' }).first().click();
  await expect(page.getByText(/AI cleaning recipe accepted/)).toBeVisible({ timeout: 60_000 });

  // The prompt log lists it; its full text names columns but not the hidden one, and no category label.
  await page.getByRole('button', { name: /Settings/ }).first().click();
  const log = page.getByTestId('prompt-log');
  await log.getByRole('button', { name: 'Refresh' }).click();
  const row = log.getByRole('row').filter({ hasText: 'cleaning.strategy' }).first();
  await expect(row).toBeVisible();
  await expect(row).toContainText('offline stub');
  await row.click();
  const text = (await page.locator('pre').filter({ hasText: 'Table dataset' }).first().innerText()).toLowerCase();
  expect(text).toContain('table dataset');
  expect(text).toContain('paymentmethod');
  expect(text).not.toContain('contract');
  expect(text).not.toContain('month-to-month');
  expect(text).not.toContain('electronic check');
  await expect(page.getByText('columns left out: 1')).toBeVisible();
});
