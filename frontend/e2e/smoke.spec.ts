import { expect, test } from '@playwright/test';

/**
 * Load the sample from the landing page and watch the baseline finish, against the real
 * backend. Every API call the page makes must go to the project-scoped /api/v1 routes.
 */
test('sample data loads and the baseline completes through the project API', async ({ page, request }) => {
  const apiCalls: string[] = [];
  page.on('request', (req) => {
    const url = new URL(req.url());
    if (url.port === '8000') apiCalls.push(`${req.method()} ${url.pathname}`);
  });

  // start.py starts the backend next to the UI; wait until it answers.
  const backend = 'http://127.0.0.1:8000';
  await expect.poll(async () => (await request.get(`${backend}/health`).catch(() => null))?.status(), {
    timeout: 120_000,
  }).toBe(200);
  // The API itself refuses a caller without the token; the UI below has it.
  expect((await request.get(`${backend}/api/v1/projects/`)).status()).toBe(401);

  await page.goto('/');
  await page.getByRole('button', { name: /Start with sample data/ }).first().click();

  // Real numbers from the stored data version, with its short hash.
  await expect(page.getByText(/Loaded telecom_churn\.csv \(data version [0-9a-f]{12}\): 7,043 rows, 21 columns/)).toBeVisible({
    timeout: 60_000,
  });
  // The baseline queued on load reaches a final status and becomes the champion.
  await expect(page.getByText(/holds the lead/).first()).toBeVisible({ timeout: 150_000 });

  expect(apiCalls.length).toBeGreaterThan(3);
  const outside = apiCalls.filter((c) => !/ \/api\/v1\/projects(\/|$)/.test(c));
  expect(outside, `calls outside /api/v1/projects: ${outside.join(', ')}`).toEqual([]);
  expect(apiCalls).toContain('POST /api/v1/projects/');
  expect(apiCalls.some((c) => /^POST \/api\/v1\/projects\/[^/]+\/datasets\/sample$/.test(c))).toBe(true);
  expect(apiCalls.some((c) => /^POST \/api\/v1\/projects\/[^/]+\/experiments\/baseline$/.test(c))).toBe(true);
});
