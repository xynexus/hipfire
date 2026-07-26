const { test, expect } = require('@playwright/test');

const base = 'http://127.0.0.1:18081/admin/ui/';

const user = {
  id: 'usr_01JTESTADA000000000000', name: 'Ada Research', status: 'enabled',
  rate_policy: { requests_per_minute: 120, request_burst: 30, text_tokens_per_minute: 80000,
    text_token_burst: 20000, max_in_flight_text: 4, max_in_flight_images: 1,
    megapixel_steps_per_minute: 40, megapixel_step_burst: 10, max_in_flight_training: 1 },
  token_count: 1, created_at: 1, updated_at: 1,
};
const token = {
  id: 'tok_01JPRODUCTION000000000', user_id: user.id, label: 'production-client',
  scopes: ['text', 'embeddings'], rate_policy: {}, created_at: 1, expires_at: 1999999999,
  revoked_at: null,
};

async function mockAdmin(page, state = { authorized: true }) {
  await page.route('**/admin/**', async route => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname.startsWith('/admin/ui/')) return route.fallback();
    if (!state.authorized && url.pathname !== '/admin/login') {
      return route.fulfill({ status: 401, contentType: 'application/json', body: JSON.stringify({ error: { message: 'admin authentication required' } }) });
    }
    if (url.pathname === '/admin/login') {
      state.authorized = true;
      return route.fulfill({ status: 200, contentType: 'application/json', body: '{"status":"ok"}' });
    }
    if (url.pathname === '/admin/stats') {
      return route.fulfill({ json: { generated_unix: 1, gpus: [{ card: 'card0', busy_percent: 67, vram_used_bytes: 12884901888, vram_total_bytes: 25769803776, vis_vram_used_bytes: null, vis_vram_total_bytes: null, gtt_used_bytes: 2147483648, gtt_total_bytes: 68719476736, temp_c: 61.4, power_w: 184.2, sclk_mhz: 2100, metrics: null }], host: { total_bytes: 137438953472, available_bytes: 85899345920 }, clients: [], npus: [] } });
    }
    if (url.pathname === '/admin/access/users' && request.method() === 'GET') return route.fulfill({ json: { items: [user, { ...user, id: 'usr_02JTESTOPS000000000000', name: 'Operations', status: 'disabled', token_count: 0 }], next_cursor: null } });
    if (url.pathname === '/admin/access/users' && request.method() === 'POST') return route.fulfill({ status: 201, json: { ...user, id: 'usr_new', name: 'New client', token_count: 0 } });
    if (url.pathname === `/admin/access/users/${user.id}` && request.method() === 'GET') return route.fulfill({ json: user });
    if (url.pathname === `/admin/access/users/${user.id}` && request.method() === 'PATCH') return route.fulfill({ json: { ...user, ...request.postDataJSON() } });
    if (url.pathname === `/admin/access/users/${user.id}/tokens` && request.method() === 'GET') return route.fulfill({ json: { items: [token], next_cursor: null } });
    if (url.pathname === `/admin/access/users/${user.id}/tokens` && request.method() === 'POST') return route.fulfill({ status: 201, json: { token: { ...token, id: 'tok_new', label: request.postDataJSON().label, scopes: request.postDataJSON().scopes }, secret: 'hfr_tok_new_4VbY8R0JgQeV7kR3' } });
    if (url.pathname.startsWith('/admin/access/tokens/') && request.method() === 'DELETE') return route.fulfill({ json: { revoked: true } });
    if (url.pathname === '/admin/access/usage') return route.fulfill({ json: { rows: { items: [
      { hour_start: 1783699200, user_id: user.id, token_id: token.id, workload: 'text', counters: { requests: 420, errors: 3, rate_limit_hits: 8, input_tokens: 81000, output_tokens: 22000, cache_tokens: 12000, images: 0, megapixel_steps: 0, training_seconds: 0 } },
      { hour_start: 1783702800, user_id: user.id, token_id: token.id, workload: 'text', counters: { requests: 560, errors: 1, rate_limit_hits: 2, input_tokens: 104000, output_tokens: 31000, cache_tokens: 17000, images: 0, megapixel_steps: 0, training_seconds: 0 } }
    ], next_cursor: null }, totals: { requests: 980, errors: 4, rate_limit_hits: 10, input_tokens: 185000, output_tokens: 53000, cache_tokens: 29000, images: 0, megapixel_steps: 0, training_seconds: 0 } } });
    if (url.pathname === '/admin/access/rate-limits') return route.fulfill({ json: { items: [{ user_id: user.id, token_id: token.id, effective_policy: { requests_per_minute: 120, request_burst: 30, text_tokens_per_minute: 80000, text_token_burst: 20000, max_in_flight_text: 4, max_in_flight_images: 1, megapixel_steps_per_minute: 40, megapixel_step_burst: 10, max_in_flight_training: 1 }, request_remaining: 27, text_token_remaining: 15400, active_text: 2, active_images: 0, active_training: 0 }], next_cursor: null } });
    return route.fallback();
  });
}

test('expired session presents login and recovers', async ({ page }) => {
  const state = { authorized: false };
  await mockAdmin(page, state);
  await page.goto(base);
  await expect(page.getByRole('heading', { name: 'Admin sign in' })).toBeVisible();
  await page.getByLabel('User').fill('admin');
  await page.getByLabel('Password').fill('test-password');
  await page.getByRole('button', { name: 'Sign in' }).click();
  await expect(page.getByRole('heading', { name: 'System overview' })).toBeVisible();
});

test('access and usage workflows remain clear at desktop and mobile sizes', async ({ page }) => {
  await mockAdmin(page);
  await page.goto(base);
  const accessTab = page.getByRole('button', { name: 'Access' });
  await accessTab.focus();
  await expect(accessTab).toBeFocused();
  expect(await accessTab.evaluate(element => getComputedStyle(element).outlineStyle)).not.toBe('none');
  await accessTab.press('Enter');
  await expect(page.getByRole('heading', { name: 'API access' })).toBeVisible();
  await page.getByRole('button', { name: /Ada Research/ }).click();
  await expect(page.getByRole('heading', { name: 'Workload limits' })).toBeVisible();
  await page.getByLabel('Token label').fill('batch-runner');
  await page.getByRole('checkbox', { name: 'Images' }).check();
  await page.getByRole('button', { name: 'Generate token' }).click();
  await expect(page.getByText('hfr_tok_new_4VbY8R0JgQeV7kR3')).toBeVisible();
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({ path: '../../docs/screenshots/2026-07-10-admin-access.png', fullPage: true });
  await page.getByRole('button', { name: 'I saved it' }).click();
  await page.getByRole('button', { name: 'Usage' }).click();
  await expect(page.getByRole('heading', { name: 'Usage & limits' })).toBeVisible();
  await expect(page.getByText('980', { exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Current rate state' })).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({ path: '../../docs/screenshots/2026-07-10-admin-usage-mobile.png', fullPage: true });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
  const bodyContrast = await page.evaluate(() => {
    const parse = value => value.match(/[\d.]+/g).slice(0, 3).map(Number);
    const luminance = rgb => {
      const channels = rgb.map(value => { value /= 255; return value <= .04045 ? value / 12.92 : ((value + .055) / 1.055) ** 2.4; });
      return .2126 * channels[0] + .7152 * channels[1] + .0722 * channels[2];
    };
    const style = getComputedStyle(document.body);
    const [a, b] = [luminance(parse(style.color)), luminance(parse(style.backgroundColor))];
    return (Math.max(a, b) + .05) / (Math.min(a, b) + .05);
  });
  expect(bodyContrast).toBeGreaterThanOrEqual(4.5);
});
