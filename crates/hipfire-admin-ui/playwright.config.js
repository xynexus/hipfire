const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests',
  workers: 1,
  use: {
    browserName: 'chromium',
    viewport: { width: 1280, height: 900 },
  },
  webServer: {
    command: 'NO_COLOR=true trunk serve --address 127.0.0.1 --port 18081 --no-autoreload',
    url: 'http://127.0.0.1:18081/admin/ui/',
    reuseExistingServer: true,
    timeout: 120000,
  },
});
