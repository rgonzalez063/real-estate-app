/** Opens the self-contained HTML from disk (file://) and checks it runs and persists. */
import { chromium } from 'playwright';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const file = 'file://' + path.resolve(here, '..', 'release', 'Gonzalez-Appraisal-Dashboard.html');
const browser = await chromium.launch(process.env.PW_EXECUTABLE ? { executablePath: process.env.PW_EXECUTABLE } : {});
const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
await page.goto(file);
await page.getByRole('button', { name: 'Open synthetic demo' }).click();
await page.getByText('$325,125').first().waitFor({ timeout: 5000 });
await page.getByRole('button', { name: /Import/ }).first().click();
await page.getByRole('menuitem', { name: /CSV/ }).click();
await page.locator('input[type=file]').setInputFiles(path.join(here, 'fixtures', 'harmls_sample.csv'));
await page.getByText('Map fields').first().waitFor();
await page.keyboard.press('Escape');
await page.waitForTimeout(1600); // autosave
await page.reload();
await page.getByText('DEMO — Synthetic sample appraisal').first().waitFor({ timeout: 5000 });
await page.screenshot({ path: path.join(here, 'output', 'single-file.png') });
console.log(errors.length ? 'ERRORS: ' + errors.join('\n') : 'single-file build OK: runs from disk, import opens, project persisted across reload');
await browser.close();
process.exitCode = errors.length ? 1 : 0;
