/**
 * Browser smoke test. Run the preview server first (`npm run build && npm run preview`), then `npm run e2e`.
 * Set PW_EXECUTABLE to point at a Chromium binary if Playwright's default browser is not installed.
 */
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const S = path.join(here, 'output');
fs.mkdirSync(S, { recursive: true });
const BASE = process.env.E2E_BASE_URL ?? 'http://localhost:4173/';
const browser = await chromium.launch(process.env.PW_EXECUTABLE ? { executablePath: process.env.PW_EXECUTABLE } : {});
const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
page.on('console', (m) => { if (m.type() === 'error' && !m.text().startsWith('Failed to load resource')) errors.push('console: ' + m.text()); });
const log = (...a) => console.log('✔', ...a);
const fail = (msg) => { console.error('✘', msg); process.exitCode = 1; };
const expectText = async (text, where = page) => { try { await where.getByText(text, { exact: false }).first().waitFor({ timeout: 5000 }); log('found', JSON.stringify(text)); } catch { fail('missing text ' + JSON.stringify(text)); } };

await page.goto(BASE);
await page.getByRole('button', { name: 'Open synthetic demo' }).click();
await expectText('DEMO — Synthetic sample appraisal');
await expectText('$325,125');
await page.screenshot({ path: `${S}/shot-cards.png` });

// Activate area rule and set rate
const ruleBox = page.locator('div.rounded-md', { has: page.getByRole('button', { name: /^Above-grade finished area/ }) }).first();
await ruleBox.locator('label:has(input[role=switch])').click(); if (!(await ruleBox.getByRole('switch').isChecked())) fail('toggle did not activate');
await ruleBox.getByRole('button', { name: /^Above-grade finished area/ }).click();
await ruleBox.locator('input[id$="-rate"]').fill('50');
await expectText('$321,500');
await expectText('$310,000'); // comp 1 adjusted on card
log('live recalculation works');

// Select comp 1 card and check details
await page.getByRole('button', { name: /102 Example Way/ }).first().click();
const detail = page.getByLabel('Property details and reconciliation');
await detail.getByRole('button', { name: /Above-grade finished area/ }).click();
await expectText('(2,000 − 1,800) × $50.00 = +$10,000');
await page.screenshot({ path: `${S}/shot-details.png` });

// Override on the selected comp's area line
await detail.getByRole('button', { name: 'Override' }).first().click();
await detail.locator('input[id$="-amt"]').fill('7500');
await detail.getByLabel('Reason').fill('Paired sales');
await detail.getByRole('button', { name: 'Save override' }).click();
await expectText('Manual override', detail);
await expectText('$307,500');
log('override replaces computed adjustment');

// Grid view
await page.getByRole('tab', { name: 'Comparison grid' }).click();
await expectText('Original transaction price');
await page.screenshot({ path: `${S}/shot-grid.png` });
// Charts
await page.getByRole('tab', { name: 'Charts' }).click();
await expectText('Adjustment waterfall');
await page.waitForTimeout(500);
await page.screenshot({ path: `${S}/shot-charts.png` });
await page.getByRole('tab', { name: 'Cards' }).click();

// Import flow
await page.getByRole('button', { name: /Import/ }).first().click();
await page.getByRole('menuitem', { name: /CSV/ }).click();
await page.locator('input[type=file]').setInputFiles(path.join(here, 'fixtures', 'harmls_sample.csv'));
await expectText('Map fields');
await expectText('Lot sizes have no units');
await page.getByLabel('Lot size units').selectOption('auto');
await page.getByLabel('Date order').selectOption('MDY');
await page.getByRole('button', { name: 'Preview' }).click();
await expectText('400 Harbor Way');
await expectText('Missing values (kept as unknown)');
await page.getByRole('button', { name: 'Check duplicates' }).click();
await expectText('Matched by MLS number');
const dupBoxes = page.locator('div.rounded-md.border-amber-200 input[type=checkbox]');
await dupBoxes.nth(1).uncheck();
await page.getByRole('button', { name: /Import 3 records/ }).click();
await expectText('Imported 3 records');
await expectText('11 comps');
log('import with duplicate review works');
await expectText('17 Marina Ct'); // listing in market context

// Undo import
await page.keyboard.press('Control+z');
await expectText('8 comps');
await page.keyboard.press('Control+Shift+z');
await expectText('11 comps');
log('undo/redo works');

// Export grid CSV
const [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: /Export/ }).click().then(() => page.getByRole('menuitem', { name: 'Adjustment grid (CSV)' }).click())]);
const csvPath = `${S}/grid.csv`;
await dl.saveAs(csvPath);
const csv = fs.readFileSync(csvPath, 'utf8');
if (csv.includes('Adjusted sale price') && csv.includes('Weighted indication')) log('grid CSV exported'); else fail('grid CSV content');
const [dl2] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: /Export/ }).click().then(() => page.getByRole('menuitem', { name: 'Project backup (JSON)' }).click())]);
await dl2.saveAs(`${S}/backup.json`);
const backup = JSON.parse(fs.readFileSync(`${S}/backup.json`, 'utf8'));
if (backup.scenarios?.[0]?.comparables?.length === 11) log('JSON backup exported'); else fail('backup content');

// Report
await page.getByRole('button', { name: /Export/ }).click();
await page.getByRole('menuitem', { name: 'Print-ready report' }).click();
await expectText('Sales comparison summary');
await page.waitForTimeout(500);
await page.screenshot({ path: `${S}/shot-report.png`, fullPage: true });
await page.getByRole('button', { name: 'Back to dashboard' }).click();

// Scenario duplicate + compare
page.once('dialog', (d) => d.accept('Alt rates'));
await page.getByRole('button', { name: /Scenario:/ }).click();
await page.getByRole('menuitem', { name: /Duplicate current scenario/ }).click();
await expectText('Scenario: Alt rates');
await page.getByRole('button', { name: /Scenario:/ }).click();
await page.getByRole('menuitem', { name: /Compare scenarios/ }).click();
await expectText('Scenario comparison');
await page.screenshot({ path: `${S}/shot-compare.png` });
await page.keyboard.press('Escape');

// Persistence: reload
await page.waitForTimeout(1500);
await page.reload();
await expectText('DEMO — Synthetic sample appraisal');
await expectText('Scenario: Alt rates');
await expectText('11 comps');
log('IndexedDB persistence works');

// Mobile layout
await page.setViewportSize({ width: 390, height: 844 });
await page.waitForTimeout(300);
await page.screenshot({ path: `${S}/shot-mobile.png` });
await page.getByRole('tab', { name: 'Details & value' }).click();
await expectText('Weighted value indication');
await page.setViewportSize({ width: 1024, height: 768 });
await page.waitForTimeout(300);
await page.screenshot({ path: `${S}/shot-tablet.png` });

if (errors.length) { console.error('Browser errors:'); errors.forEach((e) => console.error('  ', e)); process.exitCode = 1; } else log('no browser errors');
await browser.close();
