/** Browser check for the document importer's offline path: PDF text scan -> review -> add as subject and as comp. */
import { chromium } from 'playwright';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const BASE = process.env.E2E_BASE_URL ?? 'http://localhost:4173/';
const browser = await chromium.launch(process.env.PW_EXECUTABLE ? { executablePath: process.env.PW_EXECUTABLE } : {});
const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
const ok = (m) => console.log('✔', m);
const fail = (m) => { console.error('✘', m); process.exitCode = 1; };
await page.goto(BASE);
await page.getByRole('button', { name: 'Open synthetic demo' }).click();
await page.getByText('$325,125').first().waitFor();

// Local PDF scan -> comparable
await page.getByRole('button', { name: /Import/ }).first().click();
await page.getByRole('menuitem', { name: /PDF \/ image documents/ }).click();
await page.getByText('Import from PDF or image documents').waitFor();
await page.getByText('Local text scan').click();
await page.getByRole('dialog').locator('input[type=file]').setInputFiles(path.join(here, 'fixtures', 'mls_sheet_sample.pdf'));
await page.getByRole('button', { name: 'Scan text locally' }).click();
await page.getByText('912 Lighthouse Rd').first().waitFor({ timeout: 20000 });
ok('local PDF scan extracted the address');
const soldCell = await page.getByText('$401,500').count();
if (soldCell > 0) ok('sold price read from PDF'); else fail('sold price missing');
await page.getByRole('button', { name: /Evidence/ }).click();
await page.getByText('“Sold Price: $401,500”').waitFor();
ok('evidence quotes shown');
await page.screenshot({ path: path.join(here, 'output', 'doc-review.png') });
await page.getByRole('button', { name: /Add 1 comparable/ }).click();
await page.getByText('Added 1 comparable').waitFor();
await page.getByRole('tab', { name: 'Cards' }).click();
await page.getByText('912 Lighthouse Rd').first().waitFor();
ok('record added as comparable');

// Local PDF scan -> subject
await page.getByRole('button', { name: /Import/ }).first().click();
await page.getByRole('menuitem', { name: /PDF \/ image documents/ }).click();
await page.getByText('Local text scan').click();
await page.getByRole('dialog').locator('input[type=file]').setInputFiles(path.join(here, 'fixtures', 'mls_sheet_sample.pdf'));
await page.getByLabel('What do these documents describe?').selectOption('subject');
await page.getByRole('button', { name: 'Scan text locally' }).click();
await page.getByRole('button', { name: /\+ subject/ }).waitFor({ timeout: 20000 });
await page.getByRole('button', { name: /\+ subject/ }).click();
await page.getByText('updated the subject').waitFor();
const subjAddr = await page.locator('#subject-address').inputValue();
if (subjAddr.includes('912 Lighthouse Rd')) ok('subject updated from PDF'); else fail('subject address not updated: ' + subjAddr);
await page.getByText(/Imported from mls_sheet_sample.pdf/).waitFor();

// CSV -> subject
await page.getByRole('button', { name: /Import/ }).first().click();
await page.getByRole('menuitem', { name: /CSV/ }).click();
await page.getByRole('dialog').getByRole('radio', { name: /Subject property/ }).check();
await page.getByRole('dialog').locator('input[type=file]').setInputFiles(path.join(here, 'fixtures', 'harmls_sample.csv'));
await page.getByLabel('Lot size units').selectOption('sqft');
await page.getByLabel('Date order').selectOption('MDY');
await page.getByRole('button', { name: 'Preview' }).click();
await page.getByLabel('Use row 2 as subject').check();
await page.getByRole('button', { name: 'Use selected row as subject' }).click();
await page.getByText(/Subject property updated from harmls_sample.csv/).waitFor();
const subjAddr2 = await page.locator('#subject-address').inputValue();
if (subjAddr2.includes('402 Harbor Way')) ok('subject updated from CSV row'); else fail('CSV subject not applied: ' + subjAddr2);

// AI path shows key requirement without a key
await page.getByRole('button', { name: /Import/ }).first().click();
await page.getByRole('menuitem', { name: /PDF \/ image documents/ }).click();
await page.getByRole('dialog').locator('input[type=file]').setInputFiles(path.join(here, 'fixtures', 'mls_sheet_sample.pdf'));
await page.getByText('AI extraction with Claude').click();
await page.locator('#ai-key').fill('');
const disabled = await page.getByRole('button', { name: 'Extract with Claude' }).isDisabled();
if (disabled) ok('AI extraction requires an API key'); else fail('AI button enabled without key');
await page.keyboard.press('Escape');

await page.screenshot({ path: path.join(here, 'output', 'doc-import.png') });
if (errors.length) { console.error('Browser errors:', errors); process.exitCode = 1; } else ok('no browser errors');
await browser.close();
