# Gonzalez Appraisal Company — Residential Appraisal Dashboard

Interactive, browser-only dashboard for the sales comparison approach: import comparable CSV exports, select comps, apply supported adjustments, and watch the indicated value update in real time.

Everything runs locally in the browser. There is no backend, no account, and no paid API. Projects are saved in IndexedDB and can be exported as JSON backups.

## Easiest way to run: the self-contained file

`release/Gonzalez-Appraisal-Dashboard.html` is the whole app in one file. Download it, double-click it, and it opens in your browser. No install, no server, no internet.

- Works in Chrome, Edge and Firefox. Projects are saved inside that browser (IndexedDB), so keep using the same browser and the same copy of the file to see your saved projects. Use *Export → Project backup (JSON)* for a portable copy you can restore anywhere.
- Safari may clear local storage for files opened from disk; export a backup before closing if you use Safari.
- To rebuild it after code changes: `npm run build:single`.

## Setup for development

Requires Node 18+.

```bash
cd appraisal-dashboard
npm install
npm run dev        # http://localhost:5173
```

Other scripts:

```bash
npm test           # Vitest engine tests (adjustments, reconciliation, parsing, dates, exports, persistence)
npm run typecheck  # tsc
npm run lint       # eslint
npm run build      # production build in dist/
npm run preview    # serve dist/ on http://localhost:4173
npm run e2e        # Playwright browser smoke test against the preview server (needs a Chromium; set PW_EXECUTABLE if not auto-detected)
npm run build:single  # rebuild release/Gonzalez-Appraisal-Dashboard.html (single self-contained file)
npm run e2e:single    # open the single file from disk in a browser and check it runs
npm run e2e:docs      # browser check of the PDF/CSV subject and document import flows (offline path)
npm run e2e:grid      # browser check of typed adjustments on the sales grid and the cards snapshot
```

## How to use

1. **Projects** (top bar) → *New project*, or *Open synthetic demo* for a clearly labelled sample. The demo data is invented and flagged "Demo" on every record.
2. **Import → CSV** → drop one or more MLS exports (HARMLS / Matrix style headers are recognised). Choose *Import as comparables* (records are appended, so you can import additional sales files at any time) or *Import as subject property* (pick one row on the preview step). Review the suggested header mapping, resolve any prompts (ambiguous day/month order, lot-size units, combined-baths convention), preview the parsed rows and missing values, then decide which potential duplicates to keep. Mappings can be saved as reusable presets.
3. **Import → PDF / image documents** → drop MLS sheets, closed-sale reports, public-record printouts or photos of listings. Two extraction methods:
   - **AI extraction with Claude** (default model Claude Opus 5.5; Sonnet 5.5 and Haiku 5.5 selectable). Enter your own Anthropic API key from console.anthropic.com; it is stored only in this browser if you tick *Remember*, and the document is sent to the Claude API for that request only. Reads text and scanned pages, finds several properties in one document, and quotes the exact text each value came from. The request uses structured output so values arrive in the app's field schema, and Anthropic's server-side refusal fallback is enabled.
   - **Local text scan** (offline, no account). Reads the PDF's text layer in the browser and picks up values next to recognised labels. One property per document; does not work on scanned images.

   Every extracted record lands on a review screen with the values read, confidence, parsing notes and evidence quotes. Mark each one *Comparable*, *Subject* or *Skip*, edit values if needed, and only then is anything added. Nothing is estimated: fields the document does not state stay unknown, combined baths are never split without a confirmed convention, and lot sizes without units are flagged.
4. **Subject** (left panel) → address, effective date, area by category, site, rooms, year built, parking, amenities, ratings, optional coordinates/photos and notes. Every change immediately re-runs the adjustments.
5. **Adjustment assumptions** (left panel) → every category starts inactive with a blank rate. Activate a category, choose the compared attribute, and enter a rate you can support, plus reason / source / attachment reference. Dollar-per-unit, flat, percentage, dollar-by-rating (unequal intervals allowed), market conditions (manual or monthly rate on the sale or contract date) and concessions (explicit factor on the reported amount) are supported. Custom categories can be added. Possible overlaps (age + condition, several area rules, etc.) are flagged.
6. **Sales grid** (center, default view) → the appraisal-form layout: the subject in the left column, each included sale as a column with its photo (drag one onto the box), source, price, gross and net %, indicated value and weight, then one row per item (sale price, concessions, date of sale, location, tenure, site, view, design, quality, age, condition, rooms, gross living area, basement, utility, heating, energy, garage, porch, pool, other). Every sale has a *Description* cell (editable) and a *+(−) $ Adjustment* cell: type a signed dollar amount to adjust for the difference, or leave it blank to use the global rule from the left panel (teal cells). Amber cells need input before that sale counts. The footer shows net adjustment with +/− markers, net and gross %, the adjusted sale price, notes and data source, and the **Suggested value** block below gives the weighted indication, adjusted range, mean/median and the final opinion field.
7. **Cards / Charts** (center) → the Cards view opens with a subject-versus-sales snapshot (↑ inferior / ↓ superior markers per adjustment and the estimated market value range), then filterable, sortable property cards for including or excluding sales; listings and pending sales appear as market context only. Clicking a card, grid column or chart point selects the same property everywhere.
8. **Selected property** (right panel) → attributes, source row, import notes, inclusion note, weight, and every adjustment line with its formula and source. Override an amount (replaces the rule for that comp), confirm $0, or mark N/A. Unresolved lines keep a comp out of the value indication until resolved.
9. **Reconciliation summary** (right panel) → unadjusted and adjusted ranges, mean, median, weighted indication (Σ adjusted × weight ÷ Σ weights, weights default to 1), normalized weights, and the separate final reconciled opinion with rationale.
10. **Scenarios** (top bar) → create, duplicate, rename, delete and compare scenarios side by side. Undo/redo (Ctrl+Z / Ctrl+Shift+Z) covers selections and adjustment edits.
11. **Export** → adjustment grid CSV (formula-injection safe), full project JSON (re-importable from *Projects → Restore from backup*), and a print-ready report (browser print → PDF).

## Calculation rules

- `Adjustment = (Subject quantity − Comparable quantity) × Rate` — positive adds to an inferior comparable, negative subtracts from a superior one.
- `Adjusted sale price = Original sale price + Σ signed adjustments`; `Net = Σ signed`; `Gross = Σ |each|`; percentages are relative to the original sale price.
- Market conditions: `Base price × [(1 + monthly rate)^elapsed months − 1]`, elapsed months = date-only day difference ÷ 30.4375, base = original sale price, start = sale or contract date as chosen.
- Reported concessions are source data only; an adjustment is made only with an explicit factor or per-comp override.
- Unknown values stay unknown; nothing is invented or defaulted to zero. Rounding applies to display and exports only.
- Verified by tests: subject 2,000 SF, comp 1,800 SF at $300,000 with $50/SF ⇒ +$10,000 and $310,000; with a second comp at $330,000 and weights 60/40 ⇒ $318,000.

## Layout

`src/engine` holds the single calculation engine shared by cards, grid, charts, summary, report and exports. `src/store` is the Zustand store (undo/redo, autosave to IndexedDB). `src/components` holds the UI. `src/tests` contains the Vitest suite; `e2e/` the Playwright smoke test and a sample HARMLS-style CSV.

## Document extraction notes

- AI extraction calls the Claude API directly from the browser with your key (`@anthropic-ai/sdk`, `client.beta.messages.parse` with a Zod output schema). Usage is billed to your Anthropic account. No key is bundled with the app.
- The local scan uses pdf.js in the browser; its worker is bundled, so it also works in the single-file build.
- Image files (JPG/PNG/GIF/WEBP) are only supported by AI extraction.

## Known limitations

- The print-ready report relies on the browser's print dialog for PDF output; there is no server-side PDF generator.
- Mapping presets are stored per browser (IndexedDB) and are not included in the project JSON unless saved in the project.
- No automatic time-trend derivation from imported data (by design); the monthly rate must be entered by the appraiser.
- Photos are shown only from http(s) URLs present in the data; nothing is fetched or inferred.
