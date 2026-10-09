# Gonzalez Appraisal Company — Residential Appraisal Dashboard

Interactive, browser-only dashboard for the sales comparison approach: import comparable CSV exports, select comps, apply supported adjustments, and watch the indicated value update in real time.

Everything runs locally in the browser. There is no backend, no account, and no paid API. Projects are saved in IndexedDB and can be exported as JSON backups.

## Setup

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
```

## How to use

1. **Projects** (top bar) → *New project*, or *Open synthetic demo* for a clearly labelled sample. The demo data is invented and flagged "Demo" on every record.
2. **Import CSV** → drop one or more MLS exports (HARMLS / Matrix style headers are recognised). Review the suggested header mapping, resolve any prompts (ambiguous day/month order, lot-size units, combined-baths convention), preview the parsed rows and missing values, then decide which potential duplicates to keep. Mappings can be saved as reusable presets.
3. **Subject** (left panel) → address, effective date, area by category, site, rooms, year built, parking, amenities, ratings, optional coordinates/photos and notes. Every change immediately re-runs the adjustments.
4. **Adjustment assumptions** (left panel) → every category starts inactive with a blank rate. Activate a category, choose the compared attribute, and enter a rate you can support, plus reason / source / attachment reference. Dollar-per-unit, flat, percentage, dollar-by-rating (unequal intervals allowed), market conditions (manual or monthly rate on the sale or contract date) and concessions (explicit factor on the reported amount) are supported. Custom categories can be added. Possible overlaps (age + condition, several area rules, etc.) are flagged.
5. **Cards / Comparison grid / Charts** (center) → include or exclude comps, filter and sort, read each comp's unadjusted $/SF (above-grade basis), adjusted price, net/gross adjustments and review flags. Closed sales drive the value; listings and pending sales appear as market context only. Clicking a card, grid column or chart point selects the same property everywhere.
6. **Selected property** (right panel) → attributes, source row, import notes, inclusion note, weight, and every adjustment line with its formula and source. Override an amount (replaces the rule for that comp), confirm $0, or mark N/A. Unresolved lines keep a comp out of the value indication until resolved.
7. **Reconciliation summary** (right panel) → unadjusted and adjusted ranges, mean, median, weighted indication (Σ adjusted × weight ÷ Σ weights, weights default to 1), normalized weights, and the separate final reconciled opinion with rationale.
8. **Scenarios** (top bar) → create, duplicate, rename, delete and compare scenarios side by side. Undo/redo (Ctrl+Z / Ctrl+Shift+Z) covers selections and adjustment edits.
9. **Export** → adjustment grid CSV (formula-injection safe), full project JSON (re-importable from *Projects → Restore from backup*), and a print-ready report (browser print → PDF).

## Calculation rules

- `Adjustment = (Subject quantity − Comparable quantity) × Rate` — positive adds to an inferior comparable, negative subtracts from a superior one.
- `Adjusted sale price = Original sale price + Σ signed adjustments`; `Net = Σ signed`; `Gross = Σ |each|`; percentages are relative to the original sale price.
- Market conditions: `Base price × [(1 + monthly rate)^elapsed months − 1]`, elapsed months = date-only day difference ÷ 30.4375, base = original sale price, start = sale or contract date as chosen.
- Reported concessions are source data only; an adjustment is made only with an explicit factor or per-comp override.
- Unknown values stay unknown; nothing is invented or defaulted to zero. Rounding applies to display and exports only.
- Verified by tests: subject 2,000 SF, comp 1,800 SF at $300,000 with $50/SF ⇒ +$10,000 and $310,000; with a second comp at $330,000 and weights 60/40 ⇒ $318,000.

## Layout

`src/engine` holds the single calculation engine shared by cards, grid, charts, summary, report and exports. `src/store` is the Zustand store (undo/redo, autosave to IndexedDB). `src/components` holds the UI. `src/tests` contains the Vitest suite; `e2e/` the Playwright smoke test and a sample HARMLS-style CSV.

## Known limitations

- The print-ready report relies on the browser's print dialog for PDF output; there is no server-side PDF generator.
- Mapping presets are stored per browser (IndexedDB) and are not included in the project JSON unless saved in the project.
- No automatic time-trend derivation from imported data (by design); the monthly rate must be entered by the appraiser.
- Photos are shown only from http(s) URLs present in the data; nothing is fetched or inferred.
