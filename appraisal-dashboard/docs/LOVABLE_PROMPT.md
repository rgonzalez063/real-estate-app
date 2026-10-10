# Lovable prompt: Gonzalez Appraisal Workbench

How to use this file
1. In Lovable, attach your three screenshots (the Excel-style sales grid, the "Subject property vs Comparable sales" graphic, and the dashboard you liked). Lovable can see them; tell it they are layout references.
2. Paste everything between "BEGIN MASTER PROMPT" and "END MASTER PROMPT" as your first message.
3. When the first build works, send the four follow-up prompts at the bottom, one at a time.

---

## BEGIN MASTER PROMPT

Build **Gonzalez Appraisal Workbench**, a web app that lets a residential real-estate appraiser do the sales comparison approach as fast and as obviously as filling in an appraisal form. The appraiser imports comparable sales, places them in a grid next to the subject, types a dollar adjustment for each difference, and immediately sees every sale's adjusted price and a suggested value.

Stack: React, TypeScript, Vite, Tailwind, shadcn/ui, Recharts. Supabase for auth, Postgres, file storage and edge functions. Attached screenshots are layout references; follow the Excel-style grid screenshot most closely.

### 0. Design principle

The Sales Grid IS the product. Everything else exists to fill it or to read its result. A first-time user must be able to get from "new appraisal" to "suggested value" without reading instructions. Prefer one clear screen at a time over many panels. Use appraiser words (Subject, Sale #1, Adjustment, Adjusted sale price, Gross living area, Indicated value), not software words (rules, overrides, scenarios, entities).

### 1. Guided flow with a persistent stepper

A four-step stepper is always visible at the top: **1 Subject, 2 Comparables, 3 Adjust, 4 Value**. Each step is one full-width screen with one primary button ("Next: Comparables"). Users may jump to any step. Each step shows a small status chip ("Subject: address and GLA entered", "4 sales selected", "2 adjustments need input", "Value: $325,125"). The home screen lists saved appraisals with "New appraisal" and "Open demo appraisal" (synthetic data, clearly labelled DEMO everywhere it appears).

A slim always-visible bar under the stepper shows: **Suggested value**, adjusted price range, number of sales used, autosave status ("Saved 10:42"), Undo and Redo. Nothing needs a Calculate button; every edit updates instantly.

Empty states always offer the next action as a big button (for example "Import sales from CSV or PDF").

### 2. Step 1: Subject

A single clean form, grouped like the appraisal form: Address, effective date, neighborhood; Site (area and units sf/acres); Gross living area; Basement and finished rooms below grade; Other area; Bedrooms, full baths, half baths (always separate); Year built; Quality (Q1-Q6), Condition (C1-C6), Location, View, Design/style; Garage and carport spaces; Pool, waterfront; Functional utility, heating/cooling, energy items, porch/patio/deck, other amenities; optional latitude/longitude; photo (drag and drop); appraiser notes.
Top of the screen: "Import subject from CSV or PDF" (same importer as step 2). Blank means unknown. Never convert blank to zero.

### 3. Step 2: Comparables

One import button, **Import**, accepting CSV, PDF and images in one drop zone; it detects the type.
- **CSV**: show a compact mapping table (source header, sample values, mapped field with a confidence badge). Auto-suggest from common MLS headers (HARMLS/Matrix, "GLA" = gross living area above grade). Ask explicitly when ambiguous: day/month order, lot units (sf or acres), combined baths such as "3.1" (never guess; ask for the convention). Preview rows, flag duplicates (same MLS number, or same address and date) and let the user keep or drop each. Saved mapping presets. Keep the original row and file name on every record. Offer "Use as subject" for one row.
- **PDF or photo**: send to an edge function (see section 8). Show a review screen per extracted property with the values read, a confidence badge and the quoted source text for each value; the user marks each Comparable, Subject or Skip and can edit before anything is added.

Then a **sales browser**: a dense table (default) and a card view toggle. Columns: checkbox, photo, address, status, sale date, price, GLA, $/SF (state the area basis), lot, beds/baths, year built, distance (only if data exists). Filters in a left drawer: status, dates, price, GLA, lot, neighborhood, quality, condition, pool, garage, distance. Sort on any column. A checkbox adds a sale to the grid; show "4 of 6 recommended maximum selected". Only **closed** sales count toward value. Active and pending listings appear under "Market context" with a clear badge and can never enter the value. Never auto-select comps. Each sale has an optional "why included or excluded" note.

### 4. Step 3: Adjust (the Sales Grid)

Match the Excel-style screenshot. Horizontal scroll with sticky left columns.

**Left:** a sticky label column, then a sticky **Subject** column.
**Right:** one block per selected sale (up to 8), each block has two sub-columns: **Description** and **+(-) $ Adjustment**.

Block header per sale: "Sale #n", photo drop box ("Drag photo here"), Source, Price, Gross adjustment %, Net adjustment %, Indicated value (adjusted price), Weight with its normalized share, neighborhood, address, distance from subject, an Included toggle, Remove.
Subject header: photo, address, neighborhood, and the suggested value.

Rows in this order, each with an editable description for subject and every sale and an adjustment cell for each sale: Sale price; Sale price per gross living area (computed); Data sources; then a "Value adjustments" band: Sales or financing concessions; Date of sale/time; Location; Leasehold/fee simple; Site; View; Waterfront; Design (style); Quality of construction; Actual age; Condition; Above grade bedrooms, full baths, half baths; Gross living area; Basement and finished rooms below grade; Other area; Functional utility; Heating/cooling; Energy efficient items; Garage; Carport; Porch/patio/deck; Pool; Other amenities/fence. Allow adding custom rows.

Footer rows: **Net adjustment (total)** with +/- markers, **Net % and Gross %**, **Adjusted sale price of comparable**, Notes, Data source and effective date.

**How adjusting works (this is the key to being intuitive):**
- The adjustment cell is a plain number input. The appraiser types a signed dollar amount. Positive means the sale is inferior to the subject on that item (add). Negative means the sale is superior (subtract). Put one line explaining this above the grid.
- Where the subject and a sale have different descriptions and the adjustment is empty, show a small amber dot in that cell ("Differs, no adjustment entered") and count these in a footer chip with a "Jump to next difference" button. Empty with a difference is a warning, not an error.
- If either side's description is unknown (blank), that cell is **Needs input**: the sale is excluded from the value until the user types an amount, clicks "Confirm $0", or clicks "N/A". Show these in amber and list the count per sale. Never silently treat unknown as zero.
- Clicking a row label opens a small popover: **Set a rate for this row** (for example "$50 per sf of GLA"). It pre-fills the adjustment cells with `(Subject - Sale) x rate`, shown in teal as suggestions with the formula on hover, and a field for the support note ("paired sales, 2026-Q3") and an attachment reference. Typing a number in any cell replaces the suggestion for that cell only (one adjustment per cell, never added twice). Clearing the cell returns to the suggestion. Every rate starts blank and nothing is pre-filled; never ship invented market rates.
- Categorical rows (quality, condition, location, view): the popover lets the appraiser enter a dollar value per rating (Q3 = $X, Q4 = $Y). Do not assume equal steps between ratings.
- Date of sale/time row: popover with two choices: type dollars per sale, or use a monthly percentage rate with a start-date choice (sale date or contract date). Show start date, effective date, elapsed months, and the formula. Elapsed months = date-only day difference / 30.4375. Adjustment = base price x ((1 + monthly rate)^elapsed months - 1), base is the original sale price. Do not derive a rate from the data.
- Concessions row: show the reported concession as source data in the description cell. The adjustment is whatever the appraiser types (or a factor they set); never deduct automatically.
- Keyboard: Tab moves right, Enter moves down, Shift+Enter up, arrow keys when not editing. Pasting a column of numbers fills down. Numbers show thousands separators when not focused and must never reformat while typing.
- Warn (do not block) about likely double counting, for example age plus condition, or two area rows active.
- Optional review flags with user-editable thresholds (net over 15%, gross over 25%, single line over 10%, outlier vs median 15%), labelled "your review thresholds".

Below the grid, the **Suggested value** panel: weighted indication in large type, adjusted range, unadjusted range, mean and median, number of sales used and number excluded with the reason. Button "Use as final opinion". An editable final reconciled opinion field with rationale. Say "observed range of adjusted prices", never "confidence interval".

On small screens the grid becomes a horizontally swipeable table with the label column pinned, plus a "one sale at a time" compare view (subject beside a single sale).

### 5. Step 4: Value

A clean summary page for the person who will sign: the grid in read-only form, charts, weights table (nonnegative weights, default equal, auto-normalized, zero total weight shows a clear message and no value), final opinion and rationale, reconciliation notes, and an export area.
Charts (Recharts, accessible, not color-only): before/after price per sale; adjustment waterfall for the selected sale; adjusted price distribution with markers for the weighted indication and final opinion; price vs GLA scatter; price timeline. Clicking any sale in any chart selects it everywhere and opens a slide-over with its details.
Exports: print-ready report (browser print to PDF) with subject, assumptions and rates with their support notes, the grid, weights, reconciliation and charts; adjustment-grid CSV that protects against spreadsheet formula injection (prefix text starting with = + - @ with an apostrophe, keep real numbers numeric); full JSON backup and restore. Exports must match the screen and list anything unresolved.

### 6. Calculation rules (single shared engine, unit tested)

- Adjustment for a quantity = (Subject quantity - Sale quantity) x Rate.
- Adjusted sale price = original sale price + sum of signed adjustments.
- Net adjustment = sum of signed adjustments. Gross adjustment = sum of absolute values. Net % and Gross % are relative to the original sale price.
- Weighted indication = sum(adjusted price x weight) / sum(weights), over included, calculable closed sales. No selections or zero total weight shows a message and no number.
- Keep full precision internally. Round only for display and export, using a visible rounding setting (nearest $1, $10, $100, $1,000).
- Dates are calendar dates with no time zone.
- Unit conversion: 1 acre = 43,560 sf. Keep unknown values as null everywhere.
- Required test: subject GLA 2,000 sf, Sale A GLA 1,800 sf at $300,000, rate $50/sf gives +$10,000 and an adjusted price of $310,000. A second sale with an adjusted price of $330,000 at weights 60% and 40% gives a weighted indication of $318,000. Add tests for override-replaces-rule, sign direction, net vs gross, missing values, zero weight, CSV parsing, acre conversion, date math and backup/restore.

### 7. Data model (Supabase Postgres, row-level security per user)

`appraisals` (id, user_id, name, effective_date, settings jsonb, final_value, final_rationale, notes), `subjects` (appraisal_id, attributes jsonb, photo_path, source), `sales` (id, appraisal_id, attributes jsonb, status, original_row jsonb, source_file, included, weight, note, photo_path, extraction jsonb), `adjustment_rows` (appraisal_id, key, label, kind, rate, rating_values jsonb, market_config jsonb, support_note, attachment_ref), `adjustments` (sale_id, row_key, amount, source 'typed' or 'rate', resolution 'zero' or 'na' or null, reason), `mapping_presets`. Autosave with debounce. Support multiple named versions of an appraisal ("what if" copies) that can be duplicated and compared side by side, and undo/redo for edits.

### 8. AI document extraction (server side only)

Create an edge function `extract-document` that receives a PDF or image from storage and calls the Claude API (or the platform's AI gateway) with a structured JSON schema covering the property fields. The API key lives in Supabase secrets, never in the browser. The system instructions: transcribe only what the document states, never estimate or infer, keep full and half baths separate, return values as written with units, one record per property, quote the exact source text for every non-null value, give a confidence of high, medium or low. Return records for the review screen; nothing is saved until the user confirms. Also provide an offline fallback that reads the text layer of a PDF in the browser and matches labels like "Sold Price" and "Year Built".

### 9. Quality and safety

Keep imported data private to the user. Sanitize imported text, validate photo URLs (http/https only), cap image size. Accessible: labels on every input, visible focus, full keyboard use, color never the only signal (also use icons or text). Never use a hidden or off-screen input in a way that can scroll the page when focused; keep the page itself from scrolling and let only the panels scroll. Add a top-level error boundary so a rendering error shows a message and a Try again button instead of a blank page. Format numbers on blur only. Responsive for desktop, tablet and phone. No placeholder buttons: if a feature is not built, do not show it.

### 10. Visual design

Navy (#13233d), slate neutrals, white, and one restrained teal accent (#2a958a). Dense, calm, spreadsheet-like grid with 28px rows, 12px text, thin borders; large readable numbers for the suggested value. Teal tint = value suggested by a rate, white = typed by the appraiser, amber = needs attention, grey = confirmed $0 or N/A. Add short helper text where a concept is new, and hover tooltips that show formulas.

### 11. Acceptance checklist (verify before finishing)

1. A new user can create an appraisal, enter a subject, import a CSV, tick four sales, type adjustments, and read a suggested value without leaving the four-step flow.
2. Typing +10,000 in Sale A's GLA cell shows $310,000 as its adjusted price and updates the suggested value instantly.
3. Setting a $50/sf rate on the GLA row fills every sale's GLA cell with teal suggestions; typing in one cell replaces only that cell; clearing it restores the suggestion.
4. A blank description on either side shows Needs input and keeps that sale out of the value until resolved.
5. Listings and pending sales never change the suggested value.
6. Toggling any switch or checkbox anywhere never scrolls or hides the page.
7. Exports match the screen, JSON backup restores exactly, CSV is formula-injection safe.
8. The demo appraisal loads with synthetic data labelled DEMO.

## END MASTER PROMPT

---

## Follow-up prompts (send one at a time after the first build works)

**Follow-up 1: polish the grid.** "Make the Sales Grid feel exactly like the Excel-style screenshot. Tighten the rows, make the two sub-columns per sale line up under each header, keep the label and subject columns sticky, add the amber 'differs, no adjustment' dots and the 'Jump to next difference' chip, and add Tab/Enter/paste-down navigation. Test it with six sales on a laptop screen."

**Follow-up 2: imports.** "Build the importer: one Import button that accepts CSV, PDF and images. For CSV, the compact mapping table with confidence badges, the ambiguity questions, duplicates review and saved presets. For PDF and photos, the extract-document edge function and the review screen with quoted evidence. Allow importing more files at any time and 'Use as subject' for one row."

**Follow-up 3: value and reports.** "Build step 4: weights table, final opinion with rationale, the five charts with click-to-select, the print-ready report, grid CSV (formula-injection safe), and JSON backup and restore. Add named versions of an appraisal with duplicate and side-by-side compare."

**Follow-up 4: harden.** "Add the unit tests listed in the calculation section, including the $310,000 and $318,000 checks. Add the error boundary, keyboard and screen-reader pass, mobile compare-one-sale view, and a synthetic DEMO appraisal. Then walk through the acceptance checklist and report anything that fails."
