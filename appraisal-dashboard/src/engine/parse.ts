import type { LotUnits, StatusCategory } from '../types';

export interface ParseResult<T> {
  value: T | null;
  issue?: string;
  /** For dates: true when the day/month order could not be determined from the text alone. */
  ambiguous?: boolean;
}

const MAX_TEXT = 4000;

/** Remove control characters, collapse whitespace, trim and cap length. */
export function sanitizeText(input: string | null | undefined): string | null {
  if (input === null || input === undefined) return null;
  // eslint-disable-next-line no-control-regex
  const cleaned = String(input).replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, '').replace(/\s+/g, ' ').trim();
  if (!cleaned) return null;
  return cleaned.length > MAX_TEXT ? cleaned.slice(0, MAX_TEXT) : cleaned;
}

const EMPTY_TOKENS = new Set(['', '-', '--', 'n/a', 'na', 'null', 'none', 'unknown', '?', 'tbd', '#n/a']);

export function isEmptyToken(s: string): boolean {
  return EMPTY_TOKENS.has(s.trim().toLowerCase());
}

/** Parse a numeric value that may include currency symbols, commas, units text, or accounting negatives. */
export function parseNumber(input: string | number | null | undefined): ParseResult<number> {
  if (input === null || input === undefined) return { value: null };
  if (typeof input === 'number') return Number.isFinite(input) ? { value: input } : { value: null, issue: 'Not a finite number' };
  const text = String(input).trim();
  if (isEmptyToken(text)) return { value: null };
  let negative = false;
  let s = text;
  if (/^\(.*\)$/.test(s)) {
    negative = true;
    s = s.slice(1, -1);
  }
  s = s.replace(/[$,\s]/g, '').replace(/^\+/, '');
  if (s.startsWith('-')) {
    negative = !negative;
    s = s.slice(1);
  }
  // Strip trailing unit words like "sqft", "sf", "ac", "acres", "mi"
  const m = s.match(/^(\d*\.?\d+)([a-zA-Z/%]*)$/);
  if (!m) {
    // Try pulling a leading number out of mixed text like "2,100 (approx)"
    const m2 = s.match(/^(\d*\.?\d+)/);
    if (!m2) return { value: null, issue: `Could not parse "${text}" as a number` };
    const n = Number(m2[1]);
    return { value: negative ? -n : n, issue: `Parsed number from "${text}"; please verify` };
  }
  const n = Number(m[1]);
  if (!Number.isFinite(n)) return { value: null, issue: `Could not parse "${text}" as a number` };
  return { value: negative ? -n : n };
}

export function parseCurrency(input: string | number | null | undefined): ParseResult<number> {
  return parseNumber(input);
}

const MONTHS: Record<string, number> = {
  jan: 1, january: 1, feb: 2, february: 2, mar: 3, march: 3, apr: 4, april: 4, may: 5, jun: 6, june: 6,
  jul: 7, july: 7, aug: 8, august: 8, sep: 9, sept: 9, september: 9, oct: 10, october: 10, nov: 11, november: 11, dec: 12, december: 12,
};

function validYmd(y: number, m: number, d: number): boolean {
  if (!Number.isInteger(y) || !Number.isInteger(m) || !Number.isInteger(d)) return false;
  if (m < 1 || m > 12 || d < 1 || d > 31) return false;
  const dt = new Date(Date.UTC(y, m - 1, d));
  return dt.getUTCFullYear() === y && dt.getUTCMonth() === m - 1 && dt.getUTCDate() === d;
}

function iso(y: number, m: number, d: number): string {
  return `${String(y).padStart(4, '0')}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
}

function expandYear(y: number): number {
  if (y >= 100) return y;
  return y >= 70 ? 1900 + y : 2000 + y;
}

export type DateOrder = 'MDY' | 'DMY' | 'YMD' | 'auto';

/**
 * Parse common date formats into YYYY-MM-DD.
 * Supports ISO (2024-03-05, with optional time), US (03/05/2024, 3/5/24), dashed/dotted numerics,
 * and month names (Mar 5, 2024 / 5 Mar 2024). When both day and month are <= 12 and the order
 * is 'auto', the result is flagged ambiguous and MDY is assumed.
 */
export function parseDate(input: string | null | undefined, order: DateOrder = 'auto'): ParseResult<string> {
  if (input === null || input === undefined) return { value: null };
  const text = String(input).trim();
  if (isEmptyToken(text)) return { value: null };

  // ISO or YYYY/MM/DD with optional time
  let m = text.match(/^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[T\s].*)?$/);
  if (m) {
    const y = Number(m[1]), mo = Number(m[2]), d = Number(m[3]);
    return validYmd(y, mo, d) ? { value: iso(y, mo, d) } : { value: null, issue: `Invalid date "${text}"` };
  }

  // Numeric with separators: a/b/c
  m = text.match(/^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})(?:[T\s].*)?$/);
  if (m) {
    const a = Number(m[1]), b = Number(m[2]), y = expandYear(Number(m[3]));
    let mo: number, d: number;
    let ambiguous = false;
    if (order === 'DMY') {
      d = a; mo = b;
    } else if (order === 'MDY' || order === 'YMD') {
      mo = a; d = b;
    } else {
      // auto: decide from magnitude when possible
      if (a > 12 && b <= 12) { d = a; mo = b; }
      else if (b > 12 && a <= 12) { mo = a; d = b; }
      else { mo = a; d = b; ambiguous = a !== b; }
    }
    if (!validYmd(y, mo, d)) return { value: null, issue: `Invalid date "${text}"` };
    return { value: iso(y, mo, d), ambiguous };
  }

  // Month name forms
  m = text.match(/^([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{2,4})$/);
  if (m && MONTHS[m[1].toLowerCase()]) {
    const mo = MONTHS[m[1].toLowerCase()], d = Number(m[2]), y = expandYear(Number(m[3]));
    return validYmd(y, mo, d) ? { value: iso(y, mo, d) } : { value: null, issue: `Invalid date "${text}"` };
  }
  m = text.match(/^(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{2,4})$/);
  if (m && MONTHS[m[2].toLowerCase()]) {
    const d = Number(m[1]), mo = MONTHS[m[2].toLowerCase()], y = expandYear(Number(m[3]));
    return validYmd(y, mo, d) ? { value: iso(y, mo, d) } : { value: null, issue: `Invalid date "${text}"` };
  }
  // Compact YYYYMMDD
  m = text.match(/^(\d{4})(\d{2})(\d{2})$/);
  if (m) {
    const y = Number(m[1]), mo = Number(m[2]), d = Number(m[3]);
    return validYmd(y, mo, d) ? { value: iso(y, mo, d) } : { value: null, issue: `Invalid date "${text}"` };
  }
  return { value: null, issue: `Unrecognised date format "${text}"` };
}

const YES = new Set(['y', 'yes', 'true', '1', 't', 'x', 'included', 'present']);
const NO = new Set(['n', 'no', 'false', '0', 'f', 'none', 'no pool', 'not applicable']);

export function parseBoolean(input: string | boolean | null | undefined): ParseResult<boolean> {
  if (input === null || input === undefined) return { value: null };
  if (typeof input === 'boolean') return { value: input };
  const text = String(input).trim().toLowerCase();
  if (text === '' || text === '-' || text === 'n/a' || text === 'unknown') return { value: null };
  if (YES.has(text)) return { value: true };
  if (NO.has(text)) return { value: false };
  if (/^(no|none|without)\b/.test(text)) return { value: false, issue: `Interpreted "${input}" as No` };
  // Descriptive text (e.g. "In Ground, Gunite") is treated as presence of the feature.
  return { value: true, issue: `Interpreted "${input}" as Yes` };
}

export function parseStatus(input: string | null | undefined): { category: StatusCategory | null; raw: string | null } {
  const raw = sanitizeText(input);
  if (!raw) return { category: null, raw: null };
  const s = raw.toLowerCase();
  if (/\b(closed|sold|cs|sld|settled|close)\b/.test(s) || s === 's' || s === 'c') return { category: 'closed', raw };
  if (/\b(pending|under contract|contingent|pnd|uc|act under contract|active under contract|p|ac|auc|backup)\b/.test(s)) return { category: 'pending', raw };
  if (/\b(active|act|for sale|available|new|a|coming soon|price change|back on market)\b/.test(s)) return { category: 'active', raw };
  if (/\b(expired|withdrawn|cancelled|canceled|exp|wdn|cnl|temp off market|hold|leased|rented)\b/.test(s)) return { category: 'other', raw };
  return { category: 'other', raw };
}

export const SQFT_PER_ACRE = 43560;

export interface LotParse {
  raw: number | null;
  units: LotUnits | null;
  sqft: number | null;
  issue?: string;
  needsUnits?: boolean;
}

export function parseLotUnits(input: string | null | undefined): LotUnits | null {
  if (!input) return null;
  const s = String(input).trim().toLowerCase();
  if (/^(ac|acre|acres|acreage)$/.test(s) || s.includes('acre')) return 'acres';
  if (/^(sf|sqft|sq\.? ?ft|square feet|sq feet|ft2|ft²)$/.test(s) || s.includes('sq') || s.includes('feet')) return 'sqft';
  return null;
}

/**
 * Parse a lot size. Units are taken, in order, from text inside the value ("0.25 ac"),
 * an explicit units column, or the default chosen by the user during mapping.
 * 'auto' treats values under 100 as acres and larger values as square feet.
 */
export function parseLot(input: string | number | null | undefined, unitsHint: LotUnits | 'auto' | null): LotParse {
  if (input === null || input === undefined) return { raw: null, units: null, sqft: null };
  const text = String(input).trim();
  if (isEmptyToken(text)) return { raw: null, units: null, sqft: null };
  const num = parseNumber(text);
  if (num.value === null) return { raw: null, units: null, sqft: null, issue: num.issue ?? `Could not parse lot size "${text}"` };
  let units: LotUnits | null = null;
  const lower = text.toLowerCase();
  if (/\b(ac|acre|acres)\b/.test(lower)) units = 'acres';
  else if (/\b(sf|sqft|sq\.?\s?ft|square feet)\b/.test(lower)) units = 'sqft';
  else if (unitsHint === 'auto') units = num.value < 100 ? 'acres' : 'sqft';
  else units = unitsHint;
  if (!units) return { raw: num.value, units: null, sqft: null, needsUnits: true, issue: `Lot size "${text}" has no units; choose SF or acres` };
  const sqft = units === 'acres' ? num.value * SQFT_PER_ACRE : num.value;
  return { raw: num.value, units, sqft };
}

export function acresToSqft(acres: number): number {
  return acres * SQFT_PER_ACRE;
}

export function sqftToAcres(sqft: number): number {
  return sqft / SQFT_PER_ACRE;
}

/** Validate a photo URL: only http(s) URLs that parse cleanly are accepted. */
export function validatePhotoUrl(input: string | null | undefined): ParseResult<string> {
  const text = sanitizeText(input);
  if (!text) return { value: null };
  // Some exports pack several URLs separated by commas or pipes; keep the first.
  const first = text.split(/[|,\s]+/).find((p) => p.length > 0) ?? text;
  try {
    const u = new URL(first);
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return { value: null, issue: `Photo URL rejected (not http/https): "${first}"` };
    return { value: u.toString() };
  } catch {
    return { value: null, issue: `Invalid photo URL "${first}"` };
  }
}

export type BathsConvention = 'fullPointHalf' | 'decimal' | 'none';

/**
 * Split a combined bath figure using an explicitly confirmed convention.
 *  - fullPointHalf: "3.1" => 3 full, 1 half (the digit after the point is a count of half baths)
 *  - decimal: "3.5" => 3 full, 1 half (0.5 = one half bath); other fractions are rejected
 *  - none: do not split; value is kept only as an informational note
 */
export function splitCombinedBaths(input: string | number | null | undefined, convention: BathsConvention): { full: number | null; half: number | null; issue?: string } {
  if (input === null || input === undefined) return { full: null, half: null };
  const text = String(input).trim();
  if (isEmptyToken(text)) return { full: null, half: null };
  if (convention === 'none') return { full: null, half: null, issue: `Combined baths "${text}" not split (no convention confirmed)` };
  const m = text.replace(/[^\d.]/g, '').match(/^(\d+)(?:\.(\d+))?$/);
  if (!m) return { full: null, half: null, issue: `Could not parse baths "${text}"` };
  const whole = Number(m[1]);
  const frac = m[2] ?? '';
  if (convention === 'fullPointHalf') {
    const half = frac === '' ? 0 : Number(frac);
    return { full: whole, half };
  }
  // decimal
  if (frac === '' || frac === '0') return { full: whole, half: 0 };
  if (frac === '5') return { full: whole, half: 1 };
  return { full: null, half: null, issue: `Baths "${text}" is not a .5 decimal; cannot split under the decimal convention` };
}
