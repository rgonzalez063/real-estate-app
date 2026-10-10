import type { Comparable, FieldKey, MappingOptions, ParseIssue, PropertyAttributes } from '../types';
import { FIELD_DEFS, emptyAttributes } from './fields';
import { makeId } from './ids';
import {
  parseBoolean,
  parseDate,
  parseLot,
  parseLotUnits,
  parseNumber,
  parseStatus,
  sanitizeText,
  splitCombinedBaths,
  validatePhotoUrl,
} from './parse';

/** Normalise a header for alias matching: lower-case, strip punctuation, collapse spaces. */
export function normalizeHeader(h: string): string {
  return h
    .toLowerCase()
    .replace(/[_\-./#()]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

const ALIAS_INDEX: Map<string, FieldKey> = new Map();
for (const def of FIELD_DEFS) {
  for (const a of def.aliases) {
    const n = normalizeHeader(a);
    if (!ALIAS_INDEX.has(n)) ALIAS_INDEX.set(n, def.key);
  }
  ALIAS_INDEX.set(normalizeHeader(def.label), def.key);
  ALIAS_INDEX.set(normalizeHeader(def.key), def.key);
}

/**
 * Suggest a field for each header. Exact alias matches win; otherwise a contained-alias match
 * is used when it is unambiguous. Each target field is suggested at most once.
 */
export function suggestMapping(headers: string[]): Record<string, FieldKey> {
  const result: Record<string, FieldKey> = {};
  const used = new Set<FieldKey>();
  const normalized = headers.map((h) => ({ h, n: normalizeHeader(h) }));

  // Pass 1: exact alias match
  for (const { h, n } of normalized) {
    const key = ALIAS_INDEX.get(n);
    if (key && !used.has(key)) {
      result[h] = key;
      used.add(key);
    }
  }
  // Pass 2: contained alias (e.g. "Close Price (USD)")
  for (const { h, n } of normalized) {
    if (result[h]) continue;
    let best: { key: FieldKey; len: number } | null = null;
    for (const [alias, key] of ALIAS_INDEX) {
      if (alias.length < 3 || used.has(key)) continue;
      const re = new RegExp(`(^|\\s)${alias.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(\\s|$)`);
      if (re.test(n) && (!best || alias.length > best.len)) best = { key, len: alias.length };
    }
    if (best) {
      result[h] = best.key;
      used.add(best.key);
    }
  }
  return result;
}

export interface NormalizeResult {
  comps: Comparable[];
  /** Any of the parsed dates were ambiguous and the user has not chosen an order. */
  datesAmbiguous: boolean;
  /** Lot sizes are present but no units could be determined. */
  lotNeedsUnits: boolean;
  /** A combined baths column is mapped but no convention has been confirmed. */
  bathsNeedsConvention: boolean;
  totalIssues: number;
  missingByField: Partial<Record<FieldKey, number>>;
}

export function defaultMappingOptions(): MappingOptions {
  return { dateOrder: 'auto', lotUnitsDefault: null, bathsConvention: null };
}

/**
 * Convert raw CSV rows into Comparable records using the chosen mapping.
 * Every original row is preserved verbatim on the record.
 */
export function normalizeRows(
  rows: Record<string, string>[],
  mapping: Record<string, FieldKey>,
  options: MappingOptions,
  sourceFile: string,
): NormalizeResult {
  const comps: Comparable[] = [];
  let datesAmbiguous = false;
  let lotNeedsUnits = false;
  let bathsNeedsConvention = false;
  let totalIssues = 0;
  const missingByField: Partial<Record<FieldKey, number>> = {};

  const byField: Partial<Record<FieldKey, string>> = {};
  for (const [header, key] of Object.entries(mapping)) if (key) byField[key] = header;

  rows.forEach((row, index) => {
    const attrs = emptyAttributes();
    const issues: ParseIssue[] = [];
    const get = (key: FieldKey): string | null => {
      const header = byField[key];
      if (header === undefined) return null;
      const v = row[header];
      return v === undefined || v === null ? null : String(v);
    };
    const note = (field: string, raw: string | null, message?: string) => {
      if (message) issues.push({ field, raw: raw ?? '', message });
    };
    const markMissing = (key: FieldKey) => {
      if (byField[key] !== undefined) missingByField[key] = (missingByField[key] ?? 0) + 1;
    };

    // Text fields
    const textKeys: (FieldKey & keyof PropertyAttributes)[] = ['mlsNumber', 'address', 'city', 'zip', 'neighborhood', 'quality', 'condition', 'renovation', 'remarks', 'location', 'view', 'design', 'tenure', 'functionalUtility', 'heatingCooling', 'energyEfficient', 'porchPatioDeck', 'amenities'];
    for (const k of textKeys) {
      const v = sanitizeText(get(k));
      (attrs as unknown as Record<string, unknown>)[k] = v;
      if (v === null) markMissing(k);
    }

    // Status
    const st = parseStatus(get('status'));
    attrs.status = st.category;
    attrs.statusRaw = st.raw;
    if (!st.raw) markMissing('status');

    // Currency / numbers
    const numKeys: (FieldKey & keyof PropertyAttributes)[] = [
      'salePrice', 'listPrice', 'daysOnMarket', 'areaAboveGrade', 'areaBelowGradeFinished', 'areaBelowGradeUnfinished', 'areaOther',
      'bedrooms', 'fullBaths', 'halfBaths', 'yearBuilt', 'age', 'garageSpaces', 'carportSpaces', 'concessions', 'latitude', 'longitude', 'distance',
    ];
    for (const k of numKeys) {
      const raw = get(k);
      const r = parseNumber(raw);
      (attrs as unknown as Record<string, unknown>)[k] = r.value;
      note(k, raw, r.issue);
      if (r.value === null) markMissing(k);
    }

    // Combined baths (only split under an explicit convention)
    const combinedRaw = get('bathsCombined');
    if (byField.bathsCombined !== undefined && combinedRaw !== null && combinedRaw.trim() !== '') {
      if (!options.bathsConvention) {
        bathsNeedsConvention = true;
        note('bathsCombined', combinedRaw, 'Combined baths need a confirmed convention before splitting');
      } else {
        const split = splitCombinedBaths(combinedRaw, options.bathsConvention);
        note('bathsCombined', combinedRaw, split.issue);
        if (attrs.fullBaths === null && split.full !== null) attrs.fullBaths = split.full;
        if (attrs.halfBaths === null && split.half !== null) attrs.halfBaths = split.half;
      }
    }

    // Dates
    for (const k of ['saleDate', 'contractDate'] as const) {
      const raw = get(k);
      const r = parseDate(raw, options.dateOrder);
      attrs[k] = r.value;
      note(k, raw, r.issue);
      if (r.ambiguous && options.dateOrder === 'auto') datesAmbiguous = true;
      if (r.value === null) markMissing(k);
    }

    // Lot size
    const lotRaw = get('lotSize');
    const unitsCol = parseLotUnits(get('lotSizeUnits'));
    const lot = parseLot(lotRaw, unitsCol ?? options.lotUnitsDefault);
    attrs.lotSizeRaw = lot.raw;
    attrs.lotSizeUnits = lot.units;
    attrs.lotSizeSqft = lot.sqft;
    note('lotSize', lotRaw, lot.issue);
    if (lot.needsUnits) lotNeedsUnits = true;
    if (lot.sqft === null) markMissing('lotSize');

    // Booleans
    for (const k of ['pool', 'waterfront', 'waterView'] as const) {
      const raw = get(k);
      const r = parseBoolean(raw);
      attrs[k] = r.value;
      note(k, raw, r.issue);
      if (r.value === null) markMissing(k);
    }

    // Photo URL
    const photoRaw = get('photoUrl');
    const photo = validatePhotoUrl(photoRaw);
    attrs.photoUrl = photo.value;
    note('photoUrl', photoRaw, photo.issue);

    // Age derived from year built is NOT inferred here; the engine computes age differences from yearBuilt when asked.

    totalIssues += issues.length;
    comps.push({
      id: makeId('comp'),
      attrs,
      raw: { ...row },
      sourceFile,
      sourceRowIndex: index,
      parseIssues: issues,
      included: false,
      selectionNote: '',
      weight: 1,
      overrides: {},
      resolutions: {},
      gridNote: '',
    });
  });

  return { comps, datesAmbiguous, lotNeedsUnits, bathsNeedsConvention, totalIssues, missingByField };
}

export interface DuplicateGroup {
  key: string;
  reason: 'mls' | 'addressDate';
  compIds: string[];
}

function addrKey(a: string | null): string | null {
  if (!a) return null;
  return a.toLowerCase().replace(/[^a-z0-9]/g, '');
}

/** Find potential duplicates by MLS number, or by address + transaction date. */
export function findDuplicates(comps: Comparable[]): DuplicateGroup[] {
  const byMls = new Map<string, string[]>();
  const byAddr = new Map<string, string[]>();
  for (const c of comps) {
    const mls = c.attrs.mlsNumber?.trim().toLowerCase();
    if (mls) byMls.set(mls, [...(byMls.get(mls) ?? []), c.id]);
    const ak = addrKey(c.attrs.address);
    const date = c.attrs.saleDate ?? c.attrs.contractDate;
    if (ak && date) {
      const k = `${ak}|${date}`;
      byAddr.set(k, [...(byAddr.get(k) ?? []), c.id]);
    }
  }
  const groups: DuplicateGroup[] = [];
  const seen = new Set<string>();
  for (const [k, ids] of byMls) {
    if (ids.length > 1) {
      groups.push({ key: `mls:${k}`, reason: 'mls', compIds: ids });
      ids.forEach((id) => seen.add(id));
    }
  }
  for (const [k, ids] of byAddr) {
    if (ids.length > 1 && !ids.every((id) => seen.has(id))) {
      groups.push({ key: `addr:${k}`, reason: 'addressDate', compIds: ids });
    }
  }
  return groups;
}

/** Headers mapped to each field, for showing required-field coverage. */
export function coverage(mapping: Record<string, FieldKey>): Set<FieldKey> {
  return new Set(Object.values(mapping).filter(Boolean));
}

export const RECOMMENDED_FIELDS: FieldKey[] = ['address', 'status', 'salePrice', 'saleDate', 'areaAboveGrade'];
