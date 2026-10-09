import type { Comparable, PropertyAttributes, StatusCategory } from '../../types';
import type { CompCalculation } from '../../types';
import { distanceToSubject } from '../../engine/geo';

export interface CompFilters {
  statuses: StatusCategory[];
  dateFrom: string | null;
  dateTo: string | null;
  priceMin: number | null;
  priceMax: number | null;
  areaMin: number | null;
  areaMax: number | null;
  lotMin: number | null;
  lotMax: number | null;
  neighborhood: string | null;
  quality: string | null;
  condition: string | null;
  pool: 'any' | 'yes' | 'no';
  garageMin: number | null;
  distanceMax: number | null;
  includedOnly: boolean;
  search: string;
}

export type SortKey = 'saleDate' | 'salePrice' | 'adjustedPrice' | 'areaAboveGrade' | 'pricePerSqft' | 'distance' | 'address';

export const defaultFilters = (): CompFilters => ({
  statuses: [],
  dateFrom: null,
  dateTo: null,
  priceMin: null,
  priceMax: null,
  areaMin: null,
  areaMax: null,
  lotMin: null,
  lotMax: null,
  neighborhood: null,
  quality: null,
  condition: null,
  pool: 'any',
  garageMin: null,
  distanceMax: null,
  includedOnly: false,
  search: '',
});

function inRange(v: number | null, min: number | null, max: number | null): boolean {
  if (min === null && max === null) return true;
  if (v === null) return false;
  if (min !== null && v < min) return false;
  if (max !== null && v > max) return false;
  return true;
}

export function applyFilters(comps: Comparable[], filters: CompFilters, subject: PropertyAttributes): Comparable[] {
  const q = filters.search.trim().toLowerCase();
  return comps.filter((c) => {
    const a = c.attrs;
    if (filters.includedOnly && !c.included) return false;
    if (filters.statuses.length && !filters.statuses.includes(a.status ?? 'other')) return false;
    if (filters.dateFrom || filters.dateTo) {
      const d = a.saleDate ?? a.contractDate;
      if (!d) return false;
      if (filters.dateFrom && d < filters.dateFrom) return false;
      if (filters.dateTo && d > filters.dateTo) return false;
    }
    if (!inRange(a.salePrice ?? a.listPrice, filters.priceMin, filters.priceMax)) return false;
    if (!inRange(a.areaAboveGrade, filters.areaMin, filters.areaMax)) return false;
    if (!inRange(a.lotSizeSqft, filters.lotMin, filters.lotMax)) return false;
    if (filters.neighborhood && a.neighborhood !== filters.neighborhood) return false;
    if (filters.quality && a.quality !== filters.quality) return false;
    if (filters.condition && a.condition !== filters.condition) return false;
    if (filters.pool === 'yes' && a.pool !== true) return false;
    if (filters.pool === 'no' && a.pool !== false) return false;
    if (filters.garageMin !== null && (a.garageSpaces === null || a.garageSpaces < filters.garageMin)) return false;
    if (filters.distanceMax !== null) {
      const d = distanceToSubject(subject, a).miles;
      if (d === null || d > filters.distanceMax) return false;
    }
    if (q) {
      const hay = [a.address, a.mlsNumber, a.city, a.neighborhood, a.remarks].filter(Boolean).join(' ').toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

export function sortComps(comps: Comparable[], key: SortKey, dir: 'asc' | 'desc', calc: Record<string, CompCalculation>, subject: PropertyAttributes): Comparable[] {
  const val = (c: Comparable): number | string | null => {
    switch (key) {
      case 'saleDate':
        return c.attrs.saleDate ?? c.attrs.contractDate;
      case 'salePrice':
        return c.attrs.salePrice ?? c.attrs.listPrice;
      case 'adjustedPrice':
        return calc[c.id]?.adjustedPrice ?? null;
      case 'areaAboveGrade':
        return c.attrs.areaAboveGrade;
      case 'pricePerSqft':
        return calc[c.id]?.pricePerSqft ?? null;
      case 'distance':
        return distanceToSubject(subject, c.attrs).miles;
      case 'address':
        return c.attrs.address;
    }
  };
  const m = dir === 'asc' ? 1 : -1;
  return [...comps].sort((a, b) => {
    const va = val(a), vb = val(b);
    if (va === null && vb === null) return 0;
    if (va === null) return 1; // unknown last
    if (vb === null) return -1;
    if (typeof va === 'string' || typeof vb === 'string') return String(va).localeCompare(String(vb)) * m;
    return (va - vb) * m;
  });
}

/** Distance filtering is only offered when reliable distance data or coordinates exist. */
export function distanceAvailable(comps: Comparable[], subject: PropertyAttributes): boolean {
  return comps.some((c) => distanceToSubject(subject, c.attrs).miles !== null);
}
