/**
 * Date helpers. All dates are calendar dates (YYYY-MM-DD) with no time zone.
 */

const DAY_MS = 86_400_000;
export const DAYS_PER_MONTH = 30.4375;

export function isIsoDate(value: string | null | undefined): value is string {
  return !!value && /^\d{4}-\d{2}-\d{2}$/.test(value) && !Number.isNaN(Date.parse(value + 'T00:00:00Z'));
}

/** Day number of an ISO calendar date, computed in UTC to avoid DST drift. */
export function dayNumber(iso: string): number {
  const [y, m, d] = iso.split('-').map(Number);
  return Math.round(Date.UTC(y, m - 1, d) / DAY_MS);
}

/** Date-only difference in days (later - earlier). */
export function daysBetween(fromIso: string, toIso: string): number {
  return dayNumber(toIso) - dayNumber(fromIso);
}

/** Elapsed months defined as day difference / 30.4375 (full precision). */
export function elapsedMonths(fromIso: string, toIso: string): number {
  return daysBetween(fromIso, toIso) / DAYS_PER_MONTH;
}

export function todayIso(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

export function formatDate(iso: string | null): string {
  if (!iso) return '—';
  if (!isIsoDate(iso)) return iso;
  const [y, m, d] = iso.split('-');
  return `${m}/${d}/${y}`;
}

export function yearOf(iso: string | null): number | null {
  if (!isIsoDate(iso)) return null;
  return Number(iso.slice(0, 4));
}
