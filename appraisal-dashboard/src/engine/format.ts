/** Display formatting. Rounding happens here only, never in the engine. */

export function roundTo(value: number, increment: number): number {
  if (!increment || increment <= 0) return value;
  return Math.round(value / increment) * increment;
}

export function fmtMoney(value: number | null | undefined, opts: { rounding?: number; decimals?: number; signed?: boolean } = {}): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const rounded = opts.rounding ? roundTo(value, opts.rounding) : value;
  const decimals = opts.decimals ?? 0;
  const abs = Math.abs(rounded).toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  const sign = rounded < 0 ? '-' : opts.signed && rounded > 0 ? '+' : '';
  return `${sign}$${abs}`;
}

export function fmtNumber(value: number | null | undefined, decimals = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return value.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: decimals });
}

export function fmtPct(value: number | null | undefined, decimals = 1, signed = false): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const sign = signed && value > 0 ? '+' : '';
  return `${sign}${value.toFixed(decimals)}%`;
}

export function fmtBool(value: boolean | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return value ? 'Yes' : 'No';
}

export function fmtText(value: string | null | undefined): string {
  if (value === null || value === undefined || value === '') return '—';
  return value;
}

export function fmtLot(sqft: number | null, raw: number | null, units: string | null): string {
  if (sqft === null) return '—';
  if (units === 'acres' && raw !== null) return `${fmtNumber(raw, 3)} ac (${fmtNumber(sqft)} SF)`;
  if (sqft >= 43560) return `${fmtNumber(sqft)} SF (${fmtNumber(sqft / 43560, 2)} ac)`;
  return `${fmtNumber(sqft)} SF`;
}
