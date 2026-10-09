import type { Reconciliation } from '../types';

export interface Contributor {
  compId: string;
  salePrice: number;
  adjustedPrice: number;
  weight: number;
}

export function median(values: number[]): number | null {
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

export function mean(values: number[]): number | null {
  if (!values.length) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function range(values: number[]): [number, number] | null {
  if (!values.length) return null;
  return [Math.min(...values), Math.max(...values)];
}

/**
 * Weighted indication = Σ(adjusted price × weight) ÷ Σ(weights).
 * Weights are non-negative; a zero total yields no indication rather than a misleading number.
 */
export function weightedIndication(contributors: Contributor[]): { value: number | null; totalWeight: number; normalized: Record<string, number | null> } {
  const totalWeight = contributors.reduce((a, c) => a + Math.max(0, c.weight), 0);
  const normalized: Record<string, number | null> = {};
  if (totalWeight <= 0) {
    for (const c of contributors) normalized[c.compId] = null;
    return { value: null, totalWeight, normalized };
  }
  let sum = 0;
  for (const c of contributors) {
    const w = Math.max(0, c.weight) / totalWeight;
    normalized[c.compId] = w;
    sum += c.adjustedPrice * w;
  }
  return { value: sum, totalWeight, normalized };
}

export function reconcile(contributors: Contributor[], includedCount: number, excludedForUnresolved: number): { reconciliation: Reconciliation; normalized: Record<string, number | null> } {
  const adjusted = contributors.map((c) => c.adjustedPrice);
  const unadjusted = contributors.map((c) => c.salePrice);
  const wi = weightedIndication(contributors);
  let message: string | null = null;
  if (includedCount === 0) message = 'No comparables are included. Include closed sales to produce a value indication.';
  else if (contributors.length === 0) message = 'No included closed sale is fully calculable. Resolve adjustments or missing data to produce an indication.';
  else if (wi.value === null) message = 'Total weight is zero. Assign a positive weight to at least one included sale.';
  return {
    reconciliation: {
      includedCount,
      contributingCount: contributors.length,
      excludedForUnresolved,
      unadjustedRange: range(unadjusted),
      adjustedRange: range(adjusted),
      adjustedMean: mean(adjusted),
      adjustedMedian: median(adjusted),
      weightedIndication: wi.value,
      totalWeight: wi.totalWeight,
      message,
    },
    normalized: wi.normalized,
  };
}
