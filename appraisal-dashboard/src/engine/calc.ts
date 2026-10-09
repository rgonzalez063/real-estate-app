import type { AdjustmentRule, Comparable, CompCalculation, Scenario, ScenarioCalculation, Warning } from '../types';
import { computeLine, reviewFlags, totalLines } from './adjustments';
import { ATTR_LABELS } from './fields';
import { median } from './reconciliation';
import { reconcile, type Contributor } from './reconciliation';

export const CALCULATION_ORDER = [
  '1. Start from the comparable original sale price (source data).',
  '2. Concessions adjustment (appraiser factor × reported concession, or manual override). Applied once.',
  '3. Market conditions / time adjustment on the original sale price basis. Applied once.',
  '4. Physical, site, location and custom adjustments: (Subject − Comparable) × Rate, flat, percentage or categorical.',
  '5. Adjusted sale price = Original sale price + Σ signed adjustments. Net = Σ signed; Gross = Σ |each|; percentages relative to original sale price.',
  '6. Weighted indication = Σ(Adjusted price × Weight) ÷ Σ Weights over included, calculable closed sales.',
];

/** Rules in display order with market and concessions first, matching the calculation order. */
export function orderedRules(rules: AdjustmentRule[]): AdjustmentRule[] {
  const prio = (r: AdjustmentRule) => (r.kind === 'concessions' ? 0 : r.kind === 'market' ? 1 : 2);
  return [...rules].sort((a, b) => prio(a) - prio(b) || a.order - b.order);
}

export function overlapWarnings(rules: AdjustmentRule[]): Warning[] {
  const active = rules.filter((r) => r.active);
  const warnings: Warning[] = [];
  const byField = new Map<string, AdjustmentRule[]>();
  for (const r of active) {
    if (r.field) byField.set(r.field, [...(byField.get(r.field) ?? []), r]);
  }
  for (const [field, rs] of byField) {
    if (rs.length > 1) warnings.push({ code: 'duplicateField', message: `${rs.length} active rules use ${ATTR_LABELS[field as keyof typeof ATTR_LABELS] ?? field}: ${rs.map((r) => r.label).join(', ')}. Possible double counting.` });
  }
  const has = (cat: string) => active.some((r) => r.category === cat);
  if (has('age') && has('condition')) warnings.push({ code: 'ageCondition', message: 'Age and Condition are both active. Confirm the two adjustments do not overlap (condition often already reflects effective age).' });
  const areaFields = active.filter((r) => r.field && String(r.field).startsWith('area'));
  if (areaFields.length > 1) warnings.push({ code: 'multiArea', message: `Multiple area adjustments are active (${areaFields.map((r) => r.label).join(', ')}). Confirm each uses a distinct area category.` });
  if (has('location') && has('view')) warnings.push({ code: 'locationView', message: 'Location and View are both active. Confirm they measure different attributes.' });
  if (has('view') && has('waterfront')) warnings.push({ code: 'viewWaterfront', message: 'View and Waterfront are both active. Confirm water view is not counted twice.' });
  return warnings;
}

function missingKeyFields(c: Comparable): string[] {
  const missing: string[] = [];
  if (c.attrs.salePrice === null) missing.push('sale price');
  if (c.attrs.saleDate === null) missing.push('sale date');
  if (c.attrs.areaAboveGrade === null) missing.push('above-grade area');
  return missing;
}

/**
 * Single shared calculation used by the grid, cards, charts, summary and exports.
 */
export function computeScenario(scenario: Scenario): ScenarioCalculation {
  const rules = orderedRules(scenario.rules);
  const ctx = { effectiveDate: scenario.effectiveDate };
  const comps: Record<string, CompCalculation> = {};

  // First pass: lines and totals
  for (const comp of scenario.comparables) {
    const lines = rules.map((r) => computeLine(r, scenario.subject, comp, ctx));
    const totals = totalLines(comp.attrs.salePrice, lines);
    const blockers: string[] = [];
    if (comp.attrs.salePrice === null) blockers.push('Sale price unknown');
    if (totals.unresolved.length) blockers.push(`${totals.unresolved.length} unresolved adjustment${totals.unresolved.length > 1 ? 's' : ''}`);
    const area = comp.attrs.areaAboveGrade;
    comps[comp.id] = {
      compId: comp.id,
      salePrice: comp.attrs.salePrice,
      lines,
      netAdjustment: totals.net,
      grossAdjustment: totals.gross,
      netPct: totals.netPct,
      grossPct: totals.grossPct,
      adjustedPrice: totals.adjustedPrice,
      calculable: totals.adjustedPrice !== null,
      blockers,
      unresolved: totals.unresolved,
      flags: [],
      normalizedWeight: null,
      pricePerSqft: comp.attrs.salePrice !== null && area ? comp.attrs.salePrice / area : null,
      pricePerSqftBasis: 'above-grade finished area',
      statusCategory: comp.attrs.status,
    };
  }

  // Contributors: included, closed, calculable
  const included = scenario.comparables.filter((c) => c.included);
  const contributors: Contributor[] = [];
  let excludedForUnresolved = 0;
  for (const c of included) {
    const calc = comps[c.id];
    if (c.attrs.status !== 'closed') continue;
    if (!calc.calculable) {
      excludedForUnresolved += 1;
      continue;
    }
    contributors.push({ compId: c.id, salePrice: c.attrs.salePrice as number, adjustedPrice: calc.adjustedPrice as number, weight: Math.max(0, c.weight) });
  }
  const closedIncluded = included.filter((c) => c.attrs.status === 'closed').length;
  const { reconciliation, normalized } = reconcile(contributors, closedIncluded, excludedForUnresolved);
  for (const [id, w] of Object.entries(normalized)) comps[id].normalizedWeight = w;

  // Flags (after median is known)
  const med = median(contributors.map((c) => c.adjustedPrice));
  for (const comp of scenario.comparables) {
    const calc = comps[comp.id];
    calc.flags = reviewFlags(
      scenario.settings.flags,
      comp.attrs.salePrice,
      { net: calc.netAdjustment, gross: calc.grossAdjustment, netPct: calc.netPct, grossPct: calc.grossPct, adjustedPrice: calc.adjustedPrice, unresolved: calc.unresolved },
      calc.lines,
      missingKeyFields(comp),
      comp.included && comp.attrs.status === 'closed' ? med : null,
    );
  }

  const warnings = overlapWarnings(scenario.rules);
  const nonClosedIncluded = included.length - closedIncluded;
  if (nonClosedIncluded > 0) warnings.push({ code: 'nonClosed', message: `${nonClosedIncluded} included listing/pending record${nonClosedIncluded > 1 ? 's are' : ' is'} shown as market context only and excluded from the closed-sale indication.` });
  if (!scenario.effectiveDate) warnings.push({ code: 'noEffectiveDate', message: 'Effective date is not set; market conditions adjustments cannot be computed.' });

  return { comps, reconciliation, warnings, calculationOrder: CALCULATION_ORDER };
}
