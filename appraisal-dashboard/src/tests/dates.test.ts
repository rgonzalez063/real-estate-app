import { describe, expect, it } from 'vitest';
import { daysBetween, elapsedMonths, DAYS_PER_MONTH } from '../engine/dates';
import { computeScenario } from '../engine/calc';
import { comp, rule, scenarioWith } from './helpers';

describe('date calculations', () => {
  it('uses date-only day differences', () => {
    expect(daysBetween('2026-01-01', '2026-01-31')).toBe(30);
    expect(daysBetween('2024-02-28', '2024-03-01')).toBe(2); // leap year
    expect(daysBetween('2026-03-01', '2026-02-01')).toBe(-28);
  });
  it('defines elapsed months as days / 30.4375', () => {
    expect(DAYS_PER_MONTH).toBe(30.4375);
    expect(elapsedMonths('2026-01-01', '2026-07-01')).toBeCloseTo(181 / 30.4375, 10);
  });
  it('computes the market conditions adjustment from the chosen date basis', () => {
    const a = comp({ salePrice: 300000, saleDate: '2026-03-31', contractDate: '2026-02-28' }, { id: 'A' });
    const s = scenarioWith({}, [a], (rules) => {
      const m = rules.find((x) => x.category === 'marketConditions')!;
      m.active = true;
      m.market = { mode: 'monthlyRate', monthlyRatePct: 0.5, dateBasis: 'saleDate' };
    });
    s.effectiveDate = '2026-09-30';
    let calc = computeScenario(s);
    let line = calc.comps.A.lines.find((l) => l.category === 'marketConditions')!;
    const months = daysBetween('2026-03-31', '2026-09-30') / 30.4375;
    expect(line.amount).toBeCloseTo(300000 * (Math.pow(1.005, months) - 1), 6);
    expect(line.explanation).toMatch(/sale date/);
    expect(line.explanation).toMatch(/original sale price/i);

    rule(s, 'marketConditions').market!.dateBasis = 'contractDate';
    calc = computeScenario(s);
    line = calc.comps.A.lines.find((l) => l.category === 'marketConditions')!;
    const months2 = daysBetween('2026-02-28', '2026-09-30') / 30.4375;
    expect(line.amount).toBeCloseTo(300000 * (Math.pow(1.005, months2) - 1), 6);
    expect(line.explanation).toMatch(/contract date/);
  });
  it('is unresolved without an effective date or comp date', () => {
    const a = comp({ salePrice: 300000, saleDate: null }, { id: 'A' });
    const s = scenarioWith({}, [a], (rules) => {
      const m = rules.find((x) => x.category === 'marketConditions')!;
      m.active = true;
      m.market = { mode: 'monthlyRate', monthlyRatePct: 0.5, dateBasis: 'saleDate' };
    });
    expect(computeScenario(s).comps.A.lines.find((l) => l.category === 'marketConditions')!.status).toBe('unresolved');
    a.attrs.saleDate = '2026-01-01';
    s.effectiveDate = null;
    expect(computeScenario(s).comps.A.lines.find((l) => l.category === 'marketConditions')!.status).toBe('unresolved');
  });
  it('manual market mode requires an override and applies it once', () => {
    const a = comp({ salePrice: 300000, saleDate: '2026-01-01' }, { id: 'A' });
    const s = scenarioWith({}, [a], (rules) => {
      const m = rules.find((x) => x.category === 'marketConditions')!;
      m.active = true;
      m.market = { mode: 'manual', monthlyRatePct: null, dateBasis: 'saleDate' };
    });
    expect(computeScenario(s).comps.A.calculable).toBe(false);
    a.overrides[rule(s, 'marketConditions').id] = { amount: 6000, reason: 'paired sales', source: '', attachmentRef: '' };
    const calc = computeScenario(s);
    expect(calc.comps.A.adjustedPrice).toBe(306000);
    expect(calc.comps.A.lines.filter((l) => l.category === 'marketConditions')).toHaveLength(1);
  });
});
