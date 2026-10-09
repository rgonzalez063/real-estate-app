import { describe, expect, it } from 'vitest';
import { computeScenario } from '../engine/calc';
import { weightedIndication } from '../engine/reconciliation';
import { comp, scenarioWith } from './helpers';

describe('weighted reconciliation', () => {
  it('produces $318,000 for $310,000 @ 60% and $330,000 @ 40%', () => {
    const a = comp({ areaAboveGrade: 1800, salePrice: 300000 }, { id: 'A', weight: 60 });
    const b = comp({ areaAboveGrade: 2000, salePrice: 330000 }, { id: 'B', weight: 40 });
    const s = scenarioWith({ areaAboveGrade: 2000 }, [a, b], (rules) => {
      const r = rules.find((x) => x.category === 'areaAboveGrade')!;
      r.active = true;
      r.rate = 50;
    });
    const calc = computeScenario(s);
    expect(calc.comps.A.adjustedPrice).toBe(310000);
    expect(calc.comps.B.adjustedPrice).toBe(330000);
    expect(calc.reconciliation.weightedIndication).toBe(318000);
    expect(calc.comps.A.normalizedWeight).toBeCloseTo(0.6);
    expect(calc.comps.B.normalizedWeight).toBeCloseTo(0.4);
    expect(calc.reconciliation.adjustedRange).toEqual([310000, 330000]);
    expect(calc.reconciliation.unadjustedRange).toEqual([300000, 330000]);
    expect(calc.reconciliation.adjustedMean).toBe(320000);
    expect(calc.reconciliation.adjustedMedian).toBe(320000);
    expect(calc.reconciliation.contributingCount).toBe(2);
  });

  it('defaults to equal weights', () => {
    const a = comp({ salePrice: 300000 }, { id: 'A' });
    const b = comp({ salePrice: 340000 }, { id: 'B' });
    const calc = computeScenario(scenarioWith({}, [a, b]));
    expect(calc.reconciliation.weightedIndication).toBe(320000);
  });

  it('returns no indication for zero total weight', () => {
    const r = weightedIndication([
      { compId: 'A', salePrice: 1, adjustedPrice: 310000, weight: 0 },
      { compId: 'B', salePrice: 1, adjustedPrice: 330000, weight: 0 },
    ]);
    expect(r.value).toBeNull();
    expect(r.normalized.A).toBeNull();
    const a = comp({ salePrice: 300000 }, { id: 'A', weight: 0 });
    const calc = computeScenario(scenarioWith({}, [a]));
    expect(calc.reconciliation.weightedIndication).toBeNull();
    expect(calc.reconciliation.message).toMatch(/zero/i);
  });

  it('handles no selections without a misleading value', () => {
    const a = comp({ salePrice: 300000 }, { id: 'A', included: false });
    const calc = computeScenario(scenarioWith({}, [a]));
    expect(calc.reconciliation.weightedIndication).toBeNull();
    expect(calc.reconciliation.includedCount).toBe(0);
    expect(calc.reconciliation.adjustedRange).toBeNull();
    expect(calc.reconciliation.message).toMatch(/No comparables/);
  });

  it('keeps listings and pending sales out of the closed-sale indication', () => {
    const closed = comp({ salePrice: 300000 }, { id: 'A' });
    const active = comp({ salePrice: null, listPrice: 400000, status: 'active', statusRaw: 'Active' }, { id: 'L' });
    const pending = comp({ salePrice: 390000, status: 'pending', statusRaw: 'Pending' }, { id: 'P' });
    const calc = computeScenario(scenarioWith({}, [closed, active, pending]));
    expect(calc.reconciliation.weightedIndication).toBe(300000);
    expect(calc.reconciliation.contributingCount).toBe(1);
    expect(calc.comps.P.normalizedWeight).toBeNull();
    expect(calc.warnings.some((w) => w.code === 'nonClosed')).toBe(true);
  });

  it('ignores negative weights by clamping to zero', () => {
    const a = comp({ salePrice: 300000 }, { id: 'A', weight: -5 });
    const b = comp({ salePrice: 320000 }, { id: 'B', weight: 1 });
    const calc = computeScenario(scenarioWith({}, [a, b]));
    expect(calc.reconciliation.weightedIndication).toBe(320000);
  });
});
