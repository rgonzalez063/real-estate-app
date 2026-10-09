import { describe, expect, it } from 'vitest';
import { computeScenario } from '../engine/calc';
import { comp, rule, scenarioWith } from './helpers';

describe('adjustment signs and overrides', () => {
  it('matches the specified calculation check (+$10,000 area, $310,000 adjusted)', () => {
    const a = comp({ areaAboveGrade: 1800, salePrice: 300000 }, { id: 'A' });
    const s = scenarioWith({ areaAboveGrade: 2000 }, [a], (rules) => {
      const r = rules.find((x) => x.category === 'areaAboveGrade')!;
      r.active = true;
      r.rate = 50;
    });
    const calc = computeScenario(s);
    const line = calc.comps.A.lines.find((l) => l.category === 'areaAboveGrade')!;
    expect(line.amount).toBe(10000);
    expect(line.status).toBe('computed');
    expect(line.source).toBe('global');
    expect(line.explanation).toMatch(/inferior/);
    expect(calc.comps.A.adjustedPrice).toBe(310000);
  });

  it('subtracts when the comparable is superior', () => {
    const a = comp({ areaAboveGrade: 2200, salePrice: 300000 }, { id: 'A' });
    const s = scenarioWith({ areaAboveGrade: 2000 }, [a], (rules) => {
      const r = rules.find((x) => x.category === 'areaAboveGrade')!;
      r.active = true;
      r.rate = 50;
    });
    const line = computeScenario(s).comps.A.lines.find((l) => l.category === 'areaAboveGrade')!;
    expect(line.amount).toBe(-10000);
    expect(line.explanation).toMatch(/superior/);
  });

  it('inactive rules with blank rates contribute nothing and do not block', () => {
    const a = comp({ areaAboveGrade: 1800, salePrice: 300000 }, { id: 'A' });
    const s = scenarioWith({ areaAboveGrade: 2000 }, [a]);
    const calc = computeScenario(s);
    expect(calc.comps.A.adjustedPrice).toBe(300000);
    expect(calc.comps.A.lines.every((l) => l.status === 'inactive')).toBe(true);
  });

  it('an override replaces the computed adjustment instead of adding to it', () => {
    const a = comp({ areaAboveGrade: 1800, salePrice: 300000 }, { id: 'A' });
    const s = scenarioWith({ areaAboveGrade: 2000 }, [a], (rules) => {
      const r = rules.find((x) => x.category === 'areaAboveGrade')!;
      r.active = true;
      r.rate = 50;
    });
    const r = rule(s, 'areaAboveGrade');
    a.overrides[r.id] = { amount: 7500, reason: 'paired sales', source: 'Matched pair 1', attachmentRef: '' };
    const calc = computeScenario(s);
    const line = calc.comps.A.lines.find((l) => l.ruleId === r.id)!;
    expect(line.status).toBe('override');
    expect(line.source).toBe('manual');
    expect(line.amount).toBe(7500);
    expect(calc.comps.A.adjustedPrice).toBe(307500);
    expect(calc.comps.A.lines.filter((l) => l.ruleId === r.id)).toHaveLength(1);
  });

  it('flat adjustments follow feature direction', () => {
    const withPool = comp({ pool: true, salePrice: 300000 }, { id: 'P' });
    const noPool = comp({ pool: false, salePrice: 300000 }, { id: 'N' });
    const s = scenarioWith({ pool: false }, [withPool, noPool], (rules) => {
      const r = rules.find((x) => x.category === 'pool')!;
      r.active = true;
      r.rate = 20000;
    });
    const calc = computeScenario(s);
    expect(calc.comps.P.lines.find((l) => l.category === 'pool')!.amount).toBe(-20000);
    expect(calc.comps.N.lines.find((l) => l.category === 'pool')!.amount).toBe(0);
  });

  it('categorical adjustments use appraiser-defined dollar relationships', () => {
    const c4 = comp({ condition: 'C4', salePrice: 300000 }, { id: 'C4' });
    const c2 = comp({ condition: 'C2', salePrice: 300000 }, { id: 'C2' });
    const unknown = comp({ condition: 'C5', salePrice: 300000 }, { id: 'C5' });
    const s = scenarioWith({ condition: 'C3' }, [c4, c2, unknown], (rules) => {
      const r = rules.find((x) => x.category === 'condition')!;
      r.active = true;
      r.categoricalValues = { C2: 25000, C3: 10000, C4: 0 }; // unequal intervals are allowed
    });
    const calc = computeScenario(s);
    expect(calc.comps.C4.lines.find((l) => l.category === 'condition')!.amount).toBe(10000);
    expect(calc.comps.C2.lines.find((l) => l.category === 'condition')!.amount).toBe(-15000);
    expect(calc.comps.C5.lines.find((l) => l.category === 'condition')!.status).toBe('unresolved');
    expect(calc.comps.C5.calculable).toBe(false);
  });

  it('concessions are not deducted automatically; a factor must be set', () => {
    const a = comp({ salePrice: 300000, concessions: 5000 }, { id: 'A' });
    const s = scenarioWith({}, [a]);
    expect(computeScenario(s).comps.A.adjustedPrice).toBe(300000);
    rule(s, 'concessions').active = true;
    rule(s, 'concessions').concessionFactor = 1;
    const calc = computeScenario(s);
    expect(calc.comps.A.lines.find((l) => l.category === 'concessions')!.amount).toBe(-5000);
    expect(calc.comps.A.adjustedPrice).toBe(295000);
  });
});

describe('net versus gross adjustments', () => {
  it('nets signed amounts and sums absolute values for gross', () => {
    const a = comp({ areaAboveGrade: 1800, pool: true, salePrice: 300000 }, { id: 'A' });
    const s = scenarioWith({ areaAboveGrade: 2000, pool: false }, [a], (rules) => {
      const area = rules.find((x) => x.category === 'areaAboveGrade')!;
      area.active = true;
      area.rate = 50; // +10,000
      const pool = rules.find((x) => x.category === 'pool')!;
      pool.active = true;
      pool.rate = 15000; // -15,000
    });
    const c = computeScenario(s).comps.A;
    expect(c.netAdjustment).toBe(-5000);
    expect(c.grossAdjustment).toBe(25000);
    expect(c.netPct).toBeCloseTo(-1.6667, 3);
    expect(c.grossPct).toBeCloseTo(8.3333, 3);
    expect(c.adjustedPrice).toBe(295000);
  });

  it('N/A lines are skipped and confirmed-zero lines count as zero', () => {
    const a = comp({ areaAboveGrade: 1800, salePrice: 300000 }, { id: 'A' });
    const s = scenarioWith({ areaAboveGrade: 2000 }, [a], (rules) => {
      const area = rules.find((x) => x.category === 'areaAboveGrade')!;
      area.active = true;
      area.rate = 50;
      const pool = rules.find((x) => x.category === 'pool')!;
      pool.active = true;
      pool.rate = 15000;
      const loc = rules.find((x) => x.category === 'location')!;
      loc.active = true;
    });
    a.resolutions[rule(s, 'pool').id] = 'notApplicable';
    a.resolutions[rule(s, 'location').id] = 'confirmedZero';
    const c = computeScenario(s).comps.A;
    expect(c.unresolved).toHaveLength(0);
    expect(c.netAdjustment).toBe(10000);
    expect(c.grossAdjustment).toBe(10000);
    expect(c.adjustedPrice).toBe(310000);
  });
});

describe('missing values', () => {
  it('unknown comparable attributes stay unresolved rather than becoming zero', () => {
    const a = comp({ areaAboveGrade: null, salePrice: 300000 }, { id: 'A' });
    const s = scenarioWith({ areaAboveGrade: 2000 }, [a], (rules) => {
      const r = rules.find((x) => x.category === 'areaAboveGrade')!;
      r.active = true;
      r.rate = 50;
    });
    const calc = computeScenario(s);
    const line = calc.comps.A.lines.find((l) => l.category === 'areaAboveGrade')!;
    expect(line.status).toBe('unresolved');
    expect(line.amount).toBeNull();
    expect(calc.comps.A.adjustedPrice).toBeNull();
    expect(calc.reconciliation.contributingCount).toBe(0);
    expect(calc.reconciliation.excludedForUnresolved).toBe(1);
    expect(calc.reconciliation.weightedIndication).toBeNull();
  });

  it('unknown subject attributes also block the line', () => {
    const a = comp({ areaAboveGrade: 1800, salePrice: 300000 }, { id: 'A' });
    const s = scenarioWith({ areaAboveGrade: null }, [a], (rules) => {
      const r = rules.find((x) => x.category === 'areaAboveGrade')!;
      r.active = true;
      r.rate = 50;
    });
    expect(computeScenario(s).comps.A.lines.find((l) => l.category === 'areaAboveGrade')!.explanation).toMatch(/Subject/);
  });

  it('an active rule with a blank rate is unresolved, not zero', () => {
    const a = comp({ areaAboveGrade: 1800, salePrice: 300000 }, { id: 'A' });
    const s = scenarioWith({ areaAboveGrade: 2000 }, [a], (rules) => {
      rules.find((x) => x.category === 'areaAboveGrade')!.active = true;
    });
    expect(computeScenario(s).comps.A.lines.find((l) => l.category === 'areaAboveGrade')!.status).toBe('unresolved');
  });

  it('price per square foot is null without an area', () => {
    const a = comp({ areaAboveGrade: null, salePrice: 300000 }, { id: 'A' });
    const calc = computeScenario(scenarioWith({}, [a]));
    expect(calc.comps.A.pricePerSqft).toBeNull();
    expect(calc.comps.A.flags.some((f) => f.code === 'missingData')).toBe(true);
  });
});

describe('overlap warnings', () => {
  it('warns when age and condition are both active', () => {
    const s = scenarioWith({}, [], (rules) => {
      rules.find((x) => x.category === 'age')!.active = true;
      rules.find((x) => x.category === 'condition')!.active = true;
    });
    expect(computeScenario(s).warnings.some((w) => w.code === 'ageCondition')).toBe(true);
  });
});
