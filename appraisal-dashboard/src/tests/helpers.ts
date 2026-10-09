import type { AdjustmentRule, Comparable, PropertyAttributes, Scenario } from '../types';
import { emptyAttributes } from '../engine/fields';
import { newScenario } from '../engine/defaults';

export function comp(partial: Partial<PropertyAttributes>, extra: Partial<Comparable> = {}): Comparable {
  return {
    id: extra.id ?? `c_${Math.random().toString(36).slice(2, 8)}`,
    attrs: { ...emptyAttributes(), status: 'closed', statusRaw: 'Closed', ...partial },
    raw: {},
    sourceFile: 'test.csv',
    sourceRowIndex: 0,
    parseIssues: [],
    included: true,
    selectionNote: '',
    weight: 1,
    overrides: {},
    resolutions: {},
    gridNote: '',
    ...extra,
  };
}

export function scenarioWith(subject: Partial<PropertyAttributes>, comps: Comparable[], ruleEdits: (rules: AdjustmentRule[]) => void = () => {}): Scenario {
  const s = newScenario('test');
  s.effectiveDate = '2026-09-30';
  s.subject.attrs = { ...emptyAttributes(), ...subject };
  s.comparables = comps;
  ruleEdits(s.rules);
  return s;
}

export function rule(s: Scenario, category: string): AdjustmentRule {
  const r = s.rules.find((x) => x.category === category);
  if (!r) throw new Error(`rule ${category} not found`);
  return r;
}
