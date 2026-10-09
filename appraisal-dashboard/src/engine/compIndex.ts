import type { Scenario } from '../types';

/** Display numbers ("Comp 1", "Comp 2", …) for included comparables in scenario order. */
export function compIndexMap(scenario: Scenario): Record<string, number> {
  const map: Record<string, number> = {};
  let i = 1;
  for (const c of scenario.comparables) if (c.included) map[c.id] = i++;
  return map;
}
