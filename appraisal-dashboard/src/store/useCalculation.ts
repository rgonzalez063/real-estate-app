import { useMemo } from 'react';
import { computeScenario } from '../engine/calc';
import type { Scenario, ScenarioCalculation } from '../types';

const EMPTY: ScenarioCalculation = {
  comps: {},
  reconciliation: { includedCount: 0, contributingCount: 0, excludedForUnresolved: 0, unadjustedRange: null, adjustedRange: null, adjustedMean: null, adjustedMedian: null, weightedIndication: null, totalWeight: 0, message: 'No scenario loaded.' },
  warnings: [],
  calculationOrder: [],
};

/** Memoised shared calculation for the active scenario. */
export function useCalculation(scenario: Scenario | null): ScenarioCalculation {
  return useMemo(() => (scenario ? computeScenario(scenario) : EMPTY), [scenario]);
}
