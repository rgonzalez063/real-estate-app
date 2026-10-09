import type { Scenario, ScenarioCalculation } from '../../types';
import { fmtMoney } from '../../engine/format';
import { Stat } from '../ui/primitives';

export function SummaryBar({ scenario, calc }: { scenario: Scenario; calc: ScenarioCalculation }) {
  const r = calc.reconciliation;
  const rounding = scenario.settings.rounding;
  const range = r.adjustedRange ? `${fmtMoney(r.adjustedRange[0], { rounding })} – ${fmtMoney(r.adjustedRange[1], { rounding })}` : '—';
  return (
    <div className="no-print grid grid-cols-2 gap-x-4 gap-y-1 border-t border-navy-700 bg-navy-800 px-3 py-1.5 sm:flex sm:flex-wrap sm:items-center sm:gap-x-6" role="status" aria-live="polite">
      <Stat label="Weighted indication" value={r.weightedIndication !== null ? fmtMoney(r.weightedIndication, { rounding }) : '—'} tone="accent" sub={r.message ?? undefined} />
      <Stat label="Adjusted price range" value={range} sub="observed range, not a confidence interval" />
      <Stat label="Included closed sales" value={`${r.contributingCount} of ${r.includedCount}`} sub={r.excludedForUnresolved ? `${r.excludedForUnresolved} unresolved` : 'contributing'} />
      <Stat label="Adjusted median" value={fmtMoney(r.adjustedMedian, { rounding })} />
      <Stat label="Final opinion" value={scenario.finalOpinion.value !== null ? fmtMoney(scenario.finalOpinion.value) : 'not entered'} tone={scenario.finalOpinion.value !== null ? 'default' : 'muted'} />
      {calc.warnings.length > 0 && (
        <div className="text-[11px] text-amber-300 sm:ml-auto" title={calc.warnings.map((w) => w.message).join('\n')}>
          ⚠ {calc.warnings.length} warning{calc.warnings.length > 1 ? 's' : ''}
        </div>
      )}
    </div>
  );
}
