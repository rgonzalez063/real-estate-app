import { useMemo } from 'react';
import { useAppStore } from '../../store/useAppStore';
import type { Project } from '../../types';
import { computeScenario } from '../../engine/calc';
import { fmtMoney, fmtNumber } from '../../engine/format';
import { formatDate } from '../../engine/dates';
import { Modal } from '../ui/primitives';

export function ScenarioCompare({ project }: { project: Project }) {
  const { ui, setUi, setActiveScenario } = useAppStore();
  const rows = useMemo(() => project.scenarios.map((s) => ({ s, calc: computeScenario(s) })), [project]);
  const money = (v: number | null, rounding: number) => fmtMoney(v, { rounding });
  return (
    <Modal open={ui.compareOpen} onClose={() => setUi({ compareOpen: false })} title="Scenario comparison" wide>
      <div className="overflow-x-auto">
        <table className="grid-table w-full">
          <thead>
            <tr>
              <th className="sticky-col text-left">Metric</th>
              {rows.map(({ s }) => (
                <th key={s.id} className="min-w-[12rem] text-left">
                  <button className="text-navy-900 hover:underline" onClick={() => { setActiveScenario(s.id); setUi({ compareOpen: false }); }}>
                    {s.name}
                  </button>
                  {s.id === project.activeScenarioId && <span className="ml-1 text-[10px] text-teal-700">(active)</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(
              [
                ['Effective date', ({ s }) => formatDate(s.effectiveDate)],
                ['Subject area (above-grade)', ({ s }) => (s.subject.attrs.areaAboveGrade !== null ? `${fmtNumber(s.subject.attrs.areaAboveGrade)} SF` : '—')],
                ['Comparables imported', ({ s }) => String(s.comparables.length)],
                ['Included closed sales', ({ calc }) => `${calc.reconciliation.contributingCount} of ${calc.reconciliation.includedCount}`],
                ['Active adjustment rules', ({ s }) => String(s.rules.filter((r) => r.active).length)],
                ['Manual overrides', ({ s }) => String(s.comparables.reduce((a, c) => a + Object.keys(c.overrides).length, 0))],
                ['Unadjusted range', ({ s, calc }) => (calc.reconciliation.unadjustedRange ? `${money(calc.reconciliation.unadjustedRange[0], s.settings.rounding)} – ${money(calc.reconciliation.unadjustedRange[1], s.settings.rounding)}` : '—')],
                ['Adjusted range', ({ s, calc }) => (calc.reconciliation.adjustedRange ? `${money(calc.reconciliation.adjustedRange[0], s.settings.rounding)} – ${money(calc.reconciliation.adjustedRange[1], s.settings.rounding)}` : '—')],
                ['Adjusted mean', ({ s, calc }) => money(calc.reconciliation.adjustedMean, s.settings.rounding)],
                ['Adjusted median', ({ s, calc }) => money(calc.reconciliation.adjustedMedian, s.settings.rounding)],
                ['Weighted indication', ({ s, calc }) => money(calc.reconciliation.weightedIndication, s.settings.rounding)],
                ['Final opinion', ({ s }) => fmtMoney(s.finalOpinion.value)],
                ['Warnings', ({ calc }) => String(calc.warnings.length)],
                ['Notes', ({ s }) => s.reconciliationNotes || '—'],
              ] as [string, (r: (typeof rows)[number]) => string][]
            ).map(([label, fn]) => (
              <tr key={label}>
                <th scope="row" className="sticky-col text-left font-medium">
                  {label}
                </th>
                {rows.map((r) => (
                  <td key={r.s.id} className="tabular-nums">
                    {fn(r)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Modal>
  );
}
