import { useMemo, type ReactNode } from 'react';
import { useAppStore } from '../../store/useAppStore';
import type { AdjustmentLine, Comparable, Scenario, ScenarioCalculation } from '../../types';
import { ATTR_LABELS } from '../../engine/fields';
import { orderedRules } from '../../engine/calc';
import { fmtBool, fmtLot, fmtMoney, fmtNumber, fmtPct } from '../../engine/format';
import { formatDate } from '../../engine/dates';
import { Badge, EmptyState, NumberInput } from '../ui/primitives';
import { StatusBadge } from '../comps/CompCard';

function LineCell({ line, rounding }: { line: AdjustmentLine; rounding: number }) {
  if (line.status === 'inactive') return <span className="text-slate-300">—</span>;
  if (line.status === 'unresolved')
    return (
      <Badge tone="amber" title={line.explanation}>
        Unresolved
      </Badge>
    );
  if (line.status === 'notApplicable') return <span className="text-slate-400" title={line.explanation}>N/A</span>;
  return (
    <span title={`${line.formula}\n${line.explanation}`}>
      <span className={`tabular-nums ${line.amount && line.amount < 0 ? 'text-red-700' : line.amount && line.amount > 0 ? 'text-teal-700' : 'text-slate-600'}`}>{fmtMoney(line.amount, { signed: true, rounding })}</span>
      <span className="ml-1 text-[10px] uppercase text-slate-400">{line.source === 'manual' ? (line.status === 'confirmedZero' ? 'zero' : 'override') : 'rule'}</span>
    </span>
  );
}

export function ComparisonGrid({ scenario, calc }: { scenario: Scenario; calc: ScenarioCalculation }) {
  const selectedId = useAppStore((s) => s.ui.selectedCompId);
  const selectComp = useAppStore((s) => s.selectComp);
  const setCompWeight = useAppStore((s) => s.setCompWeight);
  const setCompGridNote = useAppStore((s) => s.setCompGridNote);
  const toggleIncluded = useAppStore((s) => s.toggleIncluded);
  const comps = useMemo(() => scenario.comparables.filter((c) => c.included), [scenario]);
  const rules = orderedRules(scenario.rules).filter((r) => r.active);
  const s = scenario.subject.attrs;
  const rounding = scenario.settings.rounding;

  if (!comps.length) {
    return (
      <EmptyState title="No comparables included">
        <p>Include comparables from the Cards view to populate the adjustment grid.</p>
      </EmptyState>
    );
  }

  const row = (label: string, subjectCell: ReactNode, cell: (c: Comparable) => ReactNode, cls = '') => (
    <tr className={cls}>
      <th scope="row" className="sticky-col w-44 min-w-[11rem] text-left font-medium">
        {label}
      </th>
      <td className="min-w-[10rem] bg-navy-50/40">{subjectCell}</td>
      {comps.map((c) => (
        <td key={c.id} className={`min-w-[11rem] cursor-pointer ${selectedId === c.id ? 'bg-teal-50' : ''}`} onClick={() => selectComp(c.id)}>
          {cell(c)}
        </td>
      ))}
    </tr>
  );
  const attrRow = (key: keyof typeof ATTR_LABELS, fmt: (v: unknown) => string = (v) => (v === null || v === undefined ? '—' : String(v))) => row(ATTR_LABELS[key], fmt(s[key]), (c) => fmt(c.attrs[key]));

  return (
    <div className="card overflow-hidden">
      <div className="max-h-[calc(100vh-14rem)] overflow-auto">
        <table className="grid-table w-full border-collapse">
          <thead>
            <tr>
              <th className="sticky-col text-left">Item</th>
              <th className="min-w-[10rem] text-left">Subject</th>
              {comps.map((c, i) => (
                <th key={c.id} className={`min-w-[11rem] text-left ${selectedId === c.id ? 'bg-teal-50' : ''}`}>
                  <button className="w-full text-left" onClick={() => selectComp(c.id)} aria-label={`Select comp ${i + 1}`}>
                    <div className="font-semibold text-navy-900">Comp {i + 1}</div>
                    <div className="truncate text-xs font-normal text-slate-600" title={c.attrs.address ?? ''}>
                      {c.attrs.address ?? '—'}
                    </div>
                    <div className="mt-1 flex items-center gap-1">
                      <StatusBadge status={c.attrs.status} raw={c.attrs.statusRaw} />
                      {c.attrs.status !== 'closed' && <span className="text-[10px] text-slate-500">context only</span>}
                    </div>
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {row('Included', '', (c) => (
              <label className="flex items-center gap-1 text-xs" onClick={(e) => e.stopPropagation()}>
                <input type="checkbox" checked={c.included} onChange={() => toggleIncluded(c.id)} aria-label={`Include ${c.attrs.address ?? 'comparable'}`} /> Included
              </label>
            ))}
            {attrRow('mlsNumber')}
            {row('Sale date', '', (c) => formatDate(c.attrs.saleDate))}
            {row('Contract date', '', (c) => formatDate(c.attrs.contractDate))}
            {row('Original transaction price', '', (c) => <span className="font-semibold tabular-nums">{fmtMoney(c.attrs.salePrice)}</span>, 'bg-slate-50/60')}
            {row('List price', '', (c) => fmtMoney(c.attrs.listPrice))}
            {row('Price / SF (above-grade)', '', (c) => (calc.comps[c.id].pricePerSqft !== null ? fmtMoney(calc.comps[c.id].pricePerSqft, { decimals: 2 }) : '—'))}
            {row('Reported concessions', '', (c) => fmtMoney(c.attrs.concessions))}
            {row('Days on market', '', (c) => fmtNumber(c.attrs.daysOnMarket))}
            <tr>
              <th scope="row" className="sticky-col section-title bg-slate-50 py-1" colSpan={1}>Characteristics</th>
              <td className="bg-slate-50" colSpan={comps.length + 1} />
            </tr>
            {attrRow('areaAboveGrade', (v) => (v === null ? '—' : `${fmtNumber(v as number)} SF`))}
            {attrRow('areaBelowGradeFinished', (v) => (v === null ? '—' : `${fmtNumber(v as number)} SF`))}
            {attrRow('areaOther', (v) => (v === null ? '—' : `${fmtNumber(v as number)} SF`))}
            {row('Lot size', fmtLot(s.lotSizeSqft, s.lotSizeRaw, s.lotSizeUnits), (c) => fmtLot(c.attrs.lotSizeSqft, c.attrs.lotSizeRaw, c.attrs.lotSizeUnits))}
            {attrRow('bedrooms')}
            {attrRow('fullBaths')}
            {attrRow('halfBaths')}
            {attrRow('yearBuilt')}
            {attrRow('garageSpaces')}
            {attrRow('carportSpaces')}
            {attrRow('pool', (v) => fmtBool(v as boolean | null))}
            {attrRow('waterfront', (v) => fmtBool(v as boolean | null))}
            {attrRow('waterView', (v) => fmtBool(v as boolean | null))}
            {attrRow('quality')}
            {attrRow('condition')}
            {attrRow('location')}
            <tr>
              <th scope="row" className="sticky-col section-title bg-slate-50 py-1">Adjustments</th>
              <td className="bg-slate-50" colSpan={comps.length + 1}>
                <span className="text-[11px] text-slate-500">{rules.length ? 'Hover a value for its formula. "rule" = global assumption, "override" = manual replacement.' : 'No active adjustment rules. Activate categories in the left panel.'}</span>
              </td>
            </tr>
            {rules.map((r) =>
              row(r.label, r.kind === 'market' ? formatDate(scenario.effectiveDate) : r.field ? String(s[r.field] ?? '—') : '', (c) => {
                const line = calc.comps[c.id].lines.find((l) => l.ruleId === r.id);
                return line ? <LineCell line={line} rounding={rounding} /> : null;
              }),
            )}
            {row('Net adjustment', '', (c) => (calc.comps[c.id].unresolved.length ? <Badge tone="amber">Unresolved</Badge> : <span className="font-semibold tabular-nums">{fmtMoney(calc.comps[c.id].netAdjustment, { signed: true, rounding })}</span>), 'bg-slate-50/60')}
            {row('Gross adjustment', '', (c) => (calc.comps[c.id].unresolved.length ? '—' : <span className="tabular-nums">{fmtMoney(calc.comps[c.id].grossAdjustment, { rounding })}</span>))}
            {row('Net adjustment %', '', (c) => fmtPct(calc.comps[c.id].netPct, 1, true))}
            {row('Gross adjustment %', '', (c) => fmtPct(calc.comps[c.id].grossPct, 1))}
            {row('Adjusted sale price', '', (c) => (calc.comps[c.id].adjustedPrice !== null ? <span className="text-base font-bold tabular-nums text-teal-700">{fmtMoney(calc.comps[c.id].adjustedPrice, { rounding })}</span> : <Badge tone="amber">Not calculable</Badge>), 'bg-teal-50/40')}
            {row('Weight (entered)', '', (c) => (
              <div onClick={(e) => e.stopPropagation()}>
                <NumberInput ariaLabel={`Weight for ${c.attrs.address ?? 'comparable'}`} value={c.weight} onChange={(v) => setCompWeight(c.id, v ?? 0)} min={0} className="w-24" />
              </div>
            ))}
            {row('Weight (normalized)', '', (c) => (calc.comps[c.id].normalizedWeight !== null ? fmtPct((calc.comps[c.id].normalizedWeight as number) * 100, 1) : <span className="text-slate-400">—</span>))}
            {row('Review flags', '', (c) => (calc.comps[c.id].flags.length ? <ul className="list-disc pl-4 text-[11px] text-amber-800">{calc.comps[c.id].flags.map((f) => <li key={f.code + f.message}>{f.message}</li>)}</ul> : <span className="text-slate-400">None</span>))}
            {row('Notes', scenario.subject.notes || '', (c) => (
              <div onClick={(e) => e.stopPropagation()}>
                <textarea aria-label={`Notes for ${c.attrs.address ?? 'comparable'}`} className="input text-xs" rows={2} value={c.gridNote} onChange={(e) => setCompGridNote(c.id, e.target.value)} placeholder={c.selectionNote || 'Grid note'} />
              </div>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
