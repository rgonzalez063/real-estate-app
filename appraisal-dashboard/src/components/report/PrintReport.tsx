import { useMemo } from 'react';
import { useAppStore } from '../../store/useAppStore';
import type { Project, Scenario, ScenarioCalculation } from '../../types';
import { ATTR_LABELS } from '../../engine/fields';
import { orderedRules } from '../../engine/calc';
import { fmtBool, fmtLot, fmtMoney, fmtNumber, fmtPct } from '../../engine/format';
import { formatDate } from '../../engine/dates';
import { compIndexMap } from '../../engine/compIndex';
import { ChartsView } from '../charts/ChartsView';

export function PrintReport({ project, scenario, calc }: { project: Project; scenario: Scenario; calc: ScenarioCalculation }) {
  const setUi = useAppStore((s) => s.setUi);
  const s = scenario.subject.attrs;
  const rounding = scenario.settings.rounding;
  const comps = scenario.comparables.filter((c) => c.included);
  const indexMap = useMemo(() => compIndexMap(scenario), [scenario]);
  const rules = orderedRules(scenario.rules).filter((r) => r.active);
  const r = calc.reconciliation;
  const unresolved = comps.flatMap((c) => calc.comps[c.id].lines.filter((l) => l.status === 'unresolved').map((l) => `Comp ${indexMap[c.id]} (${c.attrs.address ?? '—'}): ${l.label} — ${l.explanation}`));
  const money = (v: number | null) => fmtMoney(v, { rounding });
  const th = 'border border-slate-300 bg-slate-100 px-2 py-1 text-left text-[11px] font-semibold';
  const td = 'border border-slate-300 px-2 py-1 text-[11px] align-top';

  return (
    <div className="min-h-screen bg-white text-slate-900">
      <div className="no-print sticky top-0 z-10 flex items-center gap-2 border-b border-slate-200 bg-slate-50 px-4 py-2">
        <span className="text-sm font-medium">Print-ready report</span>
        <span className="text-xs text-slate-500">Use your browser's print dialog to save as PDF.</span>
        <div className="ml-auto flex gap-2">
          <button className="btn-primary btn-sm" onClick={() => window.print()}>
            Print / Save as PDF
          </button>
          <button className="btn-secondary btn-sm" onClick={() => setUi({ reportOpen: false })}>
            Back to dashboard
          </button>
        </div>
      </div>
      <div className="mx-auto max-w-5xl space-y-6 p-6 text-sm">
        <header className="border-b-2 border-navy-900 pb-3">
          <div className="text-[11px] uppercase tracking-widest text-slate-500">Gonzalez Appraisal Company</div>
          <h1 className="text-2xl font-bold text-navy-900">Sales comparison summary</h1>
          <div className="text-xs text-slate-600">
            Project: {project.name} · Scenario: {scenario.name} · Effective date: {formatDate(scenario.effectiveDate)} · Generated: {new Date().toLocaleString()}
            {project.isDemo && <span className="ml-2 font-semibold text-amber-700">SYNTHETIC DEMO DATA</span>}
          </div>
        </header>

        <section>
          <h2 className="mb-1 text-base font-semibold text-navy-900">Subject property</h2>
          <table className="w-full border-collapse">
            <tbody>
              {(
                [
                  ['Address', [s.address, s.city, s.zip].filter(Boolean).join(', ') || '—'],
                  ['Neighborhood', s.neighborhood ?? '—'],
                  ['Above-grade finished area', s.areaAboveGrade !== null ? `${fmtNumber(s.areaAboveGrade)} SF` : '—'],
                  ['Below-grade finished / other', `${s.areaBelowGradeFinished !== null ? fmtNumber(s.areaBelowGradeFinished) + ' SF' : '—'} / ${s.areaOther !== null ? fmtNumber(s.areaOther) + ' SF' : '—'}`],
                  ['Site', fmtLot(s.lotSizeSqft, s.lotSizeRaw, s.lotSizeUnits)],
                  ['Beds / full baths / half baths', `${s.bedrooms ?? '—'} / ${s.fullBaths ?? '—'} / ${s.halfBaths ?? '—'}`],
                  ['Year built / age', `${s.yearBuilt ?? '—'} / ${s.age ?? '—'}`],
                  ['Garage / carport', `${s.garageSpaces ?? '—'} / ${s.carportSpaces ?? '—'}`],
                  ['Pool / waterfront / water view', `${fmtBool(s.pool)} / ${fmtBool(s.waterfront)} / ${fmtBool(s.waterView)}`],
                  ['Quality / condition / location', `${s.quality ?? '—'} / ${s.condition ?? '—'} / ${s.location ?? '—'}`],
                  ['Appraiser notes', scenario.subject.notes || '—'],
                ] as [string, string][]
              ).map(([k, v]) => (
                <tr key={k}>
                  <th className={`${th} w-56`}>{k}</th>
                  <td className={td}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section>
          <h2 className="mb-1 text-base font-semibold text-navy-900">Adjustment assumptions</h2>
          {rules.length === 0 ? (
            <p className="text-xs text-slate-600">No adjustment rules are active.</p>
          ) : (
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <th className={th}>Category</th>
                  <th className={th}>Method</th>
                  <th className={th}>Rate / values</th>
                  <th className={th}>Reason / support</th>
                  <th className={th}>Source · attachment</th>
                </tr>
              </thead>
              <tbody>
                {rules.map((rule) => (
                  <tr key={rule.id}>
                    <td className={td}>{rule.label}</td>
                    <td className={td}>
                      {rule.kind === 'perUnit' && `(Subject − Comp) × rate on ${rule.field ? ATTR_LABELS[rule.field] : '—'}`}
                      {rule.kind === 'flat' && `Flat amount by ${rule.field ? ATTR_LABELS[rule.field] : '—'}`}
                      {rule.kind === 'percent' && `Percent of sale price by ${rule.field ? ATTR_LABELS[rule.field] : '—'}`}
                      {rule.kind === 'categorical' && `Dollar by rating (${rule.field ? ATTR_LABELS[rule.field] : '—'})`}
                      {rule.kind === 'market' && (rule.market?.mode === 'manual' ? 'Manual per comparable' : `Price × [(1 + rate)^months − 1], months = days ÷ 30.4375, from ${rule.market?.dateBasis === 'contractDate' ? 'contract' : 'sale'} date`)}
                      {rule.kind === 'concessions' && '−(reported concession × factor)'}
                      {rule.kind === 'manual' && 'Manual overrides only'}
                    </td>
                    <td className={td}>
                      {rule.kind === 'perUnit' || rule.kind === 'flat' ? fmtMoney(rule.rate, { decimals: 2 }) : rule.kind === 'percent' && rule.rate !== null ? `${rule.rate}%` : rule.kind === 'market' ? (rule.market?.monthlyRatePct !== null && rule.market?.monthlyRatePct !== undefined ? `${rule.market.monthlyRatePct}% / month` : '—') : rule.kind === 'concessions' ? (rule.concessionFactor ?? '—') : ''}
                      {Object.keys(rule.categoricalValues).length > 0 && Object.entries(rule.categoricalValues).map(([k, v]) => `${k}: ${rule.kind === 'percent' ? v + '%' : fmtMoney(v)}`).join('; ')}
                    </td>
                    <td className={td}>{rule.reason || '—'}</td>
                    <td className={td}>{[rule.source, rule.attachmentRef].filter(Boolean).join(' · ') || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {calc.warnings.length > 0 && (
            <ul className="mt-1 list-disc pl-5 text-[11px] text-amber-800">
              {calc.warnings.map((w) => (
                <li key={w.code + w.message}>{w.message}</li>
              ))}
            </ul>
          )}
        </section>

        <section className="print-page">
          <h2 className="mb-1 text-base font-semibold text-navy-900">Adjustment grid</h2>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <th className={th}>Item</th>
                  <th className={th}>Subject</th>
                  {comps.map((c) => (
                    <th key={c.id} className={th}>
                      Comp {indexMap[c.id]}
                      <div className="font-normal">{c.attrs.address ?? '—'}</div>
                      {c.attrs.status !== 'closed' && <div className="font-normal text-slate-500">context only</div>}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ['Status', '', (c) => c.attrs.statusRaw ?? c.attrs.status ?? '—'],
                    ['Sale date', '', (c) => formatDate(c.attrs.saleDate)],
                    ['Original sale price', '', (c) => fmtMoney(c.attrs.salePrice)],
                    ['Price / SF (above-grade)', '', (c) => (calc.comps[c.id].pricePerSqft !== null ? fmtMoney(calc.comps[c.id].pricePerSqft, { decimals: 2 }) : '—')],
                    ['Above-grade area', s.areaAboveGrade !== null ? `${fmtNumber(s.areaAboveGrade)} SF` : '—', (c) => (c.attrs.areaAboveGrade !== null ? `${fmtNumber(c.attrs.areaAboveGrade)} SF` : '—')],
                    ['Site', fmtLot(s.lotSizeSqft, s.lotSizeRaw, s.lotSizeUnits), (c) => fmtLot(c.attrs.lotSizeSqft, c.attrs.lotSizeRaw, c.attrs.lotSizeUnits)],
                    ['Beds / full / half', `${s.bedrooms ?? '—'} / ${s.fullBaths ?? '—'} / ${s.halfBaths ?? '—'}`, (c) => `${c.attrs.bedrooms ?? '—'} / ${c.attrs.fullBaths ?? '—'} / ${c.attrs.halfBaths ?? '—'}`],
                    ['Year built', s.yearBuilt ?? '—', (c) => c.attrs.yearBuilt ?? '—'],
                    ['Garage / carport', `${s.garageSpaces ?? '—'} / ${s.carportSpaces ?? '—'}`, (c) => `${c.attrs.garageSpaces ?? '—'} / ${c.attrs.carportSpaces ?? '—'}`],
                    ['Pool', fmtBool(s.pool), (c) => fmtBool(c.attrs.pool)],
                    ['Quality / condition', `${s.quality ?? '—'} / ${s.condition ?? '—'}`, (c) => `${c.attrs.quality ?? '—'} / ${c.attrs.condition ?? '—'}`],
                    ['Location', s.location ?? '—', (c) => c.attrs.location ?? '—'],
                    ['Reported concessions', '', (c) => fmtMoney(c.attrs.concessions)],
                  ] as [string, string | number, (c: (typeof comps)[number]) => string | number][]
                ).map(([label, subj, fn]) => (
                  <tr key={label}>
                    <th className={th}>{label}</th>
                    <td className={td}>{subj}</td>
                    {comps.map((c) => (
                      <td key={c.id} className={td}>
                        {fn(c)}
                      </td>
                    ))}
                  </tr>
                ))}
                {rules.map((rule) => (
                  <tr key={rule.id}>
                    <th className={th}>Adj: {rule.label}</th>
                    <td className={td} />
                    {comps.map((c) => {
                      const l = calc.comps[c.id].lines.find((x) => x.ruleId === rule.id);
                      return (
                        <td key={c.id} className={td}>
                          {!l ? '' : l.status === 'unresolved' ? <span className="font-semibold text-amber-700">UNRESOLVED</span> : l.status === 'notApplicable' ? 'N/A' : money(l.amount)}
                          {l && (l.status === 'override' || l.status === 'confirmedZero') && <div className="text-[10px] text-slate-500">manual{l.reason ? `: ${l.reason}` : ''}</div>}
                          {l && l.status === 'computed' && <div className="text-[10px] text-slate-500">{l.formula}</div>}
                        </td>
                      );
                    })}
                  </tr>
                ))}
                {(
                  [
                    ['Net adjustment', (c) => (calc.comps[c.id].unresolved.length ? 'UNRESOLVED' : `${money(calc.comps[c.id].netAdjustment)} (${fmtPct(calc.comps[c.id].netPct, 1, true)})`)],
                    ['Gross adjustment', (c) => (calc.comps[c.id].unresolved.length ? 'UNRESOLVED' : `${money(calc.comps[c.id].grossAdjustment)} (${fmtPct(calc.comps[c.id].grossPct, 1)})`)],
                    ['Adjusted sale price', (c) => (calc.comps[c.id].adjustedPrice !== null ? money(calc.comps[c.id].adjustedPrice) : 'UNRESOLVED')],
                    ['Weight (entered / normalized)', (c) => `${c.weight} / ${calc.comps[c.id].normalizedWeight !== null ? fmtPct((calc.comps[c.id].normalizedWeight as number) * 100, 1) : '—'}`],
                    ['Notes', (c) => c.gridNote || c.selectionNote || '—'],
                  ] as [string, (c: (typeof comps)[number]) => string][]
                ).map(([label, fn]) => (
                  <tr key={label}>
                    <th className={th}>{label}</th>
                    <td className={td} />
                    {comps.map((c) => (
                      <td key={c.id} className={`${td} ${label === 'Adjusted sale price' ? 'font-semibold' : ''}`}>
                        {fn(c)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section>
          <h2 className="mb-1 text-base font-semibold text-navy-900">Reconciliation</h2>
          <table className="w-full max-w-lg border-collapse">
            <tbody>
              {(
                [
                  ['Included closed sales contributing', `${r.contributingCount} of ${r.includedCount}`],
                  ['Excluded for unresolved / missing data', String(r.excludedForUnresolved)],
                  ['Unadjusted price range', r.unadjustedRange ? `${money(r.unadjustedRange[0])} – ${money(r.unadjustedRange[1])}` : '—'],
                  ['Adjusted price range (observed, not a confidence interval)', r.adjustedRange ? `${money(r.adjustedRange[0])} – ${money(r.adjustedRange[1])}` : '—'],
                  ['Adjusted mean / median', `${money(r.adjustedMean)} / ${money(r.adjustedMedian)}`],
                  ['Weighted value indication', money(r.weightedIndication)],
                  ['Final reconciled opinion of value', fmtMoney(scenario.finalOpinion.value)],
                  ['Rounding increment', `$${rounding}`],
                ] as [string, string][]
              ).map(([k, v]) => (
                <tr key={k}>
                  <th className={`${th} w-80`}>{k}</th>
                  <td className={td}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {r.message && <p className="mt-1 text-[11px] text-amber-800">{r.message}</p>}
          <div className="mt-2 text-xs">
            <div className="font-semibold">Rationale</div>
            <p className="whitespace-pre-wrap">{scenario.finalOpinion.rationale || '—'}</p>
            <div className="mt-1 font-semibold">Reconciliation notes</div>
            <p className="whitespace-pre-wrap">{scenario.reconciliationNotes || '—'}</p>
          </div>
          {unresolved.length > 0 && (
            <div className="mt-2 rounded border border-amber-300 bg-amber-50 p-2 text-[11px] text-amber-900">
              <div className="font-semibold">Unresolved items</div>
              <ul className="list-disc pl-5">
                {unresolved.map((u) => (
                  <li key={u}>{u}</li>
                ))}
              </ul>
            </div>
          )}
          <div className="mt-2 text-[11px] text-slate-600">
            <div className="font-semibold">Calculation order</div>
            <ol className="list-decimal pl-5">
              {calc.calculationOrder.map((o) => (
                <li key={o}>{o.replace(/^\d+\.\s*/, '')}</li>
              ))}
            </ol>
          </div>
        </section>

        <section className="print-page">
          <h2 className="mb-1 text-base font-semibold text-navy-900">Charts</h2>
          <ChartsView scenario={scenario} calc={calc} />
        </section>
      </div>
    </div>
  );
}
