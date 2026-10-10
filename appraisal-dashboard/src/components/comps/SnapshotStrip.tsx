import { useMemo } from 'react';
import { useAppStore } from '../../store/useAppStore';
import type { Scenario, ScenarioCalculation } from '../../types';
import { fmtMoney, fmtNumber } from '../../engine/format';
import { compIndexMap } from '../../engine/compIndex';

/**
 * "Subject vs comparable sales" snapshot: the subject on the left, each included sale on the
 * right with up/down markers per adjustment line, and the estimated value range underneath.
 * ↑ = the sale is inferior (adjustment added); ↓ = superior (subtracted); = no difference.
 */
export function SnapshotStrip({ scenario, calc }: { scenario: Scenario; calc: ScenarioCalculation }) {
  const selectComp = useAppStore((s) => s.selectComp);
  const setUi = useAppStore((s) => s.setUi);
  const indexMap = useMemo(() => compIndexMap(scenario), [scenario]);
  const comps = scenario.comparables.filter((c) => c.included && c.attrs.status === 'closed');
  const s = scenario.subject.attrs;
  const r = calc.reconciliation;
  const rounding = scenario.settings.rounding;
  const subjectPhoto = scenario.subject.photoData ?? scenario.subject.photoUrls[0] ?? null;
  if (!comps.length) return null;
  const baths = s.fullBaths === null && s.halfBaths === null ? '—' : `${s.fullBaths ?? '?'}${s.halfBaths ? `.${s.halfBaths}` : ''} BA`;
  return (
    <div className="card mb-3 overflow-hidden">
      <div className="grid grid-cols-1 md:grid-cols-[14rem_1fr]">
        <div className="bg-navy-900 p-3 text-white">
          <div className="text-[10px] font-semibold uppercase tracking-widest text-slate-300">Subject property</div>
          <div className="mt-2 h-28 overflow-hidden rounded bg-navy-800">
            {subjectPhoto ? <img src={subjectPhoto} alt="Subject" className="h-full w-full object-cover" /> : <div className="flex h-full items-center justify-center text-[11px] text-slate-400">Drop a photo on the sales grid</div>}
          </div>
          <div className="mt-2 truncate text-sm font-semibold" title={s.address ?? undefined}>
            {s.address ?? 'Subject address'}
          </div>
          <div className="text-xs text-slate-300">
            {s.bedrooms ?? '—'} BD | {baths} | {s.areaAboveGrade !== null ? `${fmtNumber(s.areaAboveGrade)} sq ft` : '— sq ft'}
          </div>
        </div>
        <div className="p-3">
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-500">Comparable sales</div>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {comps.map((c) => {
              const cc = calc.comps[c.id];
              const lines = cc.lines.filter((l) => l.status !== 'inactive' && l.status !== 'notApplicable');
              const photo = c.photoData ?? c.attrs.photoUrl ?? null;
              return (
                <button key={c.id} className="w-44 shrink-0 rounded-md border border-slate-200 text-left hover:border-teal-400" onClick={() => { selectComp(c.id); setUi({ mobileTab: 'details' }); }}>
                  <div className="relative h-20 overflow-hidden rounded-t bg-slate-100">
                    {photo ? <img src={photo} alt="" className="h-full w-full object-cover" /> : <div className="flex h-full items-center justify-center text-[10px] uppercase text-slate-400">No photo</div>}
                    <span className="absolute left-1 top-1 rounded bg-red-700 px-1.5 text-[10px] font-bold uppercase text-white">Sold</span>
                    <span className="absolute right-1 top-1 rounded bg-navy-900/80 px-1.5 text-[10px] text-white">#{indexMap[c.id]}</span>
                  </div>
                  <div className="px-2 py-1">
                    <div className="text-sm font-bold tabular-nums text-navy-900">{fmtMoney(c.attrs.salePrice)}</div>
                    <ul className="mt-1 space-y-0.5 text-[11px]">
                      {lines.slice(0, 6).map((l) => {
                        const up = l.amount !== null && l.amount > 0;
                        const down = l.amount !== null && l.amount < 0;
                        return (
                          <li key={l.ruleId} className="flex items-center justify-between gap-1">
                            <span className="truncate text-slate-600">{l.label}</span>
                            <span className={`font-semibold ${l.status === 'unresolved' ? 'text-amber-600' : up ? 'text-teal-700' : down ? 'text-red-700' : 'text-slate-400'}`} title={l.status === 'unresolved' ? 'Needs input' : fmtMoney(l.amount, { signed: true })}>
                              {l.status === 'unresolved' ? '?' : up ? '+/↑' : down ? '−/↓' : '='}
                            </span>
                          </li>
                        );
                      })}
                      {lines.length === 0 && <li className="text-slate-400">No active adjustments</li>}
                    </ul>
                    <div className="mt-1 border-t border-slate-100 pt-1 text-[11px] text-slate-600">
                      Adjusted <span className="font-semibold tabular-nums text-teal-700">{cc.adjustedPrice !== null ? fmtMoney(cc.adjustedPrice, { rounding }) : '—'}</span>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-3 rounded-md bg-navy-50 px-3 py-2">
            <div className="text-[10px] font-semibold uppercase tracking-widest text-navy-700">Estimated market value range</div>
            <div className="text-lg font-bold tabular-nums text-navy-900">{r.adjustedRange ? `${fmtMoney(r.adjustedRange[0], { rounding })} – ${fmtMoney(r.adjustedRange[1], { rounding })}` : '—'}</div>
            <div className="text-xs text-slate-600">
              Weighted indication <span className="font-semibold text-teal-700">{r.weightedIndication !== null ? fmtMoney(r.weightedIndication, { rounding }) : '—'}</span> · observed range of adjusted prices, not a confidence interval
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
