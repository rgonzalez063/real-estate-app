import { useMemo, useState } from 'react';
import { useAppStore } from '../../store/useAppStore';
import type { AdjustmentLine, AdjustmentOverride, Comparable, Scenario, ScenarioCalculation } from '../../types';
import { ATTR_LABELS } from '../../engine/fields';
import { fmtBool, fmtLot, fmtMoney, fmtNumber, fmtPct } from '../../engine/format';
import { formatDate } from '../../engine/dates';
import { distanceToSubject } from '../../engine/geo';
import { Badge, Field, NumberInput, Section, TextArea, TextInput } from '../ui/primitives';
import { StatusBadge } from '../comps/CompCard';
import { AttributeForm } from '../subject/AttributeForm';
import { compIndexMap } from '../../engine/compIndex';

function OverrideEditor({ compId, line, existing, onClose }: { compId: string; line: AdjustmentLine; existing: AdjustmentOverride | undefined; onClose: () => void }) {
  const setOverride = useAppStore((s) => s.setOverride);
  const [amount, setAmount] = useState<number | null>(existing?.amount ?? (line.amount ?? null));
  const [reason, setReason] = useState(existing?.reason ?? '');
  const [source, setSource] = useState(existing?.source ?? '');
  const [attachmentRef, setAttachmentRef] = useState(existing?.attachmentRef ?? '');
  const idp = `ov-${compId}-${line.ruleId}`;
  return (
    <div className="mt-2 space-y-2 rounded-md border border-teal-200 bg-teal-50/50 p-2">
      <Field label="Override amount ($, signed)" htmlFor={`${idp}-amt`} hint="Replaces the computed adjustment for this comparable only.">
        <NumberInput id={`${idp}-amt`} value={amount} onChange={setAmount} suffix="$" />
      </Field>
      <Field label="Reason" htmlFor={`${idp}-reason`}>
        <TextInput id={`${idp}-reason`} value={reason} onChange={(v) => setReason(v ?? '')} />
      </Field>
      <div className="grid grid-cols-2 gap-2">
        <Field label="Method / source" htmlFor={`${idp}-src`}>
          <TextInput id={`${idp}-src`} value={source} onChange={(v) => setSource(v ?? '')} />
        </Field>
        <Field label="Attachment ref" htmlFor={`${idp}-att`}>
          <TextInput id={`${idp}-att`} value={attachmentRef} onChange={(v) => setAttachmentRef(v ?? '')} />
        </Field>
      </div>
      <div className="flex justify-end gap-1">
        <button className="btn-secondary btn-sm" onClick={onClose}>
          Cancel
        </button>
        <button
          className="btn-primary btn-sm"
          disabled={amount === null}
          onClick={() => {
            if (amount === null) return;
            setOverride(compId, line.ruleId, { amount, reason, source, attachmentRef });
            onClose();
          }}
        >
          Save override
        </button>
      </div>
    </div>
  );
}

function LineRow({ comp, line, rounding }: { comp: Comparable; line: AdjustmentLine; rounding: number }) {
  const [editing, setEditing] = useState(false);
  const [open, setOpen] = useState(line.status === 'unresolved');
  const setOverride = useAppStore((s) => s.setOverride);
  const setResolution = useAppStore((s) => s.setResolution);
  const tone = line.status === 'unresolved' ? 'amber' : line.status === 'override' ? 'teal' : line.status === 'computed' ? 'navy' : 'slate';
  const statusLabel: Record<AdjustmentLine['status'], string> = { computed: 'Global rule', override: 'Manual override', unresolved: 'Unresolved', notApplicable: 'N/A', confirmedZero: 'Confirmed $0', inactive: 'Inactive' };
  return (
    <li className={`rounded-md border ${line.status === 'unresolved' ? 'border-amber-300 bg-amber-50/40' : 'border-slate-200'} p-2`}>
      <div className="flex items-start justify-between gap-2">
        <button className="min-w-0 flex-1 text-left" onClick={() => setOpen(!open)} aria-expanded={open}>
          <div className="truncate text-sm font-medium text-slate-800">{line.label}</div>
          <div className="mt-0.5">
            <Badge tone={tone}>{statusLabel[line.status]}</Badge>
          </div>
        </button>
        <div className={`whitespace-nowrap text-sm font-semibold tabular-nums ${line.amount !== null && line.amount < 0 ? 'text-red-700' : line.amount ? 'text-teal-700' : 'text-slate-500'}`}>{line.status === 'unresolved' ? '?' : line.status === 'notApplicable' ? 'N/A' : fmtMoney(line.amount, { signed: true, rounding })}</div>
      </div>
      {open && (
        <div className="mt-2 space-y-1 text-[11px] text-slate-600">
          {line.formula && <div className="font-mono text-[10px] text-slate-700">{line.formula}</div>}
          <div>{line.explanation}</div>
          {line.status !== 'inactive' && line.status !== 'override' && line.subjectValue !== null && typeof line.subjectValue !== 'undefined' && (
            <div>
              Subject: <span className="font-medium">{String(line.subjectValue)}</span> · Comparable: <span className="font-medium">{line.compValue === null ? 'unknown' : String(line.compValue)}</span>
            </div>
          )}
          {(line.reason || line.sourceNote || line.attachmentRef) && (
            <div className="text-slate-500">
              {line.reason && <div>Reason: {line.reason}</div>}
              {line.sourceNote && <div>Source: {line.sourceNote}</div>}
              {line.attachmentRef && <div>Attachment: {line.attachmentRef}</div>}
            </div>
          )}
          {line.status !== 'inactive' && !editing && (
            <div className="flex flex-wrap gap-1 pt-1">
              <button className="btn-secondary btn-sm" onClick={() => setEditing(true)}>
                {line.status === 'override' ? 'Edit override' : 'Override'}
              </button>
              {line.status === 'override' && (
                <button className="btn-secondary btn-sm" onClick={() => setOverride(comp.id, line.ruleId, null)}>
                  Clear override
                </button>
              )}
              {line.status !== 'confirmedZero' && line.status !== 'override' && (
                <button className="btn-secondary btn-sm" onClick={() => setResolution(comp.id, line.ruleId, 'confirmedZero')}>
                  Confirm $0
                </button>
              )}
              {line.status !== 'notApplicable' && line.status !== 'override' && (
                <button className="btn-secondary btn-sm" onClick={() => setResolution(comp.id, line.ruleId, 'notApplicable')}>
                  Mark N/A
                </button>
              )}
              {(line.status === 'notApplicable' || line.status === 'confirmedZero') && (
                <button className="btn-secondary btn-sm" onClick={() => setResolution(comp.id, line.ruleId, null)}>
                  Clear resolution
                </button>
              )}
            </div>
          )}
          {editing && <OverrideEditor compId={comp.id} line={line} existing={comp.overrides[line.ruleId]} onClose={() => setEditing(false)} />}
        </div>
      )}
    </li>
  );
}

function CompDetails({ comp, scenario, calc }: { comp: Comparable; scenario: Scenario; calc: ScenarioCalculation }) {
  const toggleIncluded = useAppStore((s) => s.toggleIncluded);
  const setCompWeight = useAppStore((s) => s.setCompWeight);
  const setCompNote = useAppStore((s) => s.setCompNote);
  const setCompAttrs = useAppStore((s) => s.setCompAttrs);
  const removeComparables = useAppStore((s) => s.removeComparables);
  const selectComp = useAppStore((s) => s.selectComp);
  const [showRaw, setShowRaw] = useState(false);
  const [editAttrs, setEditAttrs] = useState(false);
  const c = calc.comps[comp.id];
  const a = comp.attrs;
  const dist = distanceToSubject(scenario.subject.attrs, a);
  const index = compIndexMap(scenario)[comp.id];
  const rounding = scenario.settings.rounding;
  const activeLines = c.lines.filter((l) => l.status !== 'inactive');
  const rows: [string, string][] = [
    ['Status', a.statusRaw ?? a.status ?? '—'],
    ['Sale price', fmtMoney(a.salePrice)],
    ['List price', fmtMoney(a.listPrice)],
    ['Sale date', formatDate(a.saleDate)],
    ['Contract date', formatDate(a.contractDate)],
    ['Days on market', fmtNumber(a.daysOnMarket)],
    ['Above-grade area', a.areaAboveGrade !== null ? `${fmtNumber(a.areaAboveGrade)} SF` : '—'],
    ['Below-grade finished', a.areaBelowGradeFinished !== null ? `${fmtNumber(a.areaBelowGradeFinished)} SF` : '—'],
    ['Other area', a.areaOther !== null ? `${fmtNumber(a.areaOther)} SF` : '—'],
    ['Lot', fmtLot(a.lotSizeSqft, a.lotSizeRaw, a.lotSizeUnits)],
    ['Beds / full / half', `${a.bedrooms ?? '—'} / ${a.fullBaths ?? '—'} / ${a.halfBaths ?? '—'}`],
    ['Year built / age', `${a.yearBuilt ?? '—'} / ${a.age ?? '—'}`],
    ['Garage / carport', `${a.garageSpaces ?? '—'} / ${a.carportSpaces ?? '—'}`],
    ['Pool', fmtBool(a.pool)],
    ['Waterfront / water view', `${fmtBool(a.waterfront)} / ${fmtBool(a.waterView)}`],
    ['Quality / condition', `${a.quality ?? '—'} / ${a.condition ?? '—'}`],
    ['Location', a.location ?? '—'],
    ['Reported concessions', fmtMoney(a.concessions)],
    ['Distance', dist.miles !== null ? `${dist.miles.toFixed(2)} mi (${dist.basis})` : '—'],
    ['Renovation', a.renovation ?? '—'],
    ['Amenities', a.amenities ?? '—'],
  ];
  return (
    <div>
      <div className="relative h-36 bg-gradient-to-br from-navy-100 to-slate-200">
        {a.photoUrl ? <img src={a.photoUrl} alt={a.address ?? 'Property'} className="h-full w-full object-cover" /> : <div className="flex h-full items-center justify-center text-xs uppercase tracking-wide text-navy-300">No photo</div>}
        <div className="absolute left-2 top-2 flex gap-1">
          <StatusBadge status={a.status} raw={a.statusRaw} />
          {index && <Badge tone="navy">Comp {index}</Badge>}
          {comp.synthetic && <Badge tone="amber">Demo</Badge>}
        </div>
        <button className="absolute right-2 top-2 rounded bg-white/90 px-1.5 py-0.5 text-xs shadow" onClick={() => selectComp(null)} aria-label="Close details">
          ✕
        </button>
      </div>
      <div className="space-y-3 p-3">
        <div>
          <div className="text-base font-semibold text-navy-900">{a.address ?? 'Address unknown'}</div>
          <div className="text-xs text-slate-500">
            {a.mlsNumber ? `MLS ${a.mlsNumber}` : 'MLS —'} · {[a.city, a.zip].filter(Boolean).join(' ') || '—'} {a.neighborhood ? `· ${a.neighborhood}` : ''}
          </div>
          <div className="mt-1 text-[11px] text-slate-400">Source: {comp.sourceFile} (row {comp.sourceRowIndex + 1})</div>
        </div>

        <div className="grid grid-cols-2 gap-2 rounded-md bg-navy-50 p-2">
          <div>
            <div className="text-[10px] uppercase tracking-wide text-slate-500">Adjusted price</div>
            <div className="text-lg font-bold tabular-nums text-navy-900">{c.adjustedPrice !== null ? fmtMoney(c.adjustedPrice, { rounding }) : '—'}</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wide text-slate-500">Net / gross</div>
            <div className="text-sm font-semibold tabular-nums">{c.unresolved.length ? '—' : `${fmtMoney(c.netAdjustment, { signed: true, rounding })} / ${fmtMoney(c.grossAdjustment, { rounding })}`}</div>
            <div className="text-[11px] text-slate-500">{c.netPct !== null ? `${fmtPct(c.netPct, 1, true)} / ${fmtPct(c.grossPct, 1)}` : ''}</div>
          </div>
          {c.blockers.length > 0 && (
            <div className="col-span-2 text-[11px] text-amber-800">
              Not in the value indication: {c.blockers.join('; ')}
              {a.status !== 'closed' && ' · not a closed sale'}
            </div>
          )}
          {a.status !== 'closed' && c.blockers.length === 0 && <div className="col-span-2 text-[11px] text-slate-600">Market context only (not a closed sale).</div>}
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-1 text-sm font-medium">
            <input type="checkbox" checked={comp.included} onChange={() => toggleIncluded(comp.id)} /> Included
          </label>
          <div className="flex items-center gap-1">
            <label className="text-xs text-slate-600" htmlFor={`weight-${comp.id}`}>
              Weight
            </label>
            <NumberInput id={`weight-${comp.id}`} value={comp.weight} onChange={(v) => setCompWeight(comp.id, v ?? 0)} min={0} className="w-20" />
            <span className="text-[11px] text-slate-500">{c.normalizedWeight !== null ? `= ${fmtPct(c.normalizedWeight * 100, 1)}` : ''}</span>
          </div>
        </div>
        <Field label="Inclusion / exclusion note" htmlFor={`selnote-${comp.id}`}>
          <TextArea id={`selnote-${comp.id}`} value={comp.selectionNote} onChange={(v) => setCompNote(comp.id, v)} rows={2} placeholder="Why this comparable is included or excluded" />
        </Field>

        {c.flags.length > 0 && (
          <div className="rounded-md border border-amber-200 bg-amber-50 p-2 text-[11px] text-amber-800">
            <div className="font-semibold">Review flags (user-defined thresholds)</div>
            <ul className="list-disc pl-4">
              {c.flags.map((f) => (
                <li key={f.code + f.message}>{f.message}</li>
              ))}
            </ul>
          </div>
        )}

        <div>
          <div className="section-title mb-1">Adjustments ({activeLines.length})</div>
          {activeLines.length ? (
            <ul className="space-y-1">
              {activeLines.map((l) => (
                <LineRow key={l.ruleId} comp={comp} line={l} rounding={rounding} />
              ))}
            </ul>
          ) : (
            <p className="text-[11px] text-slate-500">No active adjustment rules.</p>
          )}
        </div>

        <div>
          <div className="section-title mb-1">Attributes</div>
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
            {rows.map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-slate-500">{k}</dt>
                <dd className="font-medium text-slate-800">{v}</dd>
              </div>
            ))}
          </dl>
          {a.remarks && <p className="mt-2 text-[11px] text-slate-600">{a.remarks}</p>}
        </div>

        {comp.parseIssues.length > 0 && (
          <div className="rounded-md border border-slate-200 bg-slate-50 p-2 text-[11px]">
            <div className="font-semibold text-slate-700">Import notes ({comp.parseIssues.length})</div>
            <ul className="list-disc pl-4 text-slate-600">
              {comp.parseIssues.map((p, i) => (
                <li key={i}>
                  {ATTR_LABELS[p.field as keyof typeof ATTR_LABELS] ?? p.field}: {p.message}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="flex flex-wrap gap-1">
          <button className="btn-secondary btn-sm" onClick={() => setEditAttrs(!editAttrs)} aria-expanded={editAttrs}>
            {editAttrs ? 'Hide editor' : 'Edit attributes'}
          </button>
          <button className="btn-secondary btn-sm" onClick={() => setShowRaw(!showRaw)} aria-expanded={showRaw}>
            {showRaw ? 'Hide source row' : 'Show source row'}
          </button>
          <button
            className="btn-danger btn-sm"
            onClick={() => {
              if (window.confirm('Remove this comparable from the scenario? This can be undone.')) {
                removeComparables([comp.id]);
                selectComp(null);
              }
            }}
          >
            Remove
          </button>
        </div>
        {editAttrs && (
          <div className="rounded-md border border-slate-200 p-2">
            <AttributeForm attrs={a} onChange={(p) => setCompAttrs(comp.id, p)} mode="comp" idPrefix={`comp-${comp.id}`} />
          </div>
        )}
        {showRaw && (
          <div className="max-h-64 overflow-auto rounded-md border border-slate-200">
            <table className="w-full text-[11px]">
              <tbody>
                {Object.entries(comp.raw).map(([k, v]) => (
                  <tr key={k} className="border-b border-slate-100">
                    <td className="bg-slate-50 px-2 py-0.5 font-medium text-slate-600">{k}</td>
                    <td className="px-2 py-0.5 text-slate-800">{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export function ReconciliationPanel({ scenario, calc }: { scenario: Scenario; calc: ScenarioCalculation }) {
  const setFinalOpinion = useAppStore((s) => s.setFinalOpinion);
  const setReconciliationNotes = useAppStore((s) => s.setReconciliationNotes);
  const setCompWeight = useAppStore((s) => s.setCompWeight);
  const selectComp = useAppStore((s) => s.selectComp);
  const r = calc.reconciliation;
  const rounding = scenario.settings.rounding;
  const indexMap = useMemo(() => compIndexMap(scenario), [scenario]);
  const contributors = scenario.comparables.filter((c) => c.included && c.attrs.status === 'closed');
  const [showOrder, setShowOrder] = useState(false);
  return (
    <div className="space-y-3 p-3">
      <div className="rounded-md bg-navy-900 p-3 text-white">
        <div className="text-[10px] uppercase tracking-wide text-slate-300">Weighted value indication</div>
        <div className="text-2xl font-bold tabular-nums text-teal-300">{r.weightedIndication !== null ? fmtMoney(r.weightedIndication, { rounding }) : '—'}</div>
        <div className="text-[11px] text-slate-300">
          {r.contributingCount} of {r.includedCount} included closed sale{r.includedCount === 1 ? '' : 's'} contributing
          {r.excludedForUnresolved > 0 && <span className="text-amber-300"> · {r.excludedForUnresolved} excluded (unresolved / missing)</span>}
        </div>
        {r.message && <div className="mt-1 text-[11px] text-amber-200">{r.message}</div>}
      </div>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
        <dt className="text-slate-500">Unadjusted range</dt>
        <dd className="font-medium tabular-nums">{r.unadjustedRange ? `${fmtMoney(r.unadjustedRange[0], { rounding })} – ${fmtMoney(r.unadjustedRange[1], { rounding })}` : '—'}</dd>
        <dt className="text-slate-500">Adjusted range</dt>
        <dd className="font-medium tabular-nums">{r.adjustedRange ? `${fmtMoney(r.adjustedRange[0], { rounding })} – ${fmtMoney(r.adjustedRange[1], { rounding })}` : '—'}</dd>
        <dt className="text-slate-500">Adjusted mean</dt>
        <dd className="font-medium tabular-nums">{fmtMoney(r.adjustedMean, { rounding })}</dd>
        <dt className="text-slate-500">Adjusted median</dt>
        <dd className="font-medium tabular-nums">{fmtMoney(r.adjustedMedian, { rounding })}</dd>
      </dl>
      <p className="text-[10px] text-slate-500">The range shown is the observed range of adjusted prices, not a statistical confidence interval.</p>
      {contributors.length > 0 && (
        <div>
          <div className="section-title mb-1">Weights</div>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-slate-500">
                <th className="font-medium">Comp</th>
                <th className="font-medium">Adjusted</th>
                <th className="font-medium">Weight</th>
                <th className="font-medium">Norm.</th>
              </tr>
            </thead>
            <tbody>
              {contributors.map((c) => {
                const cc = calc.comps[c.id];
                return (
                  <tr key={c.id} className="border-t border-slate-100">
                    <td>
                      <button className="text-left text-navy-800 hover:underline" onClick={() => selectComp(c.id)}>
                        Comp {indexMap[c.id]}
                      </button>
                    </td>
                    <td className="tabular-nums">{cc.adjustedPrice !== null ? fmtMoney(cc.adjustedPrice, { rounding }) : <Badge tone="amber">Unresolved</Badge>}</td>
                    <td>
                      <NumberInput ariaLabel={`Weight for comp ${indexMap[c.id]}`} value={c.weight} onChange={(v) => setCompWeight(c.id, v ?? 0)} min={0} className="w-16" />
                    </td>
                    <td className="tabular-nums">{cc.normalizedWeight !== null ? fmtPct(cc.normalizedWeight * 100, 1) : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="mt-1 text-[10px] text-slate-500">Weighted indication = Σ(Adjusted price × Weight) ÷ Σ Weights. Weights default to 1 (equal) and are normalized automatically.</p>
        </div>
      )}
      <div className="rounded-md border border-teal-200 bg-teal-50/40 p-2">
        <Field label="Final reconciled opinion of value ($)" htmlFor="final-value" hint="Separate from the weighted indication; the appraiser's conclusion.">
          <NumberInput id="final-value" value={scenario.finalOpinion.value} onChange={(v) => setFinalOpinion({ value: v })} suffix="$" />
        </Field>
        <div className="mt-2">
          <Field label="Rationale" htmlFor="final-rationale">
            <TextArea id="final-rationale" value={scenario.finalOpinion.rationale} onChange={(v) => setFinalOpinion({ rationale: v })} rows={3} />
          </Field>
        </div>
      </div>
      <Field label="Reconciliation notes" htmlFor="recon-notes">
        <TextArea id="recon-notes" value={scenario.reconciliationNotes} onChange={setReconciliationNotes} rows={3} />
      </Field>
      <button className="text-[11px] text-navy-700 underline" onClick={() => setShowOrder(!showOrder)} aria-expanded={showOrder}>
        {showOrder ? 'Hide' : 'Show'} calculation order
      </button>
      {showOrder && (
        <ol className="space-y-1 text-[11px] text-slate-600">
          {calc.calculationOrder.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ol>
      )}
    </div>
  );
}

export function RightPanel({ scenario, calc }: { scenario: Scenario; calc: ScenarioCalculation }) {
  const selectedId = useAppStore((s) => s.ui.selectedCompId);
  const comp = selectedId ? scenario.comparables.find((c) => c.id === selectedId) : undefined;
  return (
    <div className="h-full overflow-y-auto">
      <Section title={comp ? 'Selected property' : 'Property details'} right={undefined}>
        {comp ? <div className="-mx-3 -mb-3"><CompDetails comp={comp} scenario={scenario} calc={calc} /></div> : <p className="text-xs text-slate-500">Select a card, grid column, or chart point to see details, adjustments and overrides.</p>}
      </Section>
      <Section title="Reconciliation summary" id="reconciliation-section">
        <div className="-mx-3 -mb-3">
          <ReconciliationPanel scenario={scenario} calc={calc} />
        </div>
      </Section>
    </div>
  );
}
