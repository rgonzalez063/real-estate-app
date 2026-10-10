import { Fragment, useMemo, useState, type ReactNode } from 'react';
import { useAppStore } from '../../store/useAppStore';
import type { AdjustmentLine, AdjustmentRule, Comparable, LotUnits, PropertyAttributes, Scenario, ScenarioCalculation } from '../../types';
import { customRows, FORM_ROWS, type GridRow } from '../../engine/gridRows';
import { orderedRules } from '../../engine/calc';
import { fmtMoney, fmtNumber, fmtPct } from '../../engine/format';
import { formatDate } from '../../engine/dates';
import { distanceToSubject } from '../../engine/geo';
import { SQFT_PER_ACRE } from '../../engine/parse';
import { fileToResizedDataUrl } from '../../engine/imageResize';
import { Badge, EmptyState, NumberInput, TextArea, TextInput, TriState } from '../ui/primitives';
import { StatusBadge } from '../comps/CompCard';

const LABEL_W = 184;
const SUBJECT_W = 168;
const compact = '!px-1.5 !py-0 !h-7 !text-xs !shadow-none !rounded';

/* ---------------- photo drop box ---------------- */
function PhotoBox({ src, alt, onDrop }: { src: string | null; alt: string; onDrop: (dataUrl: string | null) => void }) {
  const [over, setOver] = useState(false);
  const toast = useAppStore((s) => s.toast);
  const handle = async (file: File | undefined) => {
    if (!file) return;
    try {
      onDrop(await fileToResizedDataUrl(file));
    } catch {
      toast('Could not read that image.', 'error');
    }
  };
  return (
    <div
      className={`relative h-[72px] w-24 shrink-0 overflow-hidden rounded border bg-slate-100 ${over ? 'border-teal-500 ring-2 ring-teal-300' : 'border-slate-300'}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); void handle(e.dataTransfer.files[0]); }}
    >
      {src ? (
        <>
          <img src={src} alt={alt} className="h-full w-full object-cover" />
          <button className="absolute right-0.5 top-0.5 rounded bg-white/90 px-1 text-[10px] leading-4 text-slate-700 shadow" onClick={() => onDrop(null)} aria-label="Remove photo">
            ✕
          </button>
        </>
      ) : (
        <label className="relative flex h-full cursor-pointer flex-col items-center justify-center text-center text-[10px] leading-tight text-slate-500">
          <span>Drag photo here</span>
          <span className="text-teal-700 underline">or browse</span>
          <input type="file" accept="image/*" className="sr-only" onChange={(e) => void handle(e.target.files?.[0])} />
        </label>
      )}
    </div>
  );
}

/* ---------------- description cell ---------------- */
function DescCell({ row, attrs, onChange, id }: { row: GridRow; attrs: PropertyAttributes; onChange: (p: Partial<PropertyAttributes>) => void; id: string }) {
  if (!row.field) return null;
  const f = row.field;
  const label = `${row.label}`;
  switch (row.input) {
    case 'number':
      return <NumberInput id={id} ariaLabel={label} value={attrs[f] as number | null} onChange={(v) => onChange({ [f]: v })} className={compact} />;
    case 'currency':
      return <NumberInput id={id} ariaLabel={label} value={attrs[f] as number | null} onChange={(v) => onChange({ [f]: v })} className={compact} suffix="$" />;
    case 'date':
      return <TextInput id={id} ariaLabel={label} type="date" value={attrs[f] as string | null} onChange={(v) => onChange({ [f]: v })} className={compact} />;
    case 'boolean':
      return <div className="[&>select]:!h-7 [&>select]:!py-0 [&>select]:!text-xs [&>select]:!shadow-none"><TriState id={id} ariaLabel={label} value={attrs[f] as boolean | null} onChange={(v) => onChange({ [f]: v })} /></div>;
    case 'lot': {
      const setLot = (raw: number | null, units: LotUnits | null) => onChange({ lotSizeRaw: raw, lotSizeUnits: units, lotSizeSqft: raw === null || units === null ? null : units === 'acres' ? raw * SQFT_PER_ACRE : raw });
      return (
        <div className="flex gap-0.5">
          <NumberInput id={id} ariaLabel={`${label} size`} value={attrs.lotSizeRaw} onChange={(v) => setLot(v, attrs.lotSizeUnits)} className={`${compact} min-w-0 flex-1`} />
          <select aria-label={`${label} units`} className={`input w-14 ${compact}`} value={attrs.lotSizeUnits ?? ''} onChange={(e) => setLot(attrs.lotSizeRaw, (e.target.value || null) as LotUnits | null)}>
            <option value="">?</option>
            <option value="sqft">sf</option>
            <option value="acres">ac</option>
          </select>
        </div>
      );
    }
    default:
      return <TextInput id={id} ariaLabel={label} value={attrs[f] as string | null} onChange={(v) => onChange({ [f]: v })} className={compact} />;
  }
}

/* ---------------- adjustment cell ---------------- */
function AdjCell({ comp, rule, line, rounding }: { comp: Comparable; rule: AdjustmentRule; line: AdjustmentLine | undefined; rounding: number }) {
  const setOverride = useAppStore((s) => s.setOverride);
  const setResolution = useAppStore((s) => s.setResolution);
  const override = comp.overrides[rule.id];
  const status = line?.status ?? 'inactive';
  const placeholder = status === 'computed' ? fmtMoney(line?.amount ?? 0, { signed: true, rounding }) : status === 'unresolved' ? '? enter $' : status === 'notApplicable' ? 'N/A' : status === 'confirmedZero' ? '$0 confirmed' : '0';
  const bg = override ? 'bg-white' : status === 'computed' ? 'bg-teal-50' : status === 'unresolved' ? 'bg-amber-50' : status === 'confirmedZero' || status === 'notApplicable' ? 'bg-slate-100' : 'bg-slate-50';
  const title = line ? [line.formula, line.explanation].filter(Boolean).join('\n') : '';
  return (
    <div className="flex flex-col gap-0.5" title={title}>
      <NumberInput
        ariaLabel={`${rule.label} adjustment for ${comp.attrs.address ?? 'comparable'}`}
        value={override?.amount ?? null}
        onChange={(v) => {
          if (v === null) setOverride(comp.id, rule.id, null);
          else setOverride(comp.id, rule.id, { amount: v, reason: override?.reason ?? '', source: override?.source ?? 'Entered in sales grid', attachmentRef: override?.attachmentRef ?? '' });
        }}
        placeholder={placeholder}
        className={`${compact} text-right tabular-nums ${bg} ${status === 'unresolved' && !override ? 'placeholder:text-amber-700' : status === 'computed' && !override ? 'placeholder:text-teal-800' : 'placeholder:text-slate-400'}`}
      />
      <div className="flex items-center justify-between text-[9px] uppercase tracking-wide text-slate-400">
        <span>{override ? 'manual' : status === 'computed' ? 'rule' : status === 'unresolved' ? 'needs input' : status === 'confirmedZero' ? 'zero' : status === 'notApplicable' ? 'n/a' : ''}</span>
        {status === 'unresolved' && !override && (
          <span className="flex gap-1">
            <button className="text-teal-700 hover:underline" onClick={() => setResolution(comp.id, rule.id, 'confirmedZero')} title="Confirm no adjustment">
              =0
            </button>
            <button className="text-slate-600 hover:underline" onClick={() => setResolution(comp.id, rule.id, 'notApplicable')} title="Mark not applicable">
              n/a
            </button>
          </span>
        )}
        {(status === 'confirmedZero' || status === 'notApplicable') && !override && (
          <button className="text-slate-500 hover:underline" onClick={() => setResolution(comp.id, rule.id, null)} title="Clear">
            clear
          </button>
        )}
      </div>
    </div>
  );
}

/* ---------------- comp header card ---------------- */
function CompHeader({ comp, index, scenario, calc }: { comp: Comparable; index: number; scenario: Scenario; calc: ScenarioCalculation }) {
  const { selectComp, toggleIncluded, setCompWeight, setCompPhoto, setUi } = useAppStore();
  const selectedId = useAppStore((s) => s.ui.selectedCompId);
  const c = calc.comps[comp.id];
  const a = comp.attrs;
  const dist = distanceToSubject(scenario.subject.attrs, a);
  const rounding = scenario.settings.rounding;
  const src = comp.photoData ?? a.photoUrl ?? null;
  return (
    <div className={`min-w-[17rem] ${selectedId === comp.id ? 'ring-2 ring-teal-500 rounded' : ''}`}>
      <div className="flex items-center justify-between rounded-t bg-navy-800 px-2 py-1 text-xs font-semibold text-white">
        <span>Sale #{index}</span>
        <div className="flex items-center gap-2 text-[10px] font-normal">
          <StatusBadge status={a.status} raw={a.statusRaw} />
          <button className="hover:underline" onClick={() => { selectComp(comp.id); setUi({ rightOpen: true, mobileTab: 'details' }); }}>
            Details
          </button>
        </div>
      </div>
      <div className="flex gap-2 p-2">
        <PhotoBox src={src} alt={a.address ?? 'Comparable photo'} onDrop={(d) => setCompPhoto(comp.id, d)} />
        <dl className="grid flex-1 grid-cols-[auto_1fr] gap-x-2 text-[11px] leading-4">
          <dt className="text-slate-500">Source:</dt>
          <dd className="truncate" title={comp.sourceFile}>
            {comp.extraction ? (comp.extraction.method === 'ai' ? 'AI doc' : 'PDF scan') : comp.synthetic ? 'Demo' : 'CSV'}
          </dd>
          <dt className="text-slate-500">Price:</dt>
          <dd className="font-semibold tabular-nums">{fmtMoney(a.salePrice)}</dd>
          <dt className="text-slate-500">Grs %:</dt>
          <dd className="tabular-nums">{c.grossPct !== null ? c.grossPct.toFixed(1) : '—'}</dd>
          <dt className="text-slate-500">Net %:</dt>
          <dd className="tabular-nums">{c.netPct !== null ? fmtPct(c.netPct, 1, true) : '—'}</dd>
          <dt className="text-slate-500">Ind:</dt>
          <dd className="font-semibold tabular-nums text-teal-700">{c.adjustedPrice !== null ? fmtMoney(c.adjustedPrice, { rounding }) : <span className="text-amber-700">unresolved</span>}</dd>
          <dt className="text-slate-500">Weight:</dt>
          <dd className="flex items-center gap-1">
            <NumberInput ariaLabel={`Weight for sale ${index}`} value={comp.weight} onChange={(v) => setCompWeight(comp.id, v ?? 0)} min={0} className={`${compact} w-14`} />
            <span className="text-slate-500">{c.normalizedWeight !== null ? fmtPct(c.normalizedWeight * 100, 1) : ''}</span>
          </dd>
        </dl>
      </div>
      <div className="border-t border-slate-200 bg-slate-50 px-2 py-1 text-[11px]">
        <div className="flex justify-between gap-2">
          <span className="text-slate-500">Neighborhood</span>
          <span className="truncate font-medium">{a.neighborhood ?? '—'}</span>
        </div>
      </div>
      <div className="px-2 py-1 text-xs">
        <div className="truncate font-semibold text-navy-900" title={a.address ?? undefined}>
          {a.address ?? 'Address unknown'}
        </div>
        <div className="truncate text-slate-600">{[a.city, a.zip].filter(Boolean).join(', ') || '—'}</div>
      </div>
      <div className="flex items-center justify-between border-t border-slate-200 px-2 py-1 text-[11px]">
        <span className="text-slate-600">{dist.miles !== null ? `${dist.miles.toFixed(2)} mi from subject` : 'Distance unknown'}</span>
        <label className="flex items-center gap-1">
          <input type="checkbox" checked={comp.included} onChange={() => toggleIncluded(comp.id)} aria-label={`Include sale ${index}`} /> Included
        </label>
      </div>
    </div>
  );
}

function SubjectHeader({ scenario, calc }: { scenario: Scenario; calc: ScenarioCalculation }) {
  const { setSubjectPhoto } = useAppStore();
  const s = scenario.subject.attrs;
  const src = scenario.subject.photoData ?? scenario.subject.photoUrls[0] ?? null;
  const r = calc.reconciliation;
  const rounding = scenario.settings.rounding;
  return (
    <div className="min-w-[10rem]">
      <div className="rounded-t bg-navy-900 px-2 py-1 text-xs font-semibold text-white">Subject</div>
      <div className="p-2">
        <PhotoBox src={src} alt="Subject photo" onDrop={setSubjectPhoto} />
        <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-2 text-[11px] leading-4">
          <dt className="text-slate-500">Source:</dt>
          <dd className="truncate">{scenario.subject.sourceFile ? 'Imported' : 'Typed'}</dd>
          <dt className="text-slate-500">Suggested:</dt>
          <dd className="font-semibold tabular-nums text-teal-700">{r.weightedIndication !== null ? fmtMoney(r.weightedIndication, { rounding }) : '—'}</dd>
          <dt className="text-slate-500">Final:</dt>
          <dd className="font-semibold tabular-nums">{fmtMoney(scenario.finalOpinion.value)}</dd>
        </dl>
      </div>
      <div className="border-t border-slate-200 bg-slate-50 px-2 py-1 text-[11px]">
        <div className="flex justify-between gap-2">
          <span className="text-slate-500">Neighborhood</span>
          <span className="truncate font-medium">{s.neighborhood ?? '—'}</span>
        </div>
      </div>
      <div className="px-2 py-1 text-xs">
        <div className="truncate font-semibold text-navy-900">{s.address ?? 'Subject address'}</div>
        <div className="truncate text-slate-600">{[s.city, s.zip].filter(Boolean).join(', ') || '—'}</div>
      </div>
      <div className="border-t border-slate-200 px-2 py-1 text-[11px] text-slate-600">Effective {formatDate(scenario.effectiveDate)}</div>
    </div>
  );
}

/* ---------------- valuation block ---------------- */
function ValuationBlock({ scenario, calc }: { scenario: Scenario; calc: ScenarioCalculation }) {
  const { setFinalOpinion, setReconciliationNotes } = useAppStore();
  const r = calc.reconciliation;
  const rounding = scenario.settings.rounding;
  return (
    <div className="card mt-3 grid grid-cols-1 gap-3 p-3 lg:grid-cols-[1.2fr_1fr_1fr]">
      <div className="rounded-md bg-navy-900 p-3 text-white">
        <div className="text-[10px] uppercase tracking-wide text-slate-300">Suggested value · weighted indication</div>
        <div className="text-3xl font-bold tabular-nums text-teal-300">{r.weightedIndication !== null ? fmtMoney(r.weightedIndication, { rounding }) : '—'}</div>
        <div className="mt-1 text-[11px] text-slate-300">
          Σ(adjusted sale price × weight) ÷ Σ weights over {r.contributingCount} included closed sale{r.contributingCount === 1 ? '' : 's'}
          {r.excludedForUnresolved > 0 && <span className="text-amber-300"> · {r.excludedForUnresolved} excluded until adjustments are entered</span>}
        </div>
        {r.message && <div className="mt-1 text-[11px] text-amber-200">{r.message}</div>}
        <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-0.5 text-xs">
          <dt className="text-slate-300">Adjusted range</dt>
          <dd className="tabular-nums">{r.adjustedRange ? `${fmtMoney(r.adjustedRange[0], { rounding })} – ${fmtMoney(r.adjustedRange[1], { rounding })}` : '—'}</dd>
          <dt className="text-slate-300">Unadjusted range</dt>
          <dd className="tabular-nums">{r.unadjustedRange ? `${fmtMoney(r.unadjustedRange[0], { rounding })} – ${fmtMoney(r.unadjustedRange[1], { rounding })}` : '—'}</dd>
          <dt className="text-slate-300">Mean / median</dt>
          <dd className="tabular-nums">{fmtMoney(r.adjustedMean, { rounding })} / {fmtMoney(r.adjustedMedian, { rounding })}</dd>
        </dl>
        <div className="mt-1 text-[10px] text-slate-400">Observed range of adjusted prices, not a statistical confidence interval. Rounded to ${rounding}.</div>
      </div>
      <div>
        <label className="label" htmlFor="grid-final-value">
          Final reconciled opinion of value ($)
        </label>
        <NumberInput id="grid-final-value" value={scenario.finalOpinion.value} onChange={(v) => setFinalOpinion({ value: v })} suffix="$" />
        <label className="label mt-2" htmlFor="grid-final-rationale">
          Rationale
        </label>
        <TextArea id="grid-final-rationale" value={scenario.finalOpinion.rationale} onChange={(v) => setFinalOpinion({ rationale: v })} rows={4} />
      </div>
      <div>
        <label className="label" htmlFor="grid-recon-notes">
          Reconciliation notes
        </label>
        <TextArea id="grid-recon-notes" value={scenario.reconciliationNotes} onChange={setReconciliationNotes} rows={7} />
      </div>
    </div>
  );
}

/* ---------------- the grid ---------------- */
export function ComparisonGrid({ scenario, calc }: { scenario: Scenario; calc: ScenarioCalculation }) {
  const { selectComp, setSubjectAttrs, setCompAttrs, setCompGridNote, setUi } = useAppStore();
  const selectedId = useAppStore((s) => s.ui.selectedCompId);
  const comps = useMemo(() => scenario.comparables.filter((c) => c.included), [scenario]);
  const rules = orderedRules(scenario.rules);
  const ruleByCategory = useMemo(() => new Map(rules.map((r) => [r.category, r])), [rules]);
  const rows = useMemo(() => [...FORM_ROWS, ...customRows(scenario.rules)], [scenario.rules]);
  const s = scenario.subject.attrs;
  const rounding = scenario.settings.rounding;
  const lineFor = (comp: Comparable, rule: AdjustmentRule | undefined) => (rule ? calc.comps[comp.id].lines.find((l) => l.ruleId === rule.id) : undefined);

  if (!comps.length) {
    return (
      <EmptyState title="No comparable sales on the grid yet">
        <p>Tick <strong>Include</strong> on sales in the Cards view, or import more sales from CSV or PDF. Included sales appear here as columns next to the subject.</p>
        <div className="flex gap-2">
          <button className="btn-primary" onClick={() => setUi({ view: 'cards' })}>
            Choose comparables
          </button>
          <button className="btn-secondary" onClick={() => setUi({ importOpen: true })}>
            Import CSV
          </button>
        </div>
      </EmptyState>
    );
  }

  const th = (children: ReactNode, cls = '') => <th className={`border border-slate-300 bg-slate-100 px-2 py-1 text-left text-[11px] font-semibold text-slate-700 ${cls}`}>{children}</th>;
  const labelCell = (text: string, extra = '') => (
    <th scope="row" className={`sticky left-0 z-10 border border-slate-300 bg-slate-50 px-2 py-1 text-left text-[11px] font-medium text-slate-700 ${extra}`} style={{ width: LABEL_W, minWidth: LABEL_W }}>
      {text}
    </th>
  );
  const subjectCell = (children: ReactNode, extra = '') => (
    <td className={`sticky z-10 border border-slate-300 bg-navy-50 px-1 py-0.5 text-xs ${extra}`} style={{ left: LABEL_W, width: SUBJECT_W, minWidth: SUBJECT_W }}>
      {children}
    </td>
  );
  const compCls = (comp: Comparable) => (selectedId === comp.id ? 'bg-teal-50/60' : 'bg-white');

  const signBox = (net: number | null) => (
    <span className="mr-2 inline-flex gap-1 font-mono text-[11px]">
      <span className={net !== null && net > 0 ? 'font-bold text-teal-700' : 'text-slate-300'}>☒+</span>
      <span className={net !== null && net < 0 ? 'font-bold text-red-700' : 'text-slate-300'}>☒−</span>
    </span>
  );

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-600">
        <span><span className="inline-block h-3 w-3 rounded-sm bg-teal-50 align-middle ring-1 ring-teal-200" /> computed by a global rule</span>
        <span><span className="inline-block h-3 w-3 rounded-sm bg-white align-middle ring-1 ring-slate-300" /> typed adjustment (overrides the rule)</span>
        <span><span className="inline-block h-3 w-3 rounded-sm bg-amber-50 align-middle ring-1 ring-amber-300" /> needs input before the sale counts</span>
        <span className="ml-auto">Type a signed dollar amount in any adjustment cell; clear it to return to the rule. Positive adds to an inferior sale, negative subtracts from a superior one.</span>
      </div>
      <div className="card overflow-hidden">
        <div className="max-h-[calc(100vh-15rem)] overflow-auto">
          <table className="border-separate border-spacing-0">
            <thead>
              <tr>
                <th className="sticky left-0 top-0 z-30 border border-slate-300 bg-white p-0 align-top" style={{ width: LABEL_W, minWidth: LABEL_W }}>
                  <div className="p-2 text-[11px] text-slate-500">
                    <div className="font-semibold text-navy-900">Sales comparison grid</div>
                    {comps.length} sale{comps.length === 1 ? '' : 's'} on grid · {scenario.comparables.length - comps.length} not included
                  </div>
                </th>
                <th className="sticky top-0 z-30 border border-slate-300 bg-white p-0 align-top" style={{ left: LABEL_W, width: SUBJECT_W, minWidth: SUBJECT_W }}>
                  <SubjectHeader scenario={scenario} calc={calc} />
                </th>
                {comps.map((c, i) => (
                  <th key={c.id} colSpan={2} className="sticky top-0 z-20 border border-slate-300 bg-white p-0 align-top" onClick={() => selectComp(c.id)}>
                    <CompHeader comp={c} index={i + 1} scenario={scenario} calc={calc} />
                  </th>
                ))}
              </tr>
              <tr>
                {th('Item', 'sticky left-0 z-20')}
                <th className="sticky z-20 border border-slate-300 bg-slate-100 px-2 py-1 text-left text-[11px] font-semibold text-slate-700" style={{ left: LABEL_W }}>
                  Description
                </th>
                {comps.map((c) => (
                  <Fragment key={c.id}>
                    <th className="min-w-[10rem] border border-slate-300 bg-slate-100 px-2 py-1 text-left text-[11px] font-semibold text-slate-700">
                      Description
                    </th>
                    <th className="min-w-[7.5rem] border border-slate-300 bg-slate-100 px-2 py-1 text-right text-[11px] font-semibold text-slate-700">
                      +(−) $ Adjustment
                    </th>
                  </Fragment>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                if (row.section) {
                  return (
                    <tr key={row.key}>
                      <th scope="row" colSpan={2 + comps.length * 2} className="sticky left-0 border border-slate-300 bg-navy-100 px-2 py-0.5 text-left text-[11px] font-semibold uppercase tracking-wide text-navy-900">
                        {row.label}
                      </th>
                    </tr>
                  );
                }
                const rule = row.category ? ruleByCategory.get(row.category) : undefined;
                return (
                  <tr key={row.key}>
                    {labelCell(row.label)}
                    {subjectCell(
                      row.subjectBlank ? (
                        <span className="text-slate-300">—</span>
                      ) : row.computed === 'dataSource' ? (
                        <span className="text-[11px] text-slate-600">{s.mlsNumber ? `MLS# ${s.mlsNumber}` : scenario.subject.sourceFile ?? 'Typed'}</span>
                      ) : row.key === 'marketConditions' ? (
                        <span className="text-[11px] text-slate-600">Effective {formatDate(scenario.effectiveDate)}</span>
                      ) : (
                        <DescCell row={row} attrs={s} onChange={setSubjectAttrs} id={`g-subject-${row.key}`} />
                      ),
                    )}
                    {comps.map((c) => {
                      const cc = calc.comps[c.id];
                      const line = lineFor(c, rule);
                      let desc: ReactNode;
                      if (row.computed === 'pricePerGla') desc = <span className="tabular-nums">{cc.pricePerSqft !== null ? `${fmtMoney(cc.pricePerSqft, { decimals: 2 })} / SF` : '—'}</span>;
                      else if (row.computed === 'dataSource') desc = <span className="text-[11px] text-slate-600" title={c.sourceFile}>{[c.attrs.mlsNumber ? `MLS# ${c.attrs.mlsNumber}` : null, c.attrs.daysOnMarket !== null ? `DOM ${fmtNumber(c.attrs.daysOnMarket)}` : null].filter(Boolean).join('; ') || c.sourceFile}</span>;
                      else if (row.key === 'marketConditions' && rule?.market?.dateBasis === 'contractDate') desc = <DescCell row={{ ...row, field: 'contractDate', label: 'Contract date' }} attrs={c.attrs} onChange={(p) => setCompAttrs(c.id, p)} id={`g-${c.id}-${row.key}`} />;
                      else desc = <DescCell row={row} attrs={c.attrs} onChange={(p) => setCompAttrs(c.id, p)} id={`g-${c.id}-${row.key}`} />;
                      return (
                        <Fragment key={c.id}>
                          <td className={`border border-slate-300 px-1 py-0.5 text-xs ${compCls(c)}`} onClick={() => selectedId !== c.id && selectComp(c.id)}>
                            {desc}
                          </td>
                          <td className={`border border-slate-300 px-1 py-0.5 text-xs ${compCls(c)}`} onClick={() => selectedId !== c.id && selectComp(c.id)}>
                            {rule ? <AdjCell comp={c} rule={rule} line={line} rounding={rounding} /> : <span className="text-slate-300">—</span>}
                          </td>
                        </Fragment>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
            <tfoot>
              <tr>
                {labelCell('Net adjustment (total)', 'bg-slate-100')}
                {subjectCell('', 'bg-slate-100')}
                {comps.map((c) => {
                  const cc = calc.comps[c.id];
                  return (
                    <td key={c.id} colSpan={2} className="border border-slate-300 bg-slate-100 px-2 py-1 text-xs">
                      <div className="flex items-center justify-between">
                        {signBox(cc.unresolved.length ? null : cc.netAdjustment)}
                        {cc.unresolved.length ? <Badge tone="amber">{cc.unresolved.length} needs input</Badge> : <span className="font-semibold tabular-nums">{fmtMoney(cc.netAdjustment, { signed: true, rounding })}</span>}
                      </div>
                    </td>
                  );
                })}
              </tr>
              <tr>
                {labelCell('Net adj. % / Gross adj. %')}
                {subjectCell('')}
                {comps.map((c) => {
                  const cc = calc.comps[c.id];
                  return (
                    <td key={c.id} colSpan={2} className="border border-slate-300 px-2 py-1 text-xs tabular-nums">
                      Net {fmtPct(cc.netPct, 1, true)} · Gross {fmtPct(cc.grossPct, 1)}
                    </td>
                  );
                })}
              </tr>
              <tr>
                {labelCell('Adjusted sale price of comparable', 'bg-teal-50')}
                {subjectCell('', 'bg-teal-50')}
                {comps.map((c) => {
                  const cc = calc.comps[c.id];
                  return (
                    <td key={c.id} colSpan={2} className="border border-slate-300 bg-teal-50 px-2 py-1 text-right text-sm font-bold tabular-nums text-navy-900">
                      {cc.adjustedPrice !== null ? fmtMoney(cc.adjustedPrice, { rounding }) : <span className="text-xs font-medium text-amber-700">Enter the missing adjustments</span>}
                      {c.attrs.status !== 'closed' && <div className="text-[10px] font-normal text-slate-500">context only (not closed)</div>}
                    </td>
                  );
                })}
              </tr>
              <tr>
                {labelCell('Notes')}
                {subjectCell(<span className="text-[11px] text-slate-600">{scenario.subject.notes || ''}</span>)}
                {comps.map((c) => (
                  <td key={c.id} colSpan={2} className="border border-slate-300 px-1 py-0.5">
                    <textarea aria-label={`Notes for ${c.attrs.address ?? 'comparable'}`} className={`input ${compact}`} rows={2} value={c.gridNote} onChange={(e) => setCompGridNote(c.id, e.target.value)} placeholder={c.selectionNote || 'Note'} />
                  </td>
                ))}
              </tr>
              <tr>
                {labelCell('Data source / effective date')}
                {subjectCell(<span className="text-[11px] text-slate-600">{formatDate(scenario.effectiveDate)}</span>)}
                {comps.map((c) => (
                  <td key={c.id} colSpan={2} className="border border-slate-300 px-2 py-1 text-[11px] text-slate-600">
                    <div className="truncate" title={c.sourceFile}>
                      {c.sourceFile}
                    </div>
                    {formatDate(scenario.effectiveDate)}
                  </td>
                ))}
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
      <ValuationBlock scenario={scenario} calc={calc} />
    </div>
  );
}
