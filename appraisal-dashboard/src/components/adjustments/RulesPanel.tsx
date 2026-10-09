import { useMemo, useState } from 'react';
import { useAppStore } from '../../store/useAppStore';
import type { AdjustmentRule, PropertyAttributes, RuleKind, Scenario, Warning } from '../../types';
import { ATTR_LABELS, BOOLEAN_ATTRS, QUANTITATIVE_ATTRS, RATING_ATTRS } from '../../engine/fields';
import { orderedRules } from '../../engine/calc';
import { Badge, Field, NumberInput, Select, TextInput, Toggle } from '../ui/primitives';

const KIND_LABEL: Record<RuleKind, string> = {
  perUnit: 'Dollar per unit',
  flat: 'Flat dollar',
  percent: 'Percent of sale price',
  categorical: 'Dollar by rating',
  market: 'Market conditions',
  concessions: 'Concessions',
  manual: 'Manual only',
};

function ratingValuesIn(scenario: Scenario, field: keyof PropertyAttributes | null): string[] {
  if (!field) return [];
  const set = new Set<string>();
  const sv = scenario.subject.attrs[field];
  if (typeof sv === 'string') set.add(sv);
  for (const c of scenario.comparables) {
    const v = c.attrs[field];
    if (typeof v === 'string') set.add(v);
  }
  return [...set].sort();
}

function RuleEditor({ rule, scenario }: { rule: AdjustmentRule; scenario: Scenario }) {
  const updateRule = useAppStore((s) => s.updateRule);
  const removeRule = useAppStore((s) => s.removeRule);
  const [open, setOpen] = useState(false);
  const patch = (p: Partial<AdjustmentRule>) => updateRule(rule.id, p);
  const ratings = useMemo(() => ratingValuesIn(scenario, rule.field), [scenario, rule.field]);
  const [newRating, setNewRating] = useState('');
  const fieldOptions = useMemo(() => {
    const pool: (keyof PropertyAttributes)[] =
      rule.kind === 'perUnit' ? QUANTITATIVE_ATTRS : rule.kind === 'flat' ? BOOLEAN_ATTRS : rule.kind === 'percent' ? [...BOOLEAN_ATTRS, ...RATING_ATTRS] : rule.kind === 'categorical' ? RATING_ATTRS : [];
    return pool.map((k) => ({ value: k, label: ATTR_LABELS[k] }));
  }, [rule.kind]);
  const idp = `rule-${rule.id}`;
  const summary = (() => {
    if (!rule.active) return 'Inactive';
    switch (rule.kind) {
      case 'perUnit':
        return rule.rate === null ? 'Rate not set' : `$${rule.rate.toLocaleString()} per unit`;
      case 'flat':
        return rule.rate === null ? 'Amount not set' : `$${rule.rate.toLocaleString()} flat`;
      case 'percent':
        return rule.rate === null && !Object.keys(rule.categoricalValues).length ? 'Percent not set' : rule.rate !== null ? `${rule.rate}%` : 'By rating (%)';
      case 'categorical':
        return Object.keys(rule.categoricalValues).length ? `${Object.keys(rule.categoricalValues).length} ratings valued` : 'No rating values';
      case 'market':
        return rule.market?.mode === 'manual' ? 'Manual per comp' : rule.market?.monthlyRatePct === null || rule.market?.monthlyRatePct === undefined ? 'Monthly rate not set' : `${rule.market.monthlyRatePct}%/mo from ${rule.market.dateBasis === 'contractDate' ? 'contract' : 'sale'} date`;
      case 'concessions':
        return rule.concessionFactor === null || rule.concessionFactor === undefined ? 'Factor not set' : `${rule.concessionFactor} × reported`;
      default:
        return 'Manual overrides only';
    }
  })();

  return (
    <div className={`rounded-md border ${rule.active ? 'border-teal-200 bg-teal-50/40' : 'border-slate-200 bg-white'}`}>
      <div className="flex items-center gap-2 px-2 py-1.5">
        <Toggle id={`${idp}-active`} checked={rule.active} onChange={(v) => patch({ active: v })} label="" />
        <button className="min-w-0 flex-1 text-left" onClick={() => setOpen(!open)} aria-expanded={open} aria-controls={`${idp}-body`}>
          <div className="truncate text-sm font-medium text-slate-800">{rule.label}</div>
          <div className="truncate text-[11px] text-slate-500">
            {KIND_LABEL[rule.kind]} · {summary}
          </div>
        </button>
        <span className={`text-slate-400 transition ${open ? 'rotate-90' : ''}`} aria-hidden>
          ▸
        </span>
      </div>
      {open && (
        <div id={`${idp}-body`} className="space-y-2 border-t border-slate-200 px-2 py-2">
          {rule.custom && (
            <div className="grid grid-cols-2 gap-2">
              <Field label="Label" htmlFor={`${idp}-label`}>
                <TextInput id={`${idp}-label`} value={rule.label} onChange={(v) => patch({ label: v ?? '' })} />
              </Field>
              <Field label="Method" htmlFor={`${idp}-kind`}>
                <Select<RuleKind>
                  id={`${idp}-kind`}
                  value={rule.kind}
                  onChange={(v) => v && patch({ kind: v, field: null, rate: null, categoricalValues: {} })}
                  options={(['perUnit', 'flat', 'percent', 'categorical', 'manual'] as RuleKind[]).map((k) => ({ value: k, label: KIND_LABEL[k] }))}
                />
              </Field>
            </div>
          )}

          {(rule.kind === 'perUnit' || rule.kind === 'flat' || rule.kind === 'percent' || rule.kind === 'categorical') && (
            <Field label="Compared attribute" htmlFor={`${idp}-field`}>
              <Select<keyof PropertyAttributes> id={`${idp}-field`} value={rule.field} onChange={(v) => patch({ field: v })} options={fieldOptions} placeholder="Choose attribute" />
            </Field>
          )}

          {rule.kind === 'perUnit' && (
            <Field label="Rate ($ per unit)" htmlFor={`${idp}-rate`} hint="Adjustment = (Subject − Comparable) × Rate. Positive result adds to an inferior comparable; negative subtracts from a superior one.">
              <NumberInput id={`${idp}-rate`} value={rule.rate} onChange={(v) => patch({ rate: v })} suffix="$" placeholder="Blank = not set" />
            </Field>
          )}
          {rule.kind === 'flat' && (
            <Field label="Flat amount ($)" htmlFor={`${idp}-rate`} hint="Added when the subject has the feature and the comparable does not; subtracted in the opposite case.">
              <NumberInput id={`${idp}-rate`} value={rule.rate} onChange={(v) => patch({ rate: v })} suffix="$" placeholder="Blank = not set" />
            </Field>
          )}
          {rule.kind === 'percent' && rule.field && BOOLEAN_ATTRS.includes(rule.field) && (
            <Field label="Percent of comparable sale price" htmlFor={`${idp}-rate`} hint="Signed by which property has the feature.">
              <NumberInput id={`${idp}-rate`} value={rule.rate} onChange={(v) => patch({ rate: v })} suffix="%" placeholder="Blank = not set" />
            </Field>
          )}
          {((rule.kind === 'categorical') || (rule.kind === 'percent' && rule.field && RATING_ATTRS.includes(rule.field))) && (
            <div>
              <div className="label">{rule.kind === 'percent' ? 'Percent of sale price by rating' : 'Dollar value by rating'}</div>
              <p className="mb-1 text-[11px] text-slate-500">Adjustment = value(subject rating) − value(comparable rating). Intervals need not be equal.</p>
              <div className="space-y-1">
                {[...new Set([...ratings, ...Object.keys(rule.categoricalValues)])].sort().map((r) => (
                  <div key={r} className="flex items-center gap-2">
                    <span className="w-20 truncate text-xs font-medium text-slate-700" title={r}>
                      {r}
                    </span>
                    <NumberInput ariaLabel={`Value for ${r}`} value={rule.categoricalValues[r] ?? null} onChange={(v) => {
                      const next = { ...rule.categoricalValues };
                      if (v === null) delete next[r];
                      else next[r] = v;
                      patch({ categoricalValues: next });
                    }} suffix={rule.kind === 'percent' ? '%' : '$'} className="flex-1" />
                  </div>
                ))}
                <div className="flex items-center gap-2">
                  <TextInput ariaLabel="New rating label" value={newRating} onChange={(v) => setNewRating(v ?? '')} placeholder="Add rating label" className="flex-1" />
                  <button className="btn-secondary btn-sm" onClick={() => {
                    if (!newRating.trim()) return;
                    patch({ categoricalValues: { ...rule.categoricalValues, [newRating.trim()]: 0 } });
                    setNewRating('');
                  }}>
                    Add
                  </button>
                </div>
              </div>
            </div>
          )}

          {rule.kind === 'market' && rule.market && (
            <div className="space-y-2">
              <Field label="Method" htmlFor={`${idp}-mmode`}>
                <Select<'manual' | 'monthlyRate'> id={`${idp}-mmode`} value={rule.market.mode} onChange={(v) => v && patch({ market: { ...rule.market!, mode: v } })} options={[{ value: 'monthlyRate', label: 'Monthly percentage rate' }, { value: 'manual', label: 'Manual per comparable' }]} />
              </Field>
              {rule.market.mode === 'monthlyRate' && (
                <>
                  <Field label="Monthly rate (% per month)" htmlFor={`${idp}-mrate`} hint="Adjustment = Original sale price × [(1 + rate)^elapsed months − 1]. Elapsed months = date-only days ÷ 30.4375. Enter a rate you can support; it is not derived from the imported data.">
                    <NumberInput id={`${idp}-mrate`} value={rule.market.monthlyRatePct} onChange={(v) => patch({ market: { ...rule.market!, monthlyRatePct: v } })} suffix="%" step="any" placeholder="Blank = not set" />
                  </Field>
                  <Field label="Start date basis" htmlFor={`${idp}-mbasis`}>
                    <Select<'saleDate' | 'contractDate'> id={`${idp}-mbasis`} value={rule.market.dateBasis} onChange={(v) => v && patch({ market: { ...rule.market!, dateBasis: v } })} options={[{ value: 'saleDate', label: 'Sale (closing) date' }, { value: 'contractDate', label: 'Contract date' }]} />
                  </Field>
                </>
              )}
              {rule.market.mode === 'manual' && <p className="text-[11px] text-slate-500">Each included comparable needs a manual override, confirmed $0, or N/A in its detail panel.</p>}
            </div>
          )}

          {rule.kind === 'concessions' && (
            <Field label="Factor applied to reported concessions" htmlFor={`${idp}-cf`} hint="Adjustment = −(reported concession × factor). 1 = dollar for dollar. Reported amounts are never deducted automatically; set the factor you can support or use per-comparable overrides.">
              <NumberInput id={`${idp}-cf`} value={rule.concessionFactor ?? null} onChange={(v) => patch({ concessionFactor: v })} step="any" placeholder="Blank = not set" />
            </Field>
          )}

          {rule.kind === 'manual' && <p className="text-[11px] text-slate-500">No formula. Enter an override, confirm $0, or mark N/A for each included comparable.</p>}

          <div className="grid grid-cols-1 gap-2 border-t border-slate-200 pt-2">
            <Field label="Reason / support" htmlFor={`${idp}-reason`}>
              <TextInput id={`${idp}-reason`} value={rule.reason} onChange={(v) => patch({ reason: v ?? '' })} placeholder="e.g. paired-sales analysis" />
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Method / source note" htmlFor={`${idp}-source`}>
                <TextInput id={`${idp}-source`} value={rule.source} onChange={(v) => patch({ source: v ?? '' })} />
              </Field>
              <Field label="Attachment reference" htmlFor={`${idp}-attach`}>
                <TextInput id={`${idp}-attach`} value={rule.attachmentRef} onChange={(v) => patch({ attachmentRef: v ?? '' })} placeholder="file name / workfile ref" />
              </Field>
            </div>
          </div>
          {rule.custom && (
            <div className="flex justify-end">
              <button className="btn-danger btn-sm" onClick={() => removeRule(rule.id)}>
                Remove category
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function RulesPanel({ scenario, warnings }: { scenario: Scenario; warnings: Warning[] }) {
  const addCustomRule = useAppStore((s) => s.addCustomRule);
  const [label, setLabel] = useState('');
  const [kind, setKind] = useState<RuleKind>('manual');
  const rules = orderedRules(scenario.rules);
  const activeCount = rules.filter((r) => r.active).length;
  const overlap = warnings.filter((w) => !['nonClosed', 'noEffectiveDate'].includes(w.code));
  return (
    <div className="space-y-2">
      <p className="text-[11px] text-slate-500">
        All rates start blank and inactive. Activate a category and enter a rate you can support. <Badge tone="teal">{activeCount} active</Badge>
      </p>
      {overlap.length > 0 && (
        <div className="rounded-md border border-amber-200 bg-amber-50 p-2 text-[11px] text-amber-800" role="status">
          <div className="font-semibold">Possible overlapping adjustments</div>
          <ul className="list-disc pl-4">
            {overlap.map((w) => (
              <li key={w.code + w.message}>{w.message}</li>
            ))}
          </ul>
        </div>
      )}
      {rules.map((r) => (
        <RuleEditor key={r.id} rule={r} scenario={scenario} />
      ))}
      <div className="rounded-md border border-dashed border-slate-300 p-2">
        <div className="label">Add custom category</div>
        <div className="flex gap-1">
          <TextInput ariaLabel="Custom category label" value={label} onChange={(v) => setLabel(v ?? '')} placeholder="e.g. Outbuilding" className="flex-1" />
          <Select<RuleKind> ariaLabel="Custom category method" value={kind} onChange={(v) => v && setKind(v)} options={(['manual', 'perUnit', 'flat', 'percent', 'categorical'] as RuleKind[]).map((k) => ({ value: k, label: KIND_LABEL[k] }))} className="w-36" />
          <button className="btn-secondary btn-sm" disabled={!label.trim()} onClick={() => { addCustomRule(label.trim(), kind, null); setLabel(''); }}>
            Add
          </button>
        </div>
      </div>
    </div>
  );
}
