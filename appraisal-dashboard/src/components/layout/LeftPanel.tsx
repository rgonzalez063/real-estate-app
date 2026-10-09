import { useAppStore } from '../../store/useAppStore';
import type { Scenario, Warning } from '../../types';
import { Field, NumberInput, Section, Select } from '../ui/primitives';
import { SubjectForm } from '../subject/SubjectForm';
import { RulesPanel } from '../adjustments/RulesPanel';

export function LeftPanel({ scenario, warnings }: { scenario: Scenario; warnings: Warning[] }) {
  const setSettings = useAppStore((s) => s.setSettings);
  const setFlags = useAppStore((s) => s.setFlags);
  const f = scenario.settings.flags;
  return (
    <div className="h-full overflow-y-auto">
      <Section title="Subject property" id="subject-section">
        <SubjectForm subject={scenario.subject} effectiveDate={scenario.effectiveDate} />
      </Section>
      <Section title="Adjustment assumptions" id="rules-section">
        <RulesPanel scenario={scenario} warnings={warnings} />
      </Section>
      <Section title="Display & review flags" defaultOpen={false}>
        <div className="space-y-2">
          <Field label="Rounding for display and exports" htmlFor="rounding">
            <Select<string> id="rounding" value={String(scenario.settings.rounding)} onChange={(v) => setSettings({ rounding: Number(v ?? 1) })} options={[{ value: '1', label: 'Nearest $1' }, { value: '10', label: 'Nearest $10' }, { value: '100', label: 'Nearest $100' }, { value: '1000', label: 'Nearest $1,000' }]} />
          </Field>
          <p className="text-[11px] text-slate-500">User-defined review flags. Leave blank to disable a flag.</p>
          <div className="grid grid-cols-2 gap-2">
            <Field label="Net adj. over" htmlFor="flag-net">
              <NumberInput id="flag-net" value={f.netPct} onChange={(v) => setFlags({ netPct: v })} suffix="%" />
            </Field>
            <Field label="Gross adj. over" htmlFor="flag-gross">
              <NumberInput id="flag-gross" value={f.grossPct} onChange={(v) => setFlags({ grossPct: v })} suffix="%" />
            </Field>
            <Field label="Single line over" htmlFor="flag-line">
              <NumberInput id="flag-line" value={f.singleLinePct} onChange={(v) => setFlags({ singleLinePct: v })} suffix="%" />
            </Field>
            <Field label="Outlier from median" htmlFor="flag-outlier">
              <NumberInput id="flag-outlier" value={f.outlierPct} onChange={(v) => setFlags({ outlierPct: v })} suffix="%" />
            </Field>
          </div>
        </div>
      </Section>
    </div>
  );
}
