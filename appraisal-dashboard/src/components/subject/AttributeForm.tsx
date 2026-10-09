import type { LotUnits, PropertyAttributes, StatusCategory } from '../../types';
import { Field, NumberInput, Select, TextArea, TextInput, TriState } from '../ui/primitives';
import { SQFT_PER_ACRE } from '../../engine/parse';

type Patch = Partial<PropertyAttributes>;

/**
 * Attribute editor shared by the subject form and the comparable detail editor.
 * Blank inputs are stored as null (unknown), never as zero.
 */
export function AttributeForm({ attrs, onChange, mode, idPrefix }: { attrs: PropertyAttributes; onChange: (patch: Patch) => void; mode: 'subject' | 'comp'; idPrefix: string }) {
  const id = (k: string) => `${idPrefix}-${k}`;
  const setLot = (raw: number | null, units: LotUnits | null) => {
    const sqft = raw === null || units === null ? null : units === 'acres' ? raw * SQFT_PER_ACRE : raw;
    onChange({ lotSizeRaw: raw, lotSizeUnits: units, lotSizeSqft: sqft });
  };
  const num = (k: keyof PropertyAttributes, label: string, suffix?: string, hint?: string) => (
    <Field label={label} htmlFor={id(k)} hint={hint}>
      <NumberInput id={id(k)} value={attrs[k] as number | null} onChange={(v) => onChange({ [k]: v })} suffix={suffix} />
    </Field>
  );
  const text = (k: keyof PropertyAttributes, label: string) => (
    <Field label={label} htmlFor={id(k)}>
      <TextInput id={id(k)} value={attrs[k] as string | null} onChange={(v) => onChange({ [k]: v })} />
    </Field>
  );
  const bool = (k: keyof PropertyAttributes, label: string) => (
    <Field label={label} htmlFor={id(k)}>
      <TriState id={id(k)} value={attrs[k] as boolean | null} onChange={(v) => onChange({ [k]: v })} />
    </Field>
  );
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-2">
        <div className="col-span-2">{text('address', 'Address')}</div>
        {text('city', 'City')}
        {text('zip', 'ZIP')}
        {text('neighborhood', 'Neighborhood')}
        {text('mlsNumber', 'MLS number')}
      </div>

      {mode === 'comp' && (
        <div>
          <div className="section-title mb-2">Transaction</div>
          <div className="grid grid-cols-2 gap-2">
            <Field label="Status" htmlFor={id('status')} hint={attrs.statusRaw ? `Reported: ${attrs.statusRaw}` : undefined}>
              <Select<StatusCategory>
                id={id('status')}
                value={attrs.status}
                onChange={(v) => onChange({ status: v })}
                options={[
                  { value: 'closed', label: 'Closed sale' },
                  { value: 'pending', label: 'Pending / under contract' },
                  { value: 'active', label: 'Active listing' },
                  { value: 'other', label: 'Other' },
                ]}
                placeholder="Unknown"
              />
            </Field>
            {num('daysOnMarket', 'Days on market')}
            {num('salePrice', 'Sale price', '$')}
            {num('listPrice', 'List price', '$')}
            <Field label="Sale date" htmlFor={id('saleDate')}>
              <TextInput id={id('saleDate')} type="date" value={attrs.saleDate} onChange={(v) => onChange({ saleDate: v })} />
            </Field>
            <Field label="Contract date" htmlFor={id('contractDate')}>
              <TextInput id={id('contractDate')} type="date" value={attrs.contractDate} onChange={(v) => onChange({ contractDate: v })} />
            </Field>
            <div className="col-span-2">{num('concessions', 'Reported concessions (source data)', '$', 'Kept separate from the appraiser’s concession adjustment.')}</div>
          </div>
        </div>
      )}

      <div>
        <div className="section-title mb-2">Area by category</div>
        <div className="grid grid-cols-2 gap-2">
          {num('areaAboveGrade', 'Above-grade finished', 'SF')}
          {num('areaBelowGradeFinished', 'Below-grade finished', 'SF')}
          {num('areaBelowGradeUnfinished', 'Below-grade unfinished', 'SF')}
          {num('areaOther', 'Other area', 'SF')}
        </div>
      </div>

      <div>
        <div className="section-title mb-2">Site</div>
        <div className="grid grid-cols-2 gap-2">
          <Field label="Site area" htmlFor={id('lot')}>
            <NumberInput id={id('lot')} value={attrs.lotSizeRaw} onChange={(v) => setLot(v, attrs.lotSizeUnits)} />
          </Field>
          <Field label="Units" htmlFor={id('lotUnits')}>
            <Select<LotUnits> id={id('lotUnits')} value={attrs.lotSizeUnits} onChange={(v) => setLot(attrs.lotSizeRaw, v)} options={[{ value: 'sqft', label: 'Square feet' }, { value: 'acres', label: 'Acres' }]} placeholder="Choose units" />
          </Field>
          {attrs.lotSizeRaw !== null && attrs.lotSizeUnits === null && <p className="col-span-2 text-[11px] text-amber-700">Choose units so the site area can be compared.</p>}
        </div>
      </div>

      <div>
        <div className="section-title mb-2">Rooms, age, parking</div>
        <div className="grid grid-cols-3 gap-2">
          {num('bedrooms', 'Bedrooms')}
          {num('fullBaths', 'Full baths')}
          {num('halfBaths', 'Half baths')}
          {num('yearBuilt', 'Year built')}
          {num('age', 'Age (yrs)')}
          {num('garageSpaces', 'Garage')}
          {num('carportSpaces', 'Carport')}
        </div>
      </div>

      <div>
        <div className="section-title mb-2">Quality, condition, location</div>
        <div className="grid grid-cols-2 gap-2">
          {text('quality', 'Quality (e.g. Q3)')}
          {text('condition', 'Condition (e.g. C3)')}
          {text('location', 'Location')}
          {text('renovation', 'Renovation / updates')}
        </div>
      </div>

      <div>
        <div className="section-title mb-2">Amenities</div>
        <div className="grid grid-cols-3 gap-2">
          {bool('pool', 'Pool')}
          {bool('waterfront', 'Waterfront')}
          {bool('waterView', 'Water view')}
        </div>
        <div className="mt-2">{text('amenities', 'Other amenities')}</div>
      </div>

      <div>
        <div className="section-title mb-2">Coordinates (optional)</div>
        <div className="grid grid-cols-2 gap-2">
          {num('latitude', 'Latitude')}
          {num('longitude', 'Longitude')}
          {mode === 'comp' && num('distance', 'Distance to subject', 'mi')}
        </div>
      </div>

      <div>
        <Field label="Remarks" htmlFor={id('remarks')}>
          <TextArea id={id('remarks')} value={attrs.remarks ?? ''} onChange={(v) => onChange({ remarks: v || null })} rows={2} />
        </Field>
      </div>
    </div>
  );
}
