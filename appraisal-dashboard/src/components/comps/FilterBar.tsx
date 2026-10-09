import type { Comparable, PropertyAttributes, StatusCategory } from '../../types';
import { NumberInput, Select, TextInput } from '../ui/primitives';
import { defaultFilters, distanceAvailable, type CompFilters, type SortKey } from './filters';

const STATUS_OPTS: { value: StatusCategory; label: string }[] = [
  { value: 'closed', label: 'Closed' },
  { value: 'pending', label: 'Pending' },
  { value: 'active', label: 'Active' },
  { value: 'other', label: 'Other' },
];

function uniq(comps: Comparable[], key: keyof PropertyAttributes): string[] {
  return [...new Set(comps.map((c) => c.attrs[key]).filter((v): v is string => typeof v === 'string' && v !== ''))].sort();
}

export function FilterBar({ filters, onChange, comps, subject, sort, onSort, open, onToggle }: { filters: CompFilters; onChange: (f: CompFilters) => void; comps: Comparable[]; subject: PropertyAttributes; sort: { key: SortKey; dir: 'asc' | 'desc' }; onSort: (s: { key: SortKey; dir: 'asc' | 'desc' }) => void; open: boolean; onToggle: () => void }) {
  const set = (p: Partial<CompFilters>) => onChange({ ...filters, ...p });
  const activeCount = Object.entries(filters).filter(([k, v]) => {
    const d = defaultFilters()[k as keyof CompFilters];
    return JSON.stringify(v) !== JSON.stringify(d);
  }).length;
  const canDistance = distanceAvailable(comps, subject);
  return (
    <div className="card mb-3 p-2">
      <div className="flex flex-wrap items-center gap-2">
        <TextInput ariaLabel="Search comparables" value={filters.search} onChange={(v) => set({ search: v ?? '' })} placeholder="Search address, MLS, remarks…" className="max-w-xs" />
        <div className="flex items-center gap-1" role="group" aria-label="Status filter">
          {STATUS_OPTS.map((o) => {
            const on = filters.statuses.includes(o.value);
            return (
              <button key={o.value} aria-pressed={on} className={`rounded-full border px-2.5 py-1 text-xs font-medium ${on ? 'border-teal-600 bg-teal-600 text-white' : 'border-slate-300 bg-white text-slate-600 hover:bg-slate-50'}`} onClick={() => set({ statuses: on ? filters.statuses.filter((s) => s !== o.value) : [...filters.statuses, o.value] })}>
                {o.label}
              </button>
            );
          })}
        </div>
        <label className="flex items-center gap-1 text-xs text-slate-600">
          <input type="checkbox" checked={filters.includedOnly} onChange={(e) => set({ includedOnly: e.target.checked })} /> Included only
        </label>
        <div className="ml-auto flex items-center gap-1">
          <label className="text-xs text-slate-500" htmlFor="sort-key">
            Sort
          </label>
          <select id="sort-key" className="input w-auto py-1 text-xs" value={sort.key} onChange={(e) => onSort({ ...sort, key: e.target.value as SortKey })}>
            <option value="saleDate">Sale date</option>
            <option value="salePrice">Sale price</option>
            <option value="adjustedPrice">Adjusted price</option>
            <option value="areaAboveGrade">Area</option>
            <option value="pricePerSqft">$/SF</option>
            <option value="distance">Distance</option>
            <option value="address">Address</option>
          </select>
          <button className="btn-secondary btn-sm" onClick={() => onSort({ ...sort, dir: sort.dir === 'asc' ? 'desc' : 'asc' })} aria-label={`Sort ${sort.dir === 'asc' ? 'ascending' : 'descending'}`}>
            {sort.dir === 'asc' ? '↑' : '↓'}
          </button>
          <button className="btn-secondary btn-sm" onClick={onToggle} aria-expanded={open}>
            Filters{activeCount ? ` (${activeCount})` : ''}
          </button>
          {activeCount > 0 && (
            <button className="btn-secondary btn-sm" onClick={() => onChange(defaultFilters())}>
              Clear
            </button>
          )}
        </div>
      </div>
      {open && (
        <div className="mt-2 grid grid-cols-2 gap-2 border-t border-slate-200 pt-2 text-xs sm:grid-cols-3 lg:grid-cols-4">
          <div>
            <div className="label">Sale/contract date from</div>
            <TextInput ariaLabel="Date from" type="date" value={filters.dateFrom} onChange={(v) => set({ dateFrom: v })} />
          </div>
          <div>
            <div className="label">Date to</div>
            <TextInput ariaLabel="Date to" type="date" value={filters.dateTo} onChange={(v) => set({ dateTo: v })} />
          </div>
          <div>
            <div className="label">Price min / max</div>
            <div className="flex gap-1">
              <NumberInput ariaLabel="Price min" value={filters.priceMin} onChange={(v) => set({ priceMin: v })} placeholder="min" />
              <NumberInput ariaLabel="Price max" value={filters.priceMax} onChange={(v) => set({ priceMax: v })} placeholder="max" />
            </div>
          </div>
          <div>
            <div className="label">Above-grade area min / max (SF)</div>
            <div className="flex gap-1">
              <NumberInput ariaLabel="Area min" value={filters.areaMin} onChange={(v) => set({ areaMin: v })} placeholder="min" />
              <NumberInput ariaLabel="Area max" value={filters.areaMax} onChange={(v) => set({ areaMax: v })} placeholder="max" />
            </div>
          </div>
          <div>
            <div className="label">Lot size min / max (SF)</div>
            <div className="flex gap-1">
              <NumberInput ariaLabel="Lot min" value={filters.lotMin} onChange={(v) => set({ lotMin: v })} placeholder="min" />
              <NumberInput ariaLabel="Lot max" value={filters.lotMax} onChange={(v) => set({ lotMax: v })} placeholder="max" />
            </div>
          </div>
          <div>
            <div className="label">Neighborhood</div>
            <Select ariaLabel="Neighborhood" value={filters.neighborhood} onChange={(v) => set({ neighborhood: v })} options={uniq(comps, 'neighborhood').map((n) => ({ value: n, label: n }))} placeholder="Any" />
          </div>
          <div>
            <div className="label">Quality</div>
            <Select ariaLabel="Quality" value={filters.quality} onChange={(v) => set({ quality: v })} options={uniq(comps, 'quality').map((n) => ({ value: n, label: n }))} placeholder="Any" />
          </div>
          <div>
            <div className="label">Condition</div>
            <Select ariaLabel="Condition" value={filters.condition} onChange={(v) => set({ condition: v })} options={uniq(comps, 'condition').map((n) => ({ value: n, label: n }))} placeholder="Any" />
          </div>
          <div>
            <div className="label">Pool</div>
            <select aria-label="Pool filter" className="input" value={filters.pool} onChange={(e) => set({ pool: e.target.value as CompFilters['pool'] })}>
              <option value="any">Any</option>
              <option value="yes">With pool</option>
              <option value="no">Without pool</option>
            </select>
          </div>
          <div>
            <div className="label">Garage spaces at least</div>
            <NumberInput ariaLabel="Garage minimum" value={filters.garageMin} onChange={(v) => set({ garageMin: v })} />
          </div>
          <div>
            <div className="label">Distance within (mi)</div>
            <NumberInput ariaLabel="Distance maximum" value={filters.distanceMax} onChange={(v) => set({ distanceMax: v })} disabled={!canDistance} placeholder={canDistance ? '—' : 'No distance data'} />
            {!canDistance && <p className="mt-1 text-[10px] text-slate-500">Enabled when comps report distance or both subject and comps have coordinates.</p>}
          </div>
        </div>
      )}
    </div>
  );
}
