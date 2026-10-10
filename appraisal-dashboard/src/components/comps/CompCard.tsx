import type { Comparable, CompCalculation, PropertyAttributes } from '../../types';
import { fmtLot, fmtMoney, fmtNumber } from '../../engine/format';
import { formatDate } from '../../engine/dates';
import { distanceToSubject } from '../../engine/geo';
import { Badge } from '../ui/primitives';

export function StatusBadge({ status, raw }: { status: Comparable['attrs']['status']; raw: string | null }) {
  const tone = status === 'closed' ? 'green' : status === 'pending' ? 'amber' : status === 'active' ? 'navy' : 'slate';
  const label = status === 'closed' ? 'Closed' : status === 'pending' ? 'Pending' : status === 'active' ? 'Active' : status ? 'Other' : 'Unknown';
  return (
    <Badge tone={tone} title={raw ?? undefined}>
      {label}
    </Badge>
  );
}

export function CompCard({ comp, calc, subject, selected, onSelect, onToggle, rounding, index }: { comp: Comparable; calc: CompCalculation; subject: PropertyAttributes; selected: boolean; onSelect: () => void; onToggle: () => void; rounding: number; index: number | null }) {
  const a = comp.attrs;
  const dist = distanceToSubject(subject, a);
  const closed = a.status === 'closed';
  const baths = a.fullBaths === null && a.halfBaths === null ? '—' : `${a.fullBaths ?? '?'}F / ${a.halfBaths ?? '?'}H`;
  const parking = [a.garageSpaces !== null ? `${a.garageSpaces} gar` : null, a.carportSpaces ? `${a.carportSpaces} carport` : null].filter(Boolean).join(', ') || '—';
  const amenities = [a.pool ? 'Pool' : null, a.waterfront ? 'Waterfront' : null, a.waterView ? 'Water view' : null].filter(Boolean);
  const hasUnresolved = calc.unresolved.length > 0;
  return (
    <article
      className={`card relative flex cursor-pointer flex-col overflow-hidden transition hover:shadow-md ${selected ? 'ring-2 ring-teal-500' : ''} ${comp.included ? 'border-teal-300' : ''}`}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect();
        }
      }}
      tabIndex={0}
      role="button"
      aria-pressed={selected}
      aria-label={`${a.address ?? 'Unknown address'}${comp.included ? ', included' : ', excluded'}`}
    >
      <div className="relative h-32 bg-gradient-to-br from-navy-100 to-slate-200">
        {(comp.photoData ?? a.photoUrl) ? (
          <img src={comp.photoData ?? a.photoUrl ?? undefined} alt={a.address ?? 'Property photo'} className="h-full w-full object-cover" loading="lazy" onError={(e) => ((e.currentTarget.style.display = 'none'))} />
        ) : (
          <div className="flex h-full w-full flex-col items-center justify-center text-navy-300">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
              <path d="M3 11l9-8 9 8" />
              <path d="M5 10v10h14V10" />
              <path d="M10 20v-6h4v6" />
            </svg>
            <span className="text-[10px] uppercase tracking-wide">No photo</span>
          </div>
        )}
        <div className="absolute left-2 top-2">
          <StatusBadge status={a.status} raw={a.statusRaw} />
        </div>
        {index !== null && <div className="absolute right-2 top-2 rounded bg-navy-900/80 px-1.5 py-0.5 text-[11px] font-semibold text-white">Comp {index}</div>}
        <label className="absolute bottom-2 right-2 flex items-center gap-1 rounded bg-white/95 px-2 py-1 text-xs font-medium shadow" onClick={(e) => e.stopPropagation()}>
          <input type="checkbox" checked={comp.included} onChange={onToggle} aria-label={`Include ${a.address ?? 'comparable'}`} />
          {comp.included ? 'Included' : 'Include'}
        </label>
      </div>
      <div className="flex flex-1 flex-col gap-1.5 p-3">
        <div>
          <div className="truncate text-sm font-semibold text-navy-900" title={a.address ?? undefined}>
            {a.address ?? <span className="text-slate-400">Address unknown</span>}
          </div>
          <div className="truncate text-[11px] text-slate-500">
            {comp.synthetic && <span className="mr-1 rounded bg-amber-100 px-1 font-semibold uppercase text-amber-800">Demo</span>}
            {a.mlsNumber ? `MLS ${a.mlsNumber}` : 'MLS —'} · {a.city ?? '—'}
            {a.neighborhood ? ` · ${a.neighborhood}` : ''}
          </div>
        </div>
        <div className="flex items-baseline justify-between">
          <div>
            <div className="text-lg font-bold tabular-nums text-navy-900">{a.salePrice !== null ? fmtMoney(a.salePrice) : a.listPrice !== null ? <span className="text-slate-600">{fmtMoney(a.listPrice)} list</span> : '—'}</div>
            <div className="text-[11px] text-slate-500">{closed ? `Sold ${formatDate(a.saleDate)}` : a.contractDate ? `Contract ${formatDate(a.contractDate)}` : a.daysOnMarket !== null ? `${a.daysOnMarket} DOM` : ''}</div>
          </div>
          <div className="text-right">
            <div className="text-xs font-medium tabular-nums text-slate-700">{calc.pricePerSqft !== null ? `${fmtMoney(calc.pricePerSqft, { decimals: 0 })}/SF` : '—/SF'}</div>
            <div className="text-[10px] text-slate-400">unadjusted · {calc.pricePerSqftBasis}</div>
          </div>
        </div>
        <dl className="grid grid-cols-3 gap-x-2 gap-y-1 text-[11px]">
          <div>
            <dt className="text-slate-400">Area</dt>
            <dd className="font-medium tabular-nums">{a.areaAboveGrade !== null ? `${fmtNumber(a.areaAboveGrade)} SF` : '—'}</dd>
          </div>
          <div>
            <dt className="text-slate-400">Lot</dt>
            <dd className="truncate font-medium tabular-nums" title={fmtLot(a.lotSizeSqft, a.lotSizeRaw, a.lotSizeUnits)}>
              {a.lotSizeSqft !== null ? (a.lotSizeUnits === 'acres' ? `${fmtNumber(a.lotSizeRaw, 2)} ac` : `${fmtNumber(a.lotSizeSqft)} SF`) : '—'}
            </dd>
          </div>
          <div>
            <dt className="text-slate-400">Baths</dt>
            <dd className="font-medium">{baths}</dd>
          </div>
          <div>
            <dt className="text-slate-400">Built</dt>
            <dd className="font-medium">{a.yearBuilt ?? '—'}</dd>
          </div>
          <div>
            <dt className="text-slate-400">Parking</dt>
            <dd className="truncate font-medium">{parking}</dd>
          </div>
          <div>
            <dt className="text-slate-400">Distance</dt>
            <dd className="font-medium">{dist.miles !== null ? `${dist.miles.toFixed(2)} mi` : '—'}</dd>
          </div>
        </dl>
        {amenities.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {amenities.map((x) => (
              <Badge key={x} tone="teal">
                {x}
              </Badge>
            ))}
          </div>
        )}
        <div className="mt-auto flex items-center justify-between border-t border-slate-100 pt-2">
          <div className="text-[11px] text-slate-500">Adjusted</div>
          {closed ? (
            calc.adjustedPrice !== null ? (
              <div className="text-sm font-semibold tabular-nums text-teal-700">
                {fmtMoney(calc.adjustedPrice, { rounding })} <span className="text-[10px] font-normal text-slate-500">({fmtMoney(calc.netAdjustment, { signed: true, rounding })})</span>
              </div>
            ) : (
              <Badge tone={hasUnresolved ? 'amber' : 'slate'}>{hasUnresolved ? `${calc.unresolved.length} unresolved` : 'Not calculable'}</Badge>
            )
          ) : (
            <Badge tone="slate">Market context</Badge>
          )}
        </div>
        {calc.flags.length > 0 && comp.included && (
          <div className="text-[10px] text-amber-700" title={calc.flags.map((f) => f.message).join('\n')}>
            ⚑ {calc.flags.length} review flag{calc.flags.length > 1 ? 's' : ''}
          </div>
        )}
      </div>
    </article>
  );
}
