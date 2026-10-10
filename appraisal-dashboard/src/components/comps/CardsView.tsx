import { useMemo, useState } from 'react';
import { useAppStore } from '../../store/useAppStore';
import type { Comparable, Scenario, ScenarioCalculation } from '../../types';
import { EmptyState } from '../ui/primitives';
import { compIndexMap } from '../../engine/compIndex';
import { CompCard } from './CompCard';
import { FilterBar } from './FilterBar';
import { applyFilters, defaultFilters, sortComps, type CompFilters, type SortKey } from './filters';

export function CardsView({ scenario, calc }: { scenario: Scenario; calc: ScenarioCalculation }) {
  const selectedId = useAppStore((s) => s.ui.selectedCompId);
  const selectComp = useAppStore((s) => s.selectComp);
  const toggleIncluded = useAppStore((s) => s.toggleIncluded);
  const setUi = useAppStore((s) => s.setUi);
  const [filters, setFilters] = useState<CompFilters>(defaultFilters);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [sort, setSort] = useState<{ key: SortKey; dir: 'asc' | 'desc' }>({ key: 'saleDate', dir: 'desc' });
  const indexMap = useMemo(() => compIndexMap(scenario), [scenario]);

  const visible = useMemo(() => sortComps(applyFilters(scenario.comparables, filters, scenario.subject.attrs), sort.key, sort.dir, calc.comps, scenario.subject.attrs), [scenario, filters, sort, calc]);
  const closed = visible.filter((c) => c.attrs.status === 'closed');
  const context = visible.filter((c) => c.attrs.status === 'active' || c.attrs.status === 'pending');
  const other = visible.filter((c) => c.attrs.status !== 'closed' && c.attrs.status !== 'active' && c.attrs.status !== 'pending');

  if (!scenario.comparables.length) {
    return (
      <EmptyState title="No comparables yet">
        <p>Import CSV exports (HARMLS / Matrix style) or MLS PDFs and photos. Everything is reviewed before it is added, and you can import more files at any time.</p>
        <div className="flex flex-wrap justify-center gap-2">
          <button className="btn-primary" onClick={() => setUi({ importOpen: true })}>
            Import CSV
          </button>
          <button className="btn-secondary" onClick={() => setUi({ docImportOpen: true })}>
            Import PDF / images
          </button>
        </div>
      </EmptyState>
    );
  }

  const grid = (list: Comparable[]) => (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
      {list.map((c) => (
        <CompCard key={c.id} comp={c} calc={calc.comps[c.id]} subject={scenario.subject.attrs} selected={selectedId === c.id} onSelect={() => { selectComp(c.id); setUi({ mobileTab: 'details' }); }} onToggle={() => toggleIncluded(c.id)} rounding={scenario.settings.rounding} index={indexMap[c.id] ?? null} />
      ))}
    </div>
  );

  return (
    <div>
      <FilterBar filters={filters} onChange={setFilters} comps={scenario.comparables} subject={scenario.subject.attrs} sort={sort} onSort={setSort} open={filtersOpen} onToggle={() => setFiltersOpen(!filtersOpen)} />
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-navy-900">Closed sales <span className="font-normal text-slate-500">({closed.length} shown · {scenario.comparables.filter((c) => c.included && c.attrs.status === 'closed').length} included)</span></h2>
        <span className="text-[11px] text-slate-500">Closed sales support the value indication.</span>
      </div>
      {closed.length ? grid(closed) : <p className="mb-3 text-sm text-slate-500">No closed sales match the current filters.</p>}
      {context.length > 0 && (
        <>
          <div className="mb-2 mt-5 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-navy-900">Market context: listings &amp; pending <span className="font-normal text-slate-500">({context.length})</span></h2>
            <span className="text-[11px] text-slate-500">Shown for context only; never enters the closed-sale estimate.</span>
          </div>
          {grid(context)}
        </>
      )}
      {other.length > 0 && (
        <>
          <h2 className="mb-2 mt-5 text-sm font-semibold text-navy-900">Other / unknown status <span className="font-normal text-slate-500">({other.length})</span></h2>
          {grid(other)}
        </>
      )}
    </div>
  );
}
