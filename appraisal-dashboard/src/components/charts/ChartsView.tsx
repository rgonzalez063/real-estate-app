import { useMemo } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, Legend, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts';
import { useAppStore } from '../../store/useAppStore';
import type { Scenario, ScenarioCalculation } from '../../types';
import { fmtMoney, fmtNumber } from '../../engine/format';
import { dayNumber, formatDate } from '../../engine/dates';
import { compIndexMap } from '../../engine/compIndex';
import { EmptyState } from '../ui/primitives';

const NAVY = '#27416a';
const TEAL = '#2a958a';
const SLATE = '#94a3b8';
const AMBER = '#d97706';
const RED = '#b91c1c';

function ChartCard({ title, children, note }: { title: string; children: React.ReactNode; note?: string }) {
  return (
    <div className="card p-3">
      <h3 className="mb-1 text-sm font-semibold text-navy-900">{title}</h3>
      {note && <p className="mb-2 text-[11px] text-slate-500">{note}</p>}
      <div className="h-64">{children}</div>
    </div>
  );
}

const money = (v: number) => fmtMoney(v);
const moneyShort = (v: number) => (Math.abs(v) >= 1000 ? `$${Math.round(v / 1000)}k` : `$${v}`);

export function ChartsView({ scenario, calc }: { scenario: Scenario; calc: ScenarioCalculation }) {
  const selectedId = useAppStore((s) => s.ui.selectedCompId);
  const selectComp = useAppStore((s) => s.selectComp);
  const indexMap = useMemo(() => compIndexMap(scenario), [scenario]);
  const rounding = scenario.settings.rounding;

  const included = scenario.comparables.filter((c) => c.included && c.attrs.status === 'closed');
  const closedAll = scenario.comparables.filter((c) => c.attrs.status === 'closed' && c.attrs.salePrice !== null);
  const r = calc.reconciliation;
  const finalValue = scenario.finalOpinion.value;

  const beforeAfter = included
    .filter((c) => c.attrs.salePrice !== null)
    .map((c) => ({ id: c.id, name: `Comp ${indexMap[c.id]}`, sale: c.attrs.salePrice as number, adjusted: calc.comps[c.id].adjustedPrice, unresolved: calc.comps[c.id].adjustedPrice === null }));

  const selected = selectedId ? scenario.comparables.find((c) => c.id === selectedId) : undefined;
  const selectedCalc = selectedId ? calc.comps[selectedId] : undefined;
  const { waterfall, waterfallFloor } = useMemo(() => {
    if (!selected || !selectedCalc || selected.attrs.salePrice === null) return { waterfall: [], waterfallFloor: 0 };
    const steps: { name: string; base: number; delta: number; sign: 'pos' | 'neg' | 'total'; value: number }[] = [];
    let running = selected.attrs.salePrice;
    const levels = [running];
    steps.push({ name: 'Sale price', base: 0, delta: running, sign: 'total', value: running });
    for (const l of selectedCalc.lines) {
      if (l.amount === null || l.status === 'inactive' || l.status === 'notApplicable') continue;
      const next = running + l.amount;
      steps.push({ name: l.label, base: Math.min(running, next), delta: Math.abs(l.amount), sign: l.amount >= 0 ? 'pos' : 'neg', value: l.amount });
      running = next;
      levels.push(next);
    }
    if (selectedCalc.adjustedPrice !== null) steps.push({ name: 'Adjusted', base: 0, delta: selectedCalc.adjustedPrice, sign: 'total', value: selectedCalc.adjustedPrice });
    // Chart floor: a round number a little below the lowest running level so adjustment steps stay legible.
    const minLevel = Math.min(...levels);
    const floor = Math.max(0, Math.floor((minLevel * 0.9) / 10000) * 10000);
    const shifted = steps.map((s) => (s.sign === 'total' ? { ...s, base: 0, delta: s.delta - floor } : { ...s, base: s.base - floor }));
    return { waterfall: shifted, waterfallFloor: floor };
  }, [selected, selectedCalc]);

  const distribution = included
    .filter((c) => calc.comps[c.id].adjustedPrice !== null)
    .map((c) => ({ id: c.id, name: `Comp ${indexMap[c.id]}`, adjusted: calc.comps[c.id].adjustedPrice as number, weight: calc.comps[c.id].normalizedWeight }))
    .sort((a, b) => a.adjusted - b.adjusted);

  const scatter = closedAll
    .filter((c) => c.attrs.areaAboveGrade !== null)
    .map((c) => ({ id: c.id, x: c.attrs.areaAboveGrade as number, y: c.attrs.salePrice as number, name: c.attrs.address ?? c.id, included: c.included }));
  const subjectPoint = scenario.subject.attrs.areaAboveGrade !== null && r.weightedIndication !== null ? [{ x: scenario.subject.attrs.areaAboveGrade, y: r.weightedIndication, name: 'Subject (weighted indication)' }] : [];

  const timeline = closedAll
    .filter((c) => c.attrs.saleDate !== null)
    .map((c) => ({ id: c.id, x: dayNumber(c.attrs.saleDate as string), y: c.attrs.salePrice as number, date: c.attrs.saleDate as string, name: c.attrs.address ?? c.id, included: c.included }))
    .sort((a, b) => a.x - b.x);

  if (!scenario.comparables.length) {
    return <EmptyState title="No data to chart">Import comparables to see charts.</EmptyState>;
  }

  const pointFill = (id: string, included: boolean) => (id === selectedId ? AMBER : included ? TEAL : SLATE);

  return (
    <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
      <ChartCard title="Sale price before and after adjustments" note="Included closed sales. Click a bar to select the comparable.">
        {beforeAfter.length ? (
          <ResponsiveContainer>
            <BarChart data={beforeAfter} onClick={(e) => e && e.activePayload && selectComp(e.activePayload[0].payload.id)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={moneyShort} tick={{ fontSize: 11 }} width={56} />
              <Tooltip formatter={(v: number) => money(v)} />
              <Legend />
              <Bar dataKey="sale" name="Sale price" fill={SLATE}>
                {beforeAfter.map((d) => (
                  <Cell key={d.id} stroke={d.id === selectedId ? AMBER : 'none'} strokeWidth={2} />
                ))}
              </Bar>
              <Bar dataKey="adjusted" name="Adjusted price" fill={TEAL}>
                {beforeAfter.map((d) => (
                  <Cell key={d.id} fill={d.unresolved ? '#fde68a' : TEAL} stroke={d.id === selectedId ? AMBER : 'none'} strokeWidth={2} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-slate-500">Include closed sales to compare.</p>
        )}
      </ChartCard>

      <ChartCard title={`Adjustment waterfall${selected ? ` — ${selected.attrs.address ?? 'selected comparable'}` : ''}`} note={selected ? 'Each step shows one signed adjustment from the original sale price to the adjusted price.' : 'Select a comparable (card, grid column or chart point) to see its waterfall.'}>
        {waterfall.length ? (
          <ResponsiveContainer>
            <BarChart data={waterfall}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} angle={-20} textAnchor="end" height={50} />
              <YAxis tickFormatter={(v: number) => moneyShort(v + waterfallFloor)} tick={{ fontSize: 11 }} width={56} />
              <Tooltip formatter={(_v: number, _n, item) => [fmtMoney(item.payload.value, { signed: item.payload.sign !== 'total', rounding }), item.payload.sign === 'total' ? 'Amount' : 'Adjustment']} />
              <Bar dataKey="base" stackId="w" fill="transparent" />
              <Bar dataKey="delta" stackId="w">
                {waterfall.map((d, i) => (
                  <Cell key={i} fill={d.sign === 'total' ? NAVY : d.sign === 'pos' ? TEAL : RED} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-slate-500">{selected ? 'Selected comparable has no sale price.' : 'Nothing selected.'}</p>
        )}
      </ChartCard>

      <ChartCard title="Adjusted price distribution" note="Observed range of adjusted prices (not a statistical confidence interval). Lines mark the weighted indication and the final opinion.">
        {distribution.length ? (
          <ResponsiveContainer>
            <BarChart data={distribution} onClick={(e) => e && e.activePayload && selectComp(e.activePayload[0].payload.id)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={moneyShort} tick={{ fontSize: 11 }} width={56} domain={['auto', 'auto']} />
              <Tooltip formatter={(v: number, _n, item) => [`${money(v)} · weight ${item.payload.weight !== null ? (item.payload.weight * 100).toFixed(1) + '%' : '—'}`, 'Adjusted']} />
              <Bar dataKey="adjusted" name="Adjusted price" fill={NAVY}>
                {distribution.map((d) => (
                  <Cell key={d.id} fill={d.id === selectedId ? AMBER : NAVY} />
                ))}
              </Bar>
              {r.weightedIndication !== null && <ReferenceLine y={r.weightedIndication} stroke={TEAL} strokeWidth={2} label={{ value: `Weighted ${fmtMoney(r.weightedIndication, { rounding })}`, position: 'insideTopRight', fontSize: 11, fill: TEAL }} />}
              {finalValue !== null && <ReferenceLine y={finalValue} stroke={AMBER} strokeDasharray="4 4" strokeWidth={2} label={{ value: `Final ${fmtMoney(finalValue)}`, position: 'insideBottomRight', fontSize: 11, fill: AMBER }} />}
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-slate-500">No calculable included sales.</p>
        )}
      </ChartCard>

      <ChartCard title="Sale price vs. above-grade area" note="All closed sales; teal = included, grey = excluded, amber = selected. The subject marker is the weighted indication, not a sale.">
        {scatter.length ? (
          <ResponsiveContainer>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="x" type="number" name="Area" unit=" SF" tick={{ fontSize: 11 }} domain={['auto', 'auto']} tickFormatter={(v) => fmtNumber(v)} />
              <YAxis dataKey="y" type="number" name="Price" tickFormatter={moneyShort} tick={{ fontSize: 11 }} width={56} domain={['auto', 'auto']} />
              <ZAxis range={[60, 60]} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(v: number, name: string) => (name === 'Price' ? money(v) : `${fmtNumber(v)} SF`)} labelFormatter={() => ''} content={({ payload }) => (payload && payload.length ? <div className="rounded border border-slate-200 bg-white p-2 text-xs shadow">{payload[0].payload.name}<br />{fmtNumber(payload[0].payload.x)} SF · {money(payload[0].payload.y)}</div> : null)} />
              <Scatter data={scatter} onClick={(d) => selectComp(d.id)}>
                {scatter.map((d) => (
                  <Cell key={d.id} fill={pointFill(d.id, d.included)} />
                ))}
              </Scatter>
              {subjectPoint.length > 0 && <Scatter data={subjectPoint} fill={NAVY} shape="diamond" />}
            </ScatterChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-slate-500">No closed sales with area and price.</p>
        )}
      </ChartCard>

      <ChartCard title="Sale price timeline" note="Closed sales by sale date. Click a point to select.">
        {timeline.length ? (
          <ResponsiveContainer>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="x" type="number" domain={['auto', 'auto']} tick={{ fontSize: 11 }} tickFormatter={(v) => formatDate(new Date(v * 86400000).toISOString().slice(0, 10))} />
              <YAxis dataKey="y" type="number" tickFormatter={moneyShort} tick={{ fontSize: 11 }} width={56} domain={['auto', 'auto']} />
              <ZAxis range={[60, 60]} />
              <Tooltip content={({ payload }) => (payload && payload.length ? <div className="rounded border border-slate-200 bg-white p-2 text-xs shadow">{payload[0].payload.name}<br />{formatDate(payload[0].payload.date)} · {money(payload[0].payload.y)}</div> : null)} />
              <Scatter data={timeline} line={{ stroke: '#cbd5e1' }} onClick={(d) => selectComp(d.id)}>
                {timeline.map((d) => (
                  <Cell key={d.id} fill={pointFill(d.id, d.included)} />
                ))}
              </Scatter>
              {scenario.effectiveDate && <ReferenceLine x={dayNumber(scenario.effectiveDate)} stroke={NAVY} strokeDasharray="4 4" label={{ value: 'Effective date', fontSize: 10, position: 'insideTopLeft', fill: NAVY }} />}
            </ScatterChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-slate-500">No closed sales with dates.</p>
        )}
      </ChartCard>
    </div>
  );
}
