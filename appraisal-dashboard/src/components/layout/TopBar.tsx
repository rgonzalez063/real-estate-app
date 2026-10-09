import { useEffect, useRef, useState } from 'react';
import { useAppStore } from '../../store/useAppStore';
import type { Project, Scenario, ScenarioCalculation } from '../../types';
import { buildGridCsv, downloadText, exportProjectJson } from '../../engine/export';
import { Badge, TextInput } from '../ui/primitives';

function Menu({ label, items }: { label: string; items: { label: string; onClick: () => void; disabled?: boolean }[] }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);
  return (
    <div className="relative" ref={ref}>
      <button className="btn-ghost btn-sm" onClick={() => setOpen(!open)} aria-haspopup="menu" aria-expanded={open}>
        {label} ▾
      </button>
      {open && (
        <div className="absolute right-0 z-40 mt-1 w-56 rounded-md border border-slate-200 bg-white py-1 text-sm text-slate-800 shadow-lg" role="menu">
          {items.map((it) => (
            <button key={it.label} role="menuitem" className="block w-full px-3 py-1.5 text-left hover:bg-slate-50 disabled:opacity-50" disabled={it.disabled} onClick={() => { setOpen(false); it.onClick(); }}>
              {it.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function TopBar({ project, scenario, calc }: { project: Project; scenario: Scenario; calc: ScenarioCalculation }) {
  const { renameProject, setEffectiveDate, setUi, save, dirty, lastSavedAt, undo, redo, past, future, setActiveScenario, addScenario, duplicateActiveScenario, renameScenario, deleteScenario, toast } = useAppStore();
  const ui = useAppStore((s) => s.ui);
  const [editingName, setEditingName] = useState(false);
  const safeName = project.name.replace(/[^a-z0-9-_]+/gi, '_');

  const scenarioItems = [
    ...project.scenarios.map((s) => ({ label: `${s.id === scenario.id ? '● ' : '○ '}${s.name}`, onClick: () => setActiveScenario(s.id) })),
    { label: '＋ New blank scenario', onClick: () => { const n = window.prompt('Scenario name', `Scenario ${project.scenarios.length + 1}`); if (n) addScenario(n); } },
    { label: '⧉ Duplicate current scenario', onClick: () => { const n = window.prompt('Name for the copy', `${scenario.name} (copy)`); if (n) duplicateActiveScenario(n); } },
    { label: '✎ Rename current scenario', onClick: () => { const n = window.prompt('Scenario name', scenario.name); if (n) renameScenario(scenario.id, n); } },
    { label: '⇄ Compare scenarios', onClick: () => setUi({ compareOpen: true }) },
    { label: '🗑 Delete current scenario', disabled: project.scenarios.length <= 1, onClick: () => { if (window.confirm(`Delete scenario "${scenario.name}"?`)) deleteScenario(scenario.id); } },
  ];

  return (
    <header className="no-print flex flex-wrap items-center gap-2 bg-navy-900 px-3 py-2 text-white shadow">
      <div className="flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded bg-teal-500 text-sm font-bold text-navy-950" aria-hidden>
          GA
        </div>
        <div className="leading-tight">
          <div className="text-[10px] uppercase tracking-widest text-slate-300">Gonzalez Appraisal Company</div>
          {editingName ? (
            <input autoFocus className="rounded bg-navy-800 px-1 text-sm text-white" value={project.name} onChange={(e) => renameProject(e.target.value)} onBlur={() => setEditingName(false)} onKeyDown={(e) => e.key === 'Enter' && setEditingName(false)} aria-label="Project name" />
          ) : (
            <button className="text-sm font-semibold hover:underline" onClick={() => setEditingName(true)} title="Rename project">
              {project.name} {project.isDemo && <Badge tone="amber">Demo data</Badge>}
            </button>
          )}
        </div>
      </div>
      <div className="flex items-center gap-1 rounded bg-navy-800 px-2 py-1">
        <label className="text-[11px] text-slate-300" htmlFor="top-effective-date">
          Effective
        </label>
        <TextInput id="top-effective-date" type="date" value={scenario.effectiveDate} onChange={setEffectiveDate} className="!w-36 !py-0.5 !text-xs" />
      </div>
      <Menu label={`Scenario: ${scenario.name}`} items={scenarioItems} />
      <div className="ml-auto flex items-center gap-1">
        <button className="btn-ghost btn-sm" onClick={undo} disabled={!past.length} title="Undo (Ctrl+Z)" aria-label="Undo">
          ↶ Undo
        </button>
        <button className="btn-ghost btn-sm" onClick={redo} disabled={!future.length} title="Redo (Ctrl+Shift+Z)" aria-label="Redo">
          ↷ Redo
        </button>
        <button className="btn-ghost btn-sm" onClick={() => setUi({ importOpen: true })}>
          ⇪ Import CSV
        </button>
        <button className="btn-ghost btn-sm" onClick={() => void save().then(() => toast('Project saved locally.'))} title={lastSavedAt ? `Last saved ${new Date(lastSavedAt).toLocaleTimeString()}` : 'Not saved yet'}>
          {dirty ? '● Save' : '✓ Saved'}
        </button>
        <Menu
          label="Export"
          items={[
            { label: 'Adjustment grid (CSV)', onClick: () => downloadText(`${safeName}_${scenario.name.replace(/\W+/g, '_')}_grid.csv`, buildGridCsv(scenario, calc), 'text/csv') },
            { label: 'Project backup (JSON)', onClick: () => downloadText(`${safeName}_backup.json`, exportProjectJson(project), 'application/json') },
            { label: 'Print-ready report', onClick: () => setUi({ reportOpen: true }) },
          ]}
        />
        <button className="btn-ghost btn-sm" onClick={() => setUi({ projectsOpen: true })}>
          Projects
        </button>
        <div className="hidden items-center gap-1 lg:flex">
          <button className={`btn-ghost btn-sm ${ui.leftOpen ? '' : 'opacity-60'}`} onClick={() => setUi({ leftOpen: !ui.leftOpen })} aria-pressed={ui.leftOpen} title="Toggle left panel">
            ◧
          </button>
          <button className={`btn-ghost btn-sm ${ui.rightOpen ? '' : 'opacity-60'}`} onClick={() => setUi({ rightOpen: !ui.rightOpen })} aria-pressed={ui.rightOpen} title="Toggle right panel">
            ◨
          </button>
        </div>
      </div>
    </header>
  );
}
