import { useEffect, useState } from 'react';
import { useAppStore, activeScenarioOf } from './store/useAppStore';
import { useCalculation } from './store/useCalculation';
import { TopBar } from './components/layout/TopBar';
import { SummaryBar } from './components/layout/SummaryBar';
import { LeftPanel } from './components/layout/LeftPanel';
import { RightPanel } from './components/layout/RightPanel';
import { CardsView } from './components/comps/CardsView';
import { ComparisonGrid } from './components/grid/ComparisonGrid';
import { ChartsView } from './components/charts/ChartsView';
import { ImportWizard } from './components/import/ImportWizard';
import { ProjectsModal } from './components/scenarios/ProjectsModal';
import { ScenarioCompare } from './components/scenarios/ScenarioCompare';
import { PrintReport } from './components/report/PrintReport';
import type { CenterView, MobileTab } from './store/useAppStore';

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => (typeof window !== 'undefined' ? window.matchMedia(query).matches : true));
  useEffect(() => {
    const mql = window.matchMedia(query);
    const onChange = () => setMatches(mql.matches);
    onChange();
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, [query]);
  return matches;
}

function Toast() {
  const toast = useAppStore((s) => s.ui.toast);
  if (!toast) return null;
  return (
    <div className={`fixed bottom-4 left-1/2 z-[60] -translate-x-1/2 rounded-md px-4 py-2 text-sm shadow-lg ${toast.kind === 'error' ? 'bg-red-700 text-white' : 'bg-navy-900 text-white'}`} role="alert">
      {toast.message}
    </div>
  );
}

function Welcome() {
  const { createProject, loadDemo, setUi, projects } = useAppStore();
  return (
    <div className="flex min-h-screen items-center justify-center bg-navy-900 p-6">
      <div className="card w-full max-w-lg p-6">
        <div className="text-[11px] uppercase tracking-widest text-slate-500">Gonzalez Appraisal Company</div>
        <h1 className="text-xl font-bold text-navy-900">Residential appraisal dashboard</h1>
        <p className="mt-2 text-sm text-slate-600">Import comparable CSV exports, select comps, apply supported adjustments, and watch the indicated value update in real time. Everything stays in this browser.</p>
        <div className="mt-4 flex flex-wrap gap-2">
          <button className="btn-primary" onClick={() => createProject('New appraisal')}>
            New project
          </button>
          <button className="btn-secondary" onClick={loadDemo}>
            Open synthetic demo
          </button>
          {projects.length > 0 && (
            <button className="btn-secondary" onClick={() => setUi({ projectsOpen: true })}>
              Saved projects ({projects.length})
            </button>
          )}
        </div>
      </div>
      <ProjectsModal />
    </div>
  );
}

export default function App() {
  const init = useAppStore((s) => s.init);
  const ready = useAppStore((s) => s.ready);
  const project = useAppStore((s) => s.project);
  const ui = useAppStore((s) => s.ui);
  const setUi = useAppStore((s) => s.setUi);
  const undo = useAppStore((s) => s.undo);
  const redo = useAppStore((s) => s.redo);
  const save = useAppStore((s) => s.save);
  const dirty = useAppStore((s) => s.dirty);
  const scenario = activeScenarioOf(project);
  const calc = useCalculation(scenario);
  const isDesktop = useMediaQuery('(min-width: 1024px)');

  useEffect(() => {
    void init();
  }, [init]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable);
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        if (typing) return;
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        if (typing) return;
        e.preventDefault();
        redo();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        void save();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [undo, redo, save]);

  useEffect(() => {
    const onUnload = (e: BeforeUnloadEvent) => {
      if (dirty) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', onUnload);
    return () => window.removeEventListener('beforeunload', onUnload);
  }, [dirty]);

  if (!ready) {
    return <div className="flex min-h-screen items-center justify-center text-sm text-slate-500">Loading…</div>;
  }
  if (!project || !scenario) return <Welcome />;
  if (ui.reportOpen) return <PrintReport project={project} scenario={scenario} calc={calc} />;

  const views: { key: CenterView; label: string }[] = [
    { key: 'cards', label: 'Cards' },
    { key: 'grid', label: 'Comparison grid' },
    { key: 'charts', label: 'Charts' },
  ];
  const tabs: { key: MobileTab; label: string }[] = [
    { key: 'subject', label: 'Subject & rules' },
    { key: 'comps', label: 'Comparables' },
    { key: 'details', label: 'Details & value' },
  ];
  const center = (
    <main className="min-w-0 flex-1 overflow-y-auto p-3">
      <div className="mb-3 flex items-center gap-1" role="tablist" aria-label="Center view">
        {views.map((v) => (
          <button key={v.key} role="tab" aria-selected={ui.view === v.key} className={`rounded-md px-3 py-1.5 text-sm font-medium ${ui.view === v.key ? 'bg-navy-900 text-white' : 'bg-white text-slate-700 hover:bg-slate-50'}`} onClick={() => setUi({ view: v.key })}>
            {v.label}
          </button>
        ))}
        <span className="ml-auto whitespace-nowrap text-[11px] text-slate-500">
          {scenario.comparables.length} comp{scenario.comparables.length === 1 ? '' : 's'} · {scenario.comparables.filter((c) => c.included).length} included
        </span>
      </div>
      {ui.view === 'cards' && <CardsView scenario={scenario} calc={calc} />}
      {ui.view === 'grid' && <ComparisonGrid scenario={scenario} calc={calc} />}
      {ui.view === 'charts' && <ChartsView scenario={scenario} calc={calc} />}
    </main>
  );

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <TopBar project={project} scenario={scenario} calc={calc} />
      <SummaryBar scenario={scenario} calc={calc} />
      {isDesktop ? (
      <div className="flex min-h-0 flex-1">
        {ui.leftOpen && (
          <aside className="w-80 shrink-0 overflow-hidden border-r border-slate-200 bg-white" aria-label="Subject and assumptions">
            <LeftPanel scenario={scenario} warnings={calc.warnings} />
          </aside>
        )}
        {center}
        {ui.rightOpen && (
          <aside className="w-[22rem] shrink-0 overflow-hidden border-l border-slate-200 bg-white" aria-label="Property details and reconciliation">
            <RightPanel scenario={scenario} calc={calc} />
          </aside>
        )}
      </div>
      ) : (
      <div className="flex min-h-0 flex-1 flex-col">
        <div className="flex border-b border-slate-200 bg-white" role="tablist" aria-label="Mobile sections">
          {tabs.map((t) => (
            <button key={t.key} role="tab" aria-selected={ui.mobileTab === t.key} className={`flex-1 border-b-2 px-2 py-2 text-xs font-medium ${ui.mobileTab === t.key ? 'border-teal-600 text-navy-900' : 'border-transparent text-slate-500'}`} onClick={() => setUi({ mobileTab: t.key })}>
              {t.label}
            </button>
          ))}
        </div>
        <div className="min-h-0 flex-1 overflow-hidden">
          {ui.mobileTab === 'subject' && (
            <div className="h-full overflow-y-auto bg-white">
              <LeftPanel scenario={scenario} warnings={calc.warnings} />
            </div>
          )}
          {ui.mobileTab === 'comps' && <div className="flex h-full flex-col">{center}</div>}
          {ui.mobileTab === 'details' && (
            <div className="h-full overflow-y-auto bg-white">
              <RightPanel scenario={scenario} calc={calc} />
            </div>
          )}
        </div>
      </div>
      )}
      <ImportWizard />
      <ProjectsModal />
      <ScenarioCompare project={project} />
      <Toast />
    </div>
  );
}
