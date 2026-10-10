import { create } from 'zustand';
import type { AdjustmentOverride, AdjustmentRule, CategoryResolution, Comparable, FinalOpinion, MappingPreset, ParseIssue, Project, PropertyAttributes, Scenario, ScenarioSettings } from '../types';
import { duplicateScenario, newProject, newScenario } from '../engine/defaults';
import { buildDemoProject } from '../engine/demo';
import { makeId } from '../engine/ids';
import * as persistence from './persistence';

export type CenterView = 'cards' | 'grid' | 'charts';
export type MobileTab = 'subject' | 'comps' | 'details';

interface HistoryEntry {
  scenarioId: string;
  snapshot: Scenario;
  label: string;
  key?: string;
  at: number;
}

export interface UiState {
  view: CenterView;
  selectedCompId: string | null;
  leftOpen: boolean;
  rightOpen: boolean;
  mobileTab: MobileTab;
  importOpen: boolean;
  docImportOpen: boolean;
  compareOpen: boolean;
  reportOpen: boolean;
  projectsOpen: boolean;
  toast: { id: number; message: string; kind: 'info' | 'error' } | null;
}

export interface AppState {
  project: Project | null;
  past: HistoryEntry[];
  future: HistoryEntry[];
  ui: UiState;
  projects: persistence.ProjectSummary[];
  globalPresets: MappingPreset[];
  dirty: boolean;
  lastSavedAt: string | null;
  ready: boolean;

  // lifecycle
  init: () => Promise<void>;
  createProject: (name?: string) => void;
  openProject: (id: string) => Promise<void>;
  loadDemo: () => void;
  importProject: (project: Project) => void;
  save: () => Promise<void>;
  removeProject: (id: string) => Promise<void>;
  renameProject: (name: string) => void;
  refreshProjects: () => Promise<void>;

  // ui
  setUi: (patch: Partial<UiState>) => void;
  selectComp: (id: string | null) => void;
  toast: (message: string, kind?: 'info' | 'error') => void;

  // scenario edits
  mutateScenario: (label: string, fn: (s: Scenario) => void, opts?: { key?: string; undoable?: boolean }) => void;
  setEffectiveDate: (date: string | null) => void;
  setSubjectAttr: <K extends keyof PropertyAttributes>(key: K, value: PropertyAttributes[K]) => void;
  setSubjectAttrs: (patch: Partial<PropertyAttributes>) => void;
  setSubjectNotes: (notes: string) => void;
  setSubjectPhotos: (urls: string[]) => void;
  /** Replace subject attributes with imported values (unknown imported values leave existing ones untouched). */
  setSubjectFromImport: (attrs: Partial<PropertyAttributes>, source: { file: string; row: Record<string, string>; issues: ParseIssue[] }) => void;
  addComparables: (comps: Comparable[]) => void;
  removeComparables: (ids: string[]) => void;
  toggleIncluded: (id: string, included?: boolean) => void;
  setCompWeight: (id: string, weight: number) => void;
  setCompNote: (id: string, note: string) => void;
  setCompGridNote: (id: string, note: string) => void;
  setCompAttrs: (id: string, patch: Partial<PropertyAttributes>) => void;
  setOverride: (compId: string, ruleId: string, override: AdjustmentOverride | null) => void;
  setResolution: (compId: string, ruleId: string, resolution: CategoryResolution | null) => void;
  updateRule: (ruleId: string, patch: Partial<AdjustmentRule>) => void;
  addCustomRule: (label: string, kind: AdjustmentRule['kind'], field: AdjustmentRule['field']) => void;
  removeRule: (ruleId: string) => void;
  setSettings: (patch: Partial<ScenarioSettings>) => void;
  setFlags: (patch: Partial<ScenarioSettings['flags']>) => void;
  setFinalOpinion: (patch: Partial<FinalOpinion>) => void;
  setReconciliationNotes: (notes: string) => void;

  // scenarios
  addScenario: (name: string) => void;
  duplicateActiveScenario: (name: string) => void;
  deleteScenario: (id: string) => void;
  renameScenario: (id: string, name: string) => void;
  setActiveScenario: (id: string) => void;

  // presets
  savePreset: (preset: MappingPreset) => Promise<void>;
  deletePreset: (id: string) => Promise<void>;

  undo: () => void;
  redo: () => void;
}

const HISTORY_LIMIT = 100;
const COALESCE_MS = 1000;

function clone<T>(v: T): T {
  return JSON.parse(JSON.stringify(v));
}

export function activeScenarioOf(project: Project | null): Scenario | null {
  if (!project) return null;
  return project.scenarios.find((s) => s.id === project.activeScenarioId) ?? project.scenarios[0] ?? null;
}

let toastCounter = 0;
let saveTimer: ReturnType<typeof setTimeout> | null = null;

export const useAppStore = create<AppState>((set, get) => {
  const scheduleAutosave = () => {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      void get().save();
    }, 1200);
  };

  const touch = (project: Project) => {
    project.updatedAt = new Date().toISOString();
    set({ project, dirty: true });
    scheduleAutosave();
  };

  return {
    project: null,
    past: [],
    future: [],
    ui: {
      view: 'cards',
      selectedCompId: null,
      leftOpen: true,
      rightOpen: true,
      mobileTab: 'comps',
      importOpen: false,
      docImportOpen: false,
      compareOpen: false,
      reportOpen: false,
      projectsOpen: false,
      toast: null,
    },
    projects: [],
    globalPresets: [],
    dirty: false,
    lastSavedAt: null,
    ready: false,

    init: async () => {
      const [projects, globalPresets] = await Promise.all([persistence.listProjects(), persistence.loadGlobalPresets()]);
      set({ projects, globalPresets, ready: true });
      if (projects.length) {
        await get().openProject(projects[0].id);
      }
    },

    createProject: (name) => {
      const project = newProject(name);
      set({ project, past: [], future: [], dirty: true, ui: { ...get().ui, selectedCompId: null, projectsOpen: false } });
      scheduleAutosave();
    },

    openProject: async (id) => {
      const project = await persistence.loadProject(id);
      if (!project) {
        get().toast('Project not found.', 'error');
        return;
      }
      set({ project, past: [], future: [], dirty: false, lastSavedAt: project.updatedAt, ui: { ...get().ui, selectedCompId: null, projectsOpen: false } });
    },

    loadDemo: () => {
      const project = buildDemoProject();
      set({ project, past: [], future: [], dirty: true, ui: { ...get().ui, selectedCompId: null, projectsOpen: false, view: 'cards' } });
      scheduleAutosave();
    },

    importProject: (project) => {
      set({ project, past: [], future: [], dirty: true, ui: { ...get().ui, selectedCompId: null, projectsOpen: false } });
      scheduleAutosave();
    },

    save: async () => {
      const { project } = get();
      if (!project) return;
      await persistence.saveProject(project);
      const projects = await persistence.listProjects();
      set({ dirty: false, lastSavedAt: new Date().toISOString(), projects });
    },

    removeProject: async (id) => {
      await persistence.deleteProject(id);
      const projects = await persistence.listProjects();
      const current = get().project;
      set({ projects, ...(current?.id === id ? { project: null, past: [], future: [] } : {}) });
    },

    renameProject: (name) => {
      const project = get().project;
      if (!project) return;
      touch({ ...project, name });
    },

    refreshProjects: async () => {
      set({ projects: await persistence.listProjects() });
    },

    setUi: (patch) => set({ ui: { ...get().ui, ...patch } }),
    selectComp: (id) => set({ ui: { ...get().ui, selectedCompId: id } }),
    toast: (message, kind = 'info') => {
      const id = ++toastCounter;
      set({ ui: { ...get().ui, toast: { id, message, kind } } });
      setTimeout(() => {
        if (get().ui.toast?.id === id) set({ ui: { ...get().ui, toast: null } });
      }, 4000);
    },

    mutateScenario: (label, fn, opts = {}) => {
      const project = get().project;
      if (!project) return;
      const idx = project.scenarios.findIndex((s) => s.id === project.activeScenarioId);
      if (idx < 0) return;
      const before = project.scenarios[idx];
      const next = clone(before);
      fn(next);
      let past = get().past;
      if (opts.undoable !== false) {
        const last = past[past.length - 1];
        const now = Date.now();
        const coalesce = opts.key && last && last.key === opts.key && last.scenarioId === before.id && now - last.at < COALESCE_MS;
        if (!coalesce) {
          past = [...past, { scenarioId: before.id, snapshot: before, label, key: opts.key, at: now }].slice(-HISTORY_LIMIT);
        } else {
          past = [...past.slice(0, -1), { ...last, at: now }];
        }
        set({ past, future: [] });
      }
      const scenarios = project.scenarios.slice();
      scenarios[idx] = next;
      touch({ ...project, scenarios });
    },

    setEffectiveDate: (date) => get().mutateScenario('Effective date', (s) => { s.effectiveDate = date; }, { key: 'effectiveDate' }),
    setSubjectAttr: (key, value) => get().mutateScenario(`Subject ${String(key)}`, (s) => { s.subject.attrs[key] = value; }, { key: `subject:${String(key)}` }),
    setSubjectAttrs: (patch) => get().mutateScenario('Subject', (s) => { Object.assign(s.subject.attrs, patch); }, { key: 'subject:multi' }),
    setSubjectNotes: (notes) => get().mutateScenario('Subject notes', (s) => { s.subject.notes = notes; }, { key: 'subject:notes' }),
    setSubjectPhotos: (urls) => get().mutateScenario('Subject photos', (s) => { s.subject.photoUrls = urls; }),
    setSubjectFromImport: (attrs, source) => get().mutateScenario('Import subject', (s) => {
      for (const [k, v] of Object.entries(attrs)) {
        if (v !== null && v !== undefined) (s.subject.attrs as unknown as Record<string, unknown>)[k] = v;
      }
      s.subject.sourceFile = source.file;
      s.subject.sourceRow = source.row;
      s.subject.parseIssues = source.issues;
    }),

    addComparables: (comps) => get().mutateScenario('Import comparables', (s) => { s.comparables.push(...comps); }),
    removeComparables: (ids) => get().mutateScenario('Remove comparables', (s) => { s.comparables = s.comparables.filter((c) => !ids.includes(c.id)); }),
    toggleIncluded: (id, included) => get().mutateScenario('Include / exclude', (s) => {
      const c = s.comparables.find((x) => x.id === id);
      if (c) c.included = included ?? !c.included;
    }),
    setCompWeight: (id, weight) => get().mutateScenario('Weight', (s) => {
      const c = s.comparables.find((x) => x.id === id);
      if (c) c.weight = Math.max(0, Number.isFinite(weight) ? weight : 0);
    }, { key: `weight:${id}` }),
    setCompNote: (id, note) => get().mutateScenario('Selection note', (s) => {
      const c = s.comparables.find((x) => x.id === id);
      if (c) c.selectionNote = note;
    }, { key: `note:${id}` }),
    setCompGridNote: (id, note) => get().mutateScenario('Grid note', (s) => {
      const c = s.comparables.find((x) => x.id === id);
      if (c) c.gridNote = note;
    }, { key: `gridnote:${id}` }),
    setCompAttrs: (id, patch) => get().mutateScenario('Edit comparable', (s) => {
      const c = s.comparables.find((x) => x.id === id);
      if (c) Object.assign(c.attrs, patch);
    }, { key: `attrs:${id}:${Object.keys(patch).join(',')}` }),
    setOverride: (compId, ruleId, override) => get().mutateScenario('Override', (s) => {
      const c = s.comparables.find((x) => x.id === compId);
      if (!c) return;
      if (override) {
        c.overrides[ruleId] = override;
        delete c.resolutions[ruleId];
      } else delete c.overrides[ruleId];
    }, { key: `override:${compId}:${ruleId}` }),
    setResolution: (compId, ruleId, resolution) => get().mutateScenario('Resolution', (s) => {
      const c = s.comparables.find((x) => x.id === compId);
      if (!c) return;
      if (resolution) {
        c.resolutions[ruleId] = resolution;
        delete c.overrides[ruleId];
      } else delete c.resolutions[ruleId];
    }),
    updateRule: (ruleId, patch) => get().mutateScenario('Adjustment rule', (s) => {
      const r = s.rules.find((x) => x.id === ruleId);
      if (r) Object.assign(r, patch);
    }, { key: `rule:${ruleId}:${Object.keys(patch).join(',')}` }),
    addCustomRule: (label, kind, field) => get().mutateScenario('Add custom category', (s) => {
      const slug = label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'custom';
      s.rules.push({
        id: makeId('rule'),
        category: `custom:${slug}`,
        label,
        kind,
        field,
        active: true,
        rate: null,
        categoricalValues: {},
        reason: '',
        source: '',
        attachmentRef: '',
        custom: true,
        order: s.rules.length,
      });
    }),
    removeRule: (ruleId) => get().mutateScenario('Remove category', (s) => {
      s.rules = s.rules.filter((r) => r.id !== ruleId);
      for (const c of s.comparables) {
        delete c.overrides[ruleId];
        delete c.resolutions[ruleId];
      }
    }),
    setSettings: (patch) => get().mutateScenario('Settings', (s) => { Object.assign(s.settings, patch); }, { key: 'settings' }),
    setFlags: (patch) => get().mutateScenario('Review flags', (s) => { Object.assign(s.settings.flags, patch); }, { key: 'flags' }),
    setFinalOpinion: (patch) => get().mutateScenario('Final opinion', (s) => { Object.assign(s.finalOpinion, patch); }, { key: 'finalOpinion' }),
    setReconciliationNotes: (notes) => get().mutateScenario('Reconciliation notes', (s) => { s.reconciliationNotes = notes; }, { key: 'reconNotes' }),

    addScenario: (name) => {
      const project = get().project;
      if (!project) return;
      const s = newScenario(name);
      touch({ ...project, scenarios: [...project.scenarios, s], activeScenarioId: s.id });
    },
    duplicateActiveScenario: (name) => {
      const project = get().project;
      const active = activeScenarioOf(project);
      if (!project || !active) return;
      const s = duplicateScenario(active, name);
      touch({ ...project, scenarios: [...project.scenarios, s], activeScenarioId: s.id });
    },
    deleteScenario: (id) => {
      const project = get().project;
      if (!project || project.scenarios.length <= 1) return;
      const scenarios = project.scenarios.filter((s) => s.id !== id);
      const activeScenarioId = project.activeScenarioId === id ? scenarios[0].id : project.activeScenarioId;
      set({ past: get().past.filter((h) => h.scenarioId !== id), future: get().future.filter((h) => h.scenarioId !== id) });
      touch({ ...project, scenarios, activeScenarioId });
    },
    renameScenario: (id, name) => {
      const project = get().project;
      if (!project) return;
      touch({ ...project, scenarios: project.scenarios.map((s) => (s.id === id ? { ...s, name } : s)) });
    },
    setActiveScenario: (id) => {
      const project = get().project;
      if (!project) return;
      touch({ ...project, activeScenarioId: id });
      set({ ui: { ...get().ui, selectedCompId: null } });
    },

    savePreset: async (preset) => {
      const presets = [...get().globalPresets.filter((p) => p.id !== preset.id), preset];
      await persistence.saveGlobalPresets(presets);
      set({ globalPresets: presets });
    },
    deletePreset: async (id) => {
      const presets = get().globalPresets.filter((p) => p.id !== id);
      await persistence.saveGlobalPresets(presets);
      set({ globalPresets: presets });
    },

    undo: () => {
      const { past, project } = get();
      if (!project || !past.length) return;
      const entry = past[past.length - 1];
      const idx = project.scenarios.findIndex((s) => s.id === entry.scenarioId);
      if (idx < 0) {
        set({ past: past.slice(0, -1) });
        return;
      }
      const current = project.scenarios[idx];
      const scenarios = project.scenarios.slice();
      scenarios[idx] = entry.snapshot;
      set({ past: past.slice(0, -1), future: [...get().future, { ...entry, snapshot: current }] });
      touch({ ...project, scenarios, activeScenarioId: entry.scenarioId });
    },
    redo: () => {
      const { future, project } = get();
      if (!project || !future.length) return;
      const entry = future[future.length - 1];
      const idx = project.scenarios.findIndex((s) => s.id === entry.scenarioId);
      if (idx < 0) {
        set({ future: future.slice(0, -1) });
        return;
      }
      const current = project.scenarios[idx];
      const scenarios = project.scenarios.slice();
      scenarios[idx] = entry.snapshot;
      set({ future: future.slice(0, -1), past: [...get().past, { ...entry, snapshot: current }] });
      touch({ ...project, scenarios, activeScenarioId: entry.scenarioId });
    },
  };
});

export const useActiveScenario = () => useAppStore((s) => activeScenarioOf(s.project));
