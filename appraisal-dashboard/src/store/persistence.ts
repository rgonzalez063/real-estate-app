import { del, get, keys, set } from 'idb-keyval';
import type { MappingPreset, Project } from '../types';

const PROJECT_PREFIX = 'project:';
const PRESETS_KEY = 'mappingPresets:global';

export interface ProjectSummary {
  id: string;
  name: string;
  updatedAt: string;
  isDemo: boolean;
  scenarioCount: number;
}

export async function saveProject(project: Project): Promise<void> {
  await set(PROJECT_PREFIX + project.id, project);
}

export async function loadProject(id: string): Promise<Project | undefined> {
  return get<Project>(PROJECT_PREFIX + id);
}

export async function deleteProject(id: string): Promise<void> {
  await del(PROJECT_PREFIX + id);
}

export async function listProjects(): Promise<ProjectSummary[]> {
  const all = await keys();
  const out: ProjectSummary[] = [];
  for (const k of all) {
    if (typeof k === 'string' && k.startsWith(PROJECT_PREFIX)) {
      const p = await get<Project>(k);
      if (p) out.push({ id: p.id, name: p.name, updatedAt: p.updatedAt, isDemo: !!p.isDemo, scenarioCount: p.scenarios.length });
    }
  }
  return out.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}

export async function loadGlobalPresets(): Promise<MappingPreset[]> {
  return (await get<MappingPreset[]>(PRESETS_KEY)) ?? [];
}

export async function saveGlobalPresets(presets: MappingPreset[]): Promise<void> {
  await set(PRESETS_KEY, presets);
}
