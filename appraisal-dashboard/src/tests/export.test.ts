import { describe, expect, it } from 'vitest';
import { buildGridCsv, csvCell, exportProjectJson, importProjectJson, toCsv } from '../engine/export';
import { computeScenario } from '../engine/calc';
import { buildDemoProject } from '../engine/demo';
import { newProject } from '../engine/defaults';
import { comp, scenarioWith } from './helpers';
import * as persistence from '../store/persistence';

describe('CSV export safety', () => {
  it('neutralises formula injection while keeping numbers', () => {
    expect(csvCell('=SUM(A1)')).toBe("'=SUM(A1)");
    expect(csvCell('+1234')).toBe("'+1234");
    expect(csvCell('-foo')).toBe("'-foo");
    expect(csvCell('@cmd')).toBe("'@cmd");
    expect(csvCell(-2500)).toBe('-2500');
    expect(csvCell(310000)).toBe('310000');
    expect(csvCell('plain, text')).toBe('"plain, text"');
    expect(csvCell('say "hi"')).toBe('"say ""hi"""');
    expect(csvCell(null)).toBe('');
    expect(toCsv([[1, 'a'], [null, '=x']])).toBe("1,a\r\n,'=x");
  });

  it('grid CSV matches the visible calculation and marks unresolved values', () => {
    const a = comp({ address: '=HYPERLINK("x")', areaAboveGrade: 1800, salePrice: 300000 }, { id: 'A', weight: 60 });
    const b = comp({ address: 'B', areaAboveGrade: null, salePrice: 330000 }, { id: 'B', weight: 40 });
    const s = scenarioWith({ areaAboveGrade: 2000 }, [a, b], (rules) => {
      const r = rules.find((x) => x.category === 'areaAboveGrade')!;
      r.active = true;
      r.rate = 50;
    });
    const calc = computeScenario(s);
    const csv = buildGridCsv(s, calc);
    expect(csv).toContain("'=HYPERLINK");
    const adjRow = csv.split('\r\n').find((l) => l.startsWith('Adjusted sale price'))!;
    expect(adjRow).toBe('Adjusted sale price,,310000,UNRESOLVED');
    const wi = csv.split('\r\n').find((l) => l.startsWith('Weighted indication'))!;
    expect(wi).toBe('Weighted indication,310000');
  });
});

describe('project backup and restore', () => {
  it('round-trips a project through JSON', () => {
    const p = buildDemoProject();
    const json = exportProjectJson(p);
    const restored = importProjectJson(json);
    expect(restored.name).toBe(p.name);
    expect(restored.scenarios).toHaveLength(1);
    expect(restored.scenarios[0].comparables).toHaveLength(8);
    expect(computeScenario(restored.scenarios[0]).reconciliation.weightedIndication).toBe(computeScenario(p.scenarios[0]).reconciliation.weightedIndication);
  });
  it('rejects malformed backups', () => {
    expect(() => importProjectJson('not json')).toThrow(/JSON/);
    expect(() => importProjectJson('{"foo":1}')).toThrow(/backup/);
    expect(() => importProjectJson('{"name":"x","scenarios":[]}')).toThrow(/no scenarios/);
  });
  it('persists scenarios to IndexedDB and restores them', async () => {
    const p = newProject('Persisted');
    p.scenarios[0].comparables.push(comp({ salePrice: 300000 }, { id: 'A' }));
    p.scenarios[0].finalOpinion = { value: 305000, rationale: 'test' };
    await persistence.saveProject(p);
    const list = await persistence.listProjects();
    expect(list.some((x) => x.id === p.id && x.name === 'Persisted')).toBe(true);
    const loaded = await persistence.loadProject(p.id);
    expect(loaded?.scenarios[0].comparables[0].attrs.salePrice).toBe(300000);
    expect(loaded?.scenarios[0].finalOpinion.value).toBe(305000);
    await persistence.deleteProject(p.id);
    expect(await persistence.loadProject(p.id)).toBeUndefined();
  });
});
