import type { Project, Scenario, ScenarioCalculation } from '../types';
import { ATTR_LABELS } from './fields';
import { fmtBool, fmtLot, roundTo } from './format';
import { formatDate } from './dates';
import { defaultRules, defaultSettings, emptySubject, SCHEMA_VERSION } from './defaults';
import { emptyAttributes } from './fields';
import { orderedRules } from './calc';

/**
 * Protect spreadsheet consumers from formula injection: any text cell beginning with
 * =, +, -, @, tab or carriage return is prefixed with a single quote. Numeric cells are
 * emitted as plain numbers so legitimate negative values survive untouched.
 */
export function csvCell(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : '';
  let s = String(value);
  if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`;
  if (/[",\n\r]/.test(s)) s = `"${s.replace(/"/g, '""')}"`;
  return s;
}

export function toCsv(rows: (string | number | null | undefined)[][]): string {
  return rows.map((r) => r.map(csvCell).join(',')).join('\r\n');
}

function num(v: number | null, rounding: number): number | null {
  if (v === null) return null;
  return roundTo(v, rounding);
}

/** Build the adjustment-grid CSV. Values match the on-screen grid, including the rounding setting. */
export function buildGridCsv(scenario: Scenario, calc: ScenarioCalculation): string {
  const comps = scenario.comparables.filter((c) => c.included);
  const rounding = scenario.settings.rounding;
  const s = scenario.subject.attrs;
  const rows: (string | number | null)[][] = [];
  const header = ['Item', 'Subject', ...comps.map((_, i) => `Comp ${i + 1}`)];
  rows.push(header);
  rows.push(['Address', s.address, ...comps.map((c) => c.attrs.address)]);
  rows.push(['MLS number', s.mlsNumber, ...comps.map((c) => c.attrs.mlsNumber)]);
  rows.push(['Status', null, ...comps.map((c) => c.attrs.statusRaw ?? c.attrs.status)]);
  rows.push(['Sale date', null, ...comps.map((c) => formatDate(c.attrs.saleDate))]);
  rows.push(['Original sale price', null, ...comps.map((c) => c.attrs.salePrice)]);
  rows.push(['Price per SF (above-grade)', null, ...comps.map((c) => (calc.comps[c.id].pricePerSqft !== null ? Number((calc.comps[c.id].pricePerSqft as number).toFixed(2)) : null))]);
  const attrRows: (keyof typeof ATTR_LABELS)[] = ['areaAboveGrade', 'areaBelowGradeFinished', 'areaOther', 'bedrooms', 'fullBaths', 'halfBaths', 'yearBuilt', 'garageSpaces', 'carportSpaces', 'quality', 'condition', 'location', 'concessions'];
  rows.push(['Lot size', fmtLot(s.lotSizeSqft, s.lotSizeRaw, s.lotSizeUnits), ...comps.map((c) => fmtLot(c.attrs.lotSizeSqft, c.attrs.lotSizeRaw, c.attrs.lotSizeUnits))]);
  for (const k of attrRows) {
    rows.push([ATTR_LABELS[k], s[k] as string | number | null, ...comps.map((c) => c.attrs[k] as string | number | null)]);
  }
  for (const k of ['pool', 'waterfront', 'waterView'] as const) {
    rows.push([ATTR_LABELS[k], fmtBool(s[k]), ...comps.map((c) => fmtBool(c.attrs[k]))]);
  }
  for (const rule of orderedRules(scenario.rules)) {
    if (!rule.active) continue;
    rows.push([
      `Adj: ${rule.label}`,
      null,
      ...comps.map((c) => {
        const line = calc.comps[c.id].lines.find((l) => l.ruleId === rule.id);
        if (!line) return null;
        if (line.status === 'unresolved') return 'UNRESOLVED';
        if (line.status === 'notApplicable') return 'N/A';
        return num(line.amount, rounding);
      }),
    ]);
    rows.push([`  source: ${rule.label}`, null, ...comps.map((c) => calc.comps[c.id].lines.find((l) => l.ruleId === rule.id)?.source ?? null)]);
  }
  rows.push(['Net adjustment', null, ...comps.map((c) => (calc.comps[c.id].unresolved.length ? 'UNRESOLVED' : num(calc.comps[c.id].netAdjustment, rounding)))]);
  rows.push(['Gross adjustment', null, ...comps.map((c) => (calc.comps[c.id].unresolved.length ? 'UNRESOLVED' : num(calc.comps[c.id].grossAdjustment, rounding)))]);
  rows.push(['Net adjustment %', null, ...comps.map((c) => (calc.comps[c.id].netPct !== null ? Number((calc.comps[c.id].netPct as number).toFixed(2)) : null))]);
  rows.push(['Gross adjustment %', null, ...comps.map((c) => (calc.comps[c.id].grossPct !== null ? Number((calc.comps[c.id].grossPct as number).toFixed(2)) : null))]);
  rows.push(['Adjusted sale price', null, ...comps.map((c) => (calc.comps[c.id].adjustedPrice !== null ? num(calc.comps[c.id].adjustedPrice, rounding) : 'UNRESOLVED'))]);
  rows.push(['Weight (entered)', null, ...comps.map((c) => c.weight)]);
  rows.push(['Weight (normalized %)', null, ...comps.map((c) => (calc.comps[c.id].normalizedWeight !== null ? Number(((calc.comps[c.id].normalizedWeight as number) * 100).toFixed(2)) : null))]);
  rows.push(['Notes', scenario.subject.notes, ...comps.map((c) => c.gridNote || c.selectionNote)]);
  rows.push([]);
  const r = calc.reconciliation;
  rows.push(['Reconciliation']);
  rows.push(['Included closed sales', r.includedCount]);
  rows.push(['Contributing to indication', r.contributingCount]);
  rows.push(['Excluded (unresolved / missing)', r.excludedForUnresolved]);
  rows.push(['Adjusted range low', num(r.adjustedRange?.[0] ?? null, rounding)]);
  rows.push(['Adjusted range high', num(r.adjustedRange?.[1] ?? null, rounding)]);
  rows.push(['Adjusted mean', num(r.adjustedMean, rounding)]);
  rows.push(['Adjusted median', num(r.adjustedMedian, rounding)]);
  rows.push(['Weighted indication', num(r.weightedIndication, rounding)]);
  rows.push(['Final reconciled opinion', scenario.finalOpinion.value]);
  rows.push(['Rounding increment', rounding]);
  if (r.message) rows.push(['Note', r.message]);
  return toCsv(rows);
}

export function exportProjectJson(project: Project): string {
  return JSON.stringify({ ...project, schemaVersion: SCHEMA_VERSION, exportedAt: new Date().toISOString() }, null, 2);
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/** Validate and normalise a project backup. Missing optional pieces are filled with defaults; bad shapes throw. */
export function importProjectJson(text: string): Project {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error('File is not valid JSON.');
  }
  if (!isRecord(data) || !Array.isArray(data.scenarios) || typeof data.name !== 'string') {
    throw new Error('File does not look like an appraisal project backup (missing name or scenarios).');
  }
  const scenarios: Scenario[] = data.scenarios.map((raw, i) => {
    if (!isRecord(raw)) throw new Error(`Scenario ${i + 1} is malformed.`);
    const subjectRaw = isRecord(raw.subject) ? raw.subject : {};
    const subject = emptySubject();
    subject.attrs = { ...emptyAttributes(), ...(isRecord(subjectRaw.attrs) ? subjectRaw.attrs : {}) } as typeof subject.attrs;
    subject.photoUrls = Array.isArray(subjectRaw.photoUrls) ? subjectRaw.photoUrls.filter((u) => typeof u === 'string') : [];
    subject.notes = typeof subjectRaw.notes === 'string' ? subjectRaw.notes : '';
    const comparables = Array.isArray(raw.comparables) ? raw.comparables : [];
    for (const c of comparables) {
      if (!isRecord(c) || typeof c.id !== 'string' || !isRecord(c.attrs)) throw new Error(`Scenario ${i + 1} contains a malformed comparable.`);
      c.attrs = { ...emptyAttributes(), ...c.attrs };
      c.overrides = isRecord(c.overrides) ? c.overrides : {};
      c.resolutions = isRecord(c.resolutions) ? c.resolutions : {};
      c.raw = isRecord(c.raw) ? c.raw : {};
      c.parseIssues = Array.isArray(c.parseIssues) ? c.parseIssues : [];
      c.weight = typeof c.weight === 'number' && c.weight >= 0 ? c.weight : 1;
      c.included = !!c.included;
      c.selectionNote = typeof c.selectionNote === 'string' ? c.selectionNote : '';
      c.gridNote = typeof c.gridNote === 'string' ? c.gridNote : '';
    }
    const settings = defaultSettings();
    if (isRecord(raw.settings)) {
      if (typeof raw.settings.rounding === 'number') settings.rounding = raw.settings.rounding;
      if (isRecord(raw.settings.flags)) settings.flags = { ...settings.flags, ...raw.settings.flags };
    }
    const finalOpinion = isRecord(raw.finalOpinion) ? { value: typeof raw.finalOpinion.value === 'number' ? raw.finalOpinion.value : null, rationale: typeof raw.finalOpinion.rationale === 'string' ? raw.finalOpinion.rationale : '' } : { value: null, rationale: '' };
    return {
      id: typeof raw.id === 'string' ? raw.id : `scn_import_${i}`,
      name: typeof raw.name === 'string' ? raw.name : `Scenario ${i + 1}`,
      createdAt: typeof raw.createdAt === 'string' ? raw.createdAt : new Date().toISOString(),
      effectiveDate: typeof raw.effectiveDate === 'string' ? raw.effectiveDate : null,
      subject,
      comparables: comparables as Scenario['comparables'],
      rules: Array.isArray(raw.rules) && raw.rules.length ? (raw.rules as Scenario['rules']) : defaultRules(),
      settings,
      finalOpinion,
      reconciliationNotes: typeof raw.reconciliationNotes === 'string' ? raw.reconciliationNotes : '',
    };
  });
  if (!scenarios.length) throw new Error('Backup contains no scenarios.');
  const activeScenarioId = typeof data.activeScenarioId === 'string' && scenarios.some((s) => s.id === data.activeScenarioId) ? data.activeScenarioId : scenarios[0].id;
  return {
    id: typeof data.id === 'string' ? data.id : `proj_import_${Date.now()}`,
    name: data.name,
    createdAt: typeof data.createdAt === 'string' ? data.createdAt : new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    scenarios,
    activeScenarioId,
    mappingPresets: Array.isArray(data.mappingPresets) ? (data.mappingPresets as Project['mappingPresets']) : [],
    isDemo: !!data.isDemo,
    schemaVersion: SCHEMA_VERSION,
  };
}

export function downloadText(filename: string, text: string, mime = 'text/plain') {
  const blob = new Blob([text], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
