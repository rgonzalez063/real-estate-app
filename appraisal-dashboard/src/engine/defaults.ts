import type { AdjustmentRule, Project, Scenario, ScenarioSettings, SubjectProperty } from '../types';
import { emptyAttributes } from './fields';
import { makeId } from './ids';
import { todayIso } from './dates';

export const SCHEMA_VERSION = 1;

/** Built-in categories. All start inactive with blank rates; the appraiser supplies supported rates. */
export function defaultRules(): AdjustmentRule[] {
  const base = (partial: Partial<AdjustmentRule> & Pick<AdjustmentRule, 'category' | 'label' | 'kind' | 'field'>, order: number): AdjustmentRule => ({
    id: makeId('rule'),
    active: false,
    rate: null,
    categoricalValues: {},
    reason: '',
    source: '',
    attachmentRef: '',
    order,
    ...partial,
  });
  return [
    base({ category: 'concessions', label: 'Concessions', kind: 'concessions', field: 'concessions', concessionFactor: null }, 0),
    base({ category: 'marketConditions', label: 'Market conditions (time)', kind: 'market', field: null, market: { mode: 'monthlyRate', monthlyRatePct: null, dateBasis: 'saleDate' } }, 1),
    base({ category: 'location', label: 'Location', kind: 'categorical', field: 'location' }, 2),
    base({ category: 'tenure', label: 'Leasehold / fee simple', kind: 'manual', field: 'tenure' }, 3),
    base({ category: 'lotSize', label: 'Site', kind: 'perUnit', field: 'lotSizeSqft' }, 4),
    base({ category: 'view', label: 'View', kind: 'categorical', field: 'view' }, 5),
    base({ category: 'waterfront', label: 'Waterfront', kind: 'flat', field: 'waterfront' }, 6),
    base({ category: 'design', label: 'Design (style)', kind: 'manual', field: 'design' }, 7),
    base({ category: 'quality', label: 'Quality of construction', kind: 'categorical', field: 'quality' }, 8),
    base({ category: 'age', label: 'Actual age (via year built)', kind: 'perUnit', field: 'yearBuilt' }, 9),
    base({ category: 'condition', label: 'Condition', kind: 'categorical', field: 'condition' }, 10),
    base({ category: 'bedrooms', label: 'Bedrooms', kind: 'perUnit', field: 'bedrooms' }, 11),
    base({ category: 'fullBaths', label: 'Full baths', kind: 'perUnit', field: 'fullBaths' }, 12),
    base({ category: 'halfBaths', label: 'Half baths', kind: 'perUnit', field: 'halfBaths' }, 13),
    base({ category: 'areaAboveGrade', label: 'Gross living area', kind: 'perUnit', field: 'areaAboveGrade' }, 14),
    base({ category: 'areaBelowGradeFinished', label: 'Basement & finished rooms below grade', kind: 'perUnit', field: 'areaBelowGradeFinished' }, 15),
    base({ category: 'areaOther', label: 'Other area', kind: 'perUnit', field: 'areaOther' }, 16),
    base({ category: 'functionalUtility', label: 'Functional utility', kind: 'manual', field: 'functionalUtility' }, 17),
    base({ category: 'heatingCooling', label: 'Heating / cooling', kind: 'manual', field: 'heatingCooling' }, 18),
    base({ category: 'energyEfficient', label: 'Energy efficient items', kind: 'manual', field: 'energyEfficient' }, 19),
    base({ category: 'garageSpaces', label: 'Garage spaces', kind: 'perUnit', field: 'garageSpaces' }, 20),
    base({ category: 'carportSpaces', label: 'Carport spaces', kind: 'perUnit', field: 'carportSpaces' }, 21),
    base({ category: 'porchPatioDeck', label: 'Porch / patio / deck', kind: 'manual', field: 'porchPatioDeck' }, 22),
    base({ category: 'pool', label: 'Pool', kind: 'flat', field: 'pool' }, 23),
    base({ category: 'otherAmenities', label: 'Other amenities / fence', kind: 'manual', field: 'amenities' }, 24),
  ];
}

export function defaultSettings(): ScenarioSettings {
  return {
    rounding: 1,
    flags: { netPct: 15, grossPct: 25, singleLinePct: 10, outlierPct: 15 },
  };
}

export function emptySubject(): SubjectProperty {
  return { attrs: emptyAttributes(), photoUrls: [], notes: '' };
}

export function newScenario(name = 'Scenario 1'): Scenario {
  return {
    id: makeId('scn'),
    name,
    createdAt: new Date().toISOString(),
    effectiveDate: todayIso(),
    subject: emptySubject(),
    comparables: [],
    rules: defaultRules(),
    settings: defaultSettings(),
    finalOpinion: { value: null, rationale: '' },
    reconciliationNotes: '',
  };
}

export function newProject(name = 'Untitled appraisal'): Project {
  const scenario = newScenario();
  const now = new Date().toISOString();
  return {
    id: makeId('proj'),
    name,
    createdAt: now,
    updatedAt: now,
    scenarios: [scenario],
    activeScenarioId: scenario.id,
    mappingPresets: [],
    schemaVersion: SCHEMA_VERSION,
  };
}

/** Deep clone a scenario with new ids so it can be edited independently. */
export function duplicateScenario(s: Scenario, name: string): Scenario {
  const copy: Scenario = JSON.parse(JSON.stringify(s));
  copy.id = makeId('scn');
  copy.name = name;
  copy.createdAt = new Date().toISOString();
  return copy;
}
