import type { Comparable, Project, PropertyAttributes } from '../types';
import { emptyAttributes } from './fields';
import { newProject } from './defaults';

/**
 * SYNTHETIC DEMO DATA. Every record here is invented for preview purposes and is
 * flagged `synthetic`. It is never mixed into user-imported projects.
 */
function demoComp(
  i: number,
  partial: Partial<PropertyAttributes>,
): Comparable {
  const attrs: PropertyAttributes = {
    ...emptyAttributes(),
    mlsNumber: `DEMO-${String(1000 + i)}`,
    city: 'Demo City',
    zip: '00000',
    neighborhood: 'Demo Estates',
    status: 'closed',
    statusRaw: 'Closed',
    quality: 'Q3',
    condition: 'C3',
    location: 'Neutral',
    pool: false,
    waterfront: false,
    waterView: false,
    garageSpaces: 2,
    carportSpaces: 0,
    bedrooms: 3,
    fullBaths: 2,
    halfBaths: 0,
    lotSizeUnits: 'sqft',
    ...partial,
  };
  if (attrs.lotSizeRaw !== null && attrs.lotSizeSqft === null) attrs.lotSizeSqft = attrs.lotSizeUnits === 'acres' ? attrs.lotSizeRaw * 43560 : attrs.lotSizeRaw;
  const raw: Record<string, string> = {
    'MLS #': attrs.mlsNumber ?? '',
    Address: attrs.address ?? '',
    Status: attrs.statusRaw ?? '',
    'Close Price': attrs.salePrice !== null ? `$${attrs.salePrice.toLocaleString()}` : '',
    'Close Date': attrs.saleDate ?? '',
    GLA: attrs.areaAboveGrade !== null ? String(attrs.areaAboveGrade) : '',
  };
  return {
    id: `demo_comp_${i}`,
    attrs,
    raw,
    sourceFile: 'synthetic-demo.csv (not a real file)',
    sourceRowIndex: i,
    parseIssues: [],
    included: i <= 4 && attrs.status === 'closed',
    selectionNote: '',
    weight: 1,
    overrides: {},
    resolutions: {},
    gridNote: '',
    synthetic: true,
  };
}

export function buildDemoProject(): Project {
  const project = newProject('DEMO — Synthetic sample appraisal');
  project.isDemo = true;
  const s = project.scenarios[0];
  s.name = 'Baseline';
  s.effectiveDate = '2026-09-30';
  s.subject.attrs = {
    ...emptyAttributes(),
    address: '100 Example Way (synthetic subject)',
    city: 'Demo City',
    zip: '00000',
    neighborhood: 'Demo Estates',
    areaAboveGrade: 2000,
    areaBelowGradeFinished: 0,
    lotSizeRaw: 9000,
    lotSizeUnits: 'sqft',
    lotSizeSqft: 9000,
    bedrooms: 3,
    fullBaths: 2,
    halfBaths: 1,
    yearBuilt: 2008,
    garageSpaces: 2,
    carportSpaces: 0,
    pool: false,
    waterfront: false,
    waterView: false,
    quality: 'Q3',
    condition: 'C3',
    location: 'Neutral',
    latitude: 29.7,
    longitude: -95.4,
  };
  s.subject.notes = 'Synthetic subject for demonstration. Replace with real appraisal data by importing a CSV.';
  s.comparables = [
    demoComp(1, { address: '102 Example Way', salePrice: 300000, saleDate: '2026-07-15', contractDate: '2026-06-20', areaAboveGrade: 1800, lotSizeRaw: 8500, yearBuilt: 2006, halfBaths: 1, daysOnMarket: 21, concessions: 0, latitude: 29.701, longitude: -95.402, distance: 0.1 }),
    demoComp(2, { address: '215 Sample Court', salePrice: 330000, saleDate: '2026-05-02', contractDate: '2026-04-10', areaAboveGrade: 2150, lotSizeRaw: 9500, yearBuilt: 2010, halfBaths: 1, daysOnMarket: 14, concessions: 5000, latitude: 29.704, longitude: -95.398, distance: 0.4 }),
    demoComp(3, { address: '48 Placeholder Drive', salePrice: 318500, saleDate: '2026-08-21', contractDate: '2026-07-28', areaAboveGrade: 2040, lotSizeRaw: 0.21, lotSizeUnits: 'acres', yearBuilt: 2005, condition: 'C4', halfBaths: 0, daysOnMarket: 33, concessions: 0, latitude: 29.695, longitude: -95.41, distance: 0.7 }),
    demoComp(4, { address: '77 Mock Lane', salePrice: 352000, saleDate: '2026-03-12', contractDate: '2026-02-18', areaAboveGrade: 2300, lotSizeRaw: 10200, yearBuilt: 2012, pool: true, halfBaths: 1, daysOnMarket: 9, concessions: 2500, latitude: 29.71, longitude: -95.395, distance: 0.9 }),
    demoComp(5, { address: '310 Fixture Street', salePrice: 289000, saleDate: '2026-01-30', contractDate: '2026-01-05', areaAboveGrade: 1750, lotSizeRaw: 7800, yearBuilt: 2001, garageSpaces: 1, condition: 'C4', daysOnMarket: 48, concessions: null, latitude: 29.69, longitude: -95.42, distance: 1.3 }),
    demoComp(6, { address: '9 Specimen Circle', salePrice: 341000, saleDate: '2026-06-08', contractDate: '2026-05-15', areaAboveGrade: 2110, lotSizeRaw: 9100, yearBuilt: 2009, quality: 'Q2', halfBaths: 1, daysOnMarket: 17, concessions: 0, latitude: 29.706, longitude: -95.405, distance: 0.5 }),
    demoComp(7, { address: '54 Listing Avenue', status: 'active', statusRaw: 'Active', salePrice: null, listPrice: 345000, saleDate: null, areaAboveGrade: 2050, lotSizeRaw: 9000, yearBuilt: 2008, halfBaths: 1, daysOnMarket: 12, latitude: 29.703, longitude: -95.401, distance: 0.3 }),
    demoComp(8, { address: '61 Pending Place', status: 'pending', statusRaw: 'Pending', salePrice: null, listPrice: 329900, saleDate: null, contractDate: '2026-09-18', areaAboveGrade: 1980, lotSizeRaw: 8800, yearBuilt: 2007, halfBaths: 1, daysOnMarket: 20, latitude: 29.698, longitude: -95.399, distance: 0.4 }),
  ];
  return project;
}
