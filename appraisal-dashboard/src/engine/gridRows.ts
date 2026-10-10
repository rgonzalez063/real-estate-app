import type { AdjustmentRule, PropertyAttributes } from '../types';

export type DescInput = 'text' | 'number' | 'currency' | 'date' | 'boolean' | 'lot';

export interface GridRow {
  key: string;
  label: string;
  /** Attribute shown and edited in the description cells (null for computed rows). */
  field: keyof PropertyAttributes | null;
  /** Rule category that supplies the adjustment cell (null = no adjustment on this row). */
  category: string | null;
  input?: DescInput;
  /** Computed, read-only description (price per SF, data source). */
  computed?: 'pricePerGla' | 'dataSource';
  /** Subject cell is intentionally blank (transaction data). */
  subjectBlank?: boolean;
  section?: boolean;
}

/**
 * Sales comparison grid rows in the familiar appraisal-form order. Each row pairs a description
 * (the property fact) with the adjustment category that handles the difference.
 */
export const FORM_ROWS: GridRow[] = [
  { key: 'salePrice', label: 'Sale price', field: 'salePrice', category: null, input: 'currency', subjectBlank: true },
  { key: 'pricePerGla', label: 'Sale price / gross liv. area', field: null, category: null, computed: 'pricePerGla', subjectBlank: true },
  { key: 'dataSource', label: 'Data source(s)', field: null, category: null, computed: 'dataSource' },
  { key: 'sec-adj', label: 'Value adjustments', field: null, category: null, section: true },
  { key: 'concessions', label: 'Sales or financing concessions', field: 'concessions', category: 'concessions', input: 'currency', subjectBlank: true },
  { key: 'marketConditions', label: 'Date of sale / time', field: 'saleDate', category: 'marketConditions', input: 'date' },
  { key: 'location', label: 'Location', field: 'location', category: 'location', input: 'text' },
  { key: 'tenure', label: 'Leasehold / fee simple', field: 'tenure', category: 'tenure', input: 'text' },
  { key: 'site', label: 'Site', field: 'lotSizeRaw', category: 'lotSize', input: 'lot' },
  { key: 'view', label: 'View', field: 'view', category: 'view', input: 'text' },
  { key: 'waterfront', label: 'Waterfront', field: 'waterfront', category: 'waterfront', input: 'boolean' },
  { key: 'design', label: 'Design (style)', field: 'design', category: 'design', input: 'text' },
  { key: 'quality', label: 'Quality of construction', field: 'quality', category: 'quality', input: 'text' },
  { key: 'age', label: 'Actual age (year built)', field: 'yearBuilt', category: 'age', input: 'number' },
  { key: 'condition', label: 'Condition', field: 'condition', category: 'condition', input: 'text' },
  { key: 'bedrooms', label: 'Above grade: bedrooms', field: 'bedrooms', category: 'bedrooms', input: 'number' },
  { key: 'fullBaths', label: 'Above grade: full baths', field: 'fullBaths', category: 'fullBaths', input: 'number' },
  { key: 'halfBaths', label: 'Above grade: half baths', field: 'halfBaths', category: 'halfBaths', input: 'number' },
  { key: 'areaAboveGrade', label: 'Gross living area (SF)', field: 'areaAboveGrade', category: 'areaAboveGrade', input: 'number' },
  { key: 'areaBelowGradeFinished', label: 'Basement & finished rooms below grade (SF)', field: 'areaBelowGradeFinished', category: 'areaBelowGradeFinished', input: 'number' },
  { key: 'areaOther', label: 'Other area (SF)', field: 'areaOther', category: 'areaOther', input: 'number' },
  { key: 'functionalUtility', label: 'Functional utility', field: 'functionalUtility', category: 'functionalUtility', input: 'text' },
  { key: 'heatingCooling', label: 'Heating / cooling', field: 'heatingCooling', category: 'heatingCooling', input: 'text' },
  { key: 'energyEfficient', label: 'Energy efficient items', field: 'energyEfficient', category: 'energyEfficient', input: 'text' },
  { key: 'garageSpaces', label: 'Garage spaces', field: 'garageSpaces', category: 'garageSpaces', input: 'number' },
  { key: 'carportSpaces', label: 'Carport spaces', field: 'carportSpaces', category: 'carportSpaces', input: 'number' },
  { key: 'porchPatioDeck', label: 'Porch / patio / deck', field: 'porchPatioDeck', category: 'porchPatioDeck', input: 'text' },
  { key: 'pool', label: 'Pool', field: 'pool', category: 'pool', input: 'boolean' },
  { key: 'otherAmenities', label: 'Other amenities / fence', field: 'amenities', category: 'otherAmenities', input: 'text' },
];

const NUMERIC_FIELDS = new Set<keyof PropertyAttributes>(['areaAboveGrade', 'areaBelowGradeFinished', 'areaBelowGradeUnfinished', 'areaOther', 'bedrooms', 'fullBaths', 'halfBaths', 'yearBuilt', 'age', 'garageSpaces', 'carportSpaces', 'daysOnMarket', 'latitude', 'longitude', 'distance']);
const BOOL_FIELDS = new Set<keyof PropertyAttributes>(['pool', 'waterfront', 'waterView']);

/** Rows for custom (user-added) rules not covered by the form rows. */
export function customRows(rules: AdjustmentRule[]): GridRow[] {
  const covered = new Set(FORM_ROWS.map((r) => r.category).filter(Boolean));
  return rules
    .filter((r) => !covered.has(r.category))
    .sort((a, b) => a.order - b.order)
    .map((r) => ({
      key: `rule:${r.id}`,
      label: r.label,
      field: r.field,
      category: r.category,
      input: r.field ? (NUMERIC_FIELDS.has(r.field) ? 'number' : BOOL_FIELDS.has(r.field) ? 'boolean' : r.field === 'lotSizeSqft' || r.field === 'lotSizeRaw' ? 'lot' : r.field === 'salePrice' || r.field === 'concessions' || r.field === 'listPrice' ? 'currency' : 'text') : undefined,
    }));
}
