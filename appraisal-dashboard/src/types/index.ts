/**
 * Core domain types for the appraisal dashboard.
 *
 * Conventions:
 *  - `null` always means "unknown / not provided". It is never coerced to 0.
 *  - Dates are stored as ISO calendar strings (YYYY-MM-DD) with no time component.
 *  - Money and area values are stored at full precision; rounding happens only in display/export.
 */

export type StatusCategory = 'closed' | 'active' | 'pending' | 'other';
export type LotUnits = 'sqft' | 'acres';

/** Canonical field keys recognised by the import mapper and the calculation engine. */
export type FieldKey =
  | 'mlsNumber'
  | 'address'
  | 'city'
  | 'zip'
  | 'neighborhood'
  | 'status'
  | 'salePrice'
  | 'listPrice'
  | 'saleDate'
  | 'contractDate'
  | 'daysOnMarket'
  | 'areaAboveGrade'
  | 'areaBelowGradeFinished'
  | 'areaBelowGradeUnfinished'
  | 'areaOther'
  | 'lotSize'
  | 'lotSizeUnits'
  | 'bedrooms'
  | 'fullBaths'
  | 'halfBaths'
  | 'bathsCombined'
  | 'yearBuilt'
  | 'age'
  | 'garageSpaces'
  | 'carportSpaces'
  | 'pool'
  | 'quality'
  | 'condition'
  | 'renovation'
  | 'remarks'
  | 'waterfront'
  | 'waterView'
  | 'location'
  | 'amenities'
  | 'concessions'
  | 'latitude'
  | 'longitude'
  | 'distance'
  | 'photoUrl';

export type FieldType = 'text' | 'number' | 'currency' | 'date' | 'boolean' | 'status' | 'lotUnits' | 'url';

export interface FieldDef {
  key: FieldKey;
  label: string;
  type: FieldType;
  group: 'identity' | 'transaction' | 'area' | 'site' | 'rooms' | 'age' | 'parking' | 'ratings' | 'location' | 'other';
  /** Lower-case header aliases commonly seen in MLS exports (HARMLS / Matrix etc.). */
  aliases: string[];
  /** Unit for display (e.g. "SF"). */
  unit?: string;
  /** Whether this is a quantitative attribute usable in per-unit adjustments. */
  quantitative?: boolean;
  /** Field that is derived, not directly stored (e.g. bathsCombined splits into full/half). */
  virtual?: boolean;
}

/** Normalised property attributes shared by subject and comparables. */
export interface PropertyAttributes {
  mlsNumber: string | null;
  address: string | null;
  city: string | null;
  zip: string | null;
  neighborhood: string | null;
  status: StatusCategory | null;
  statusRaw: string | null;
  salePrice: number | null;
  listPrice: number | null;
  saleDate: string | null;
  contractDate: string | null;
  daysOnMarket: number | null;
  areaAboveGrade: number | null;
  areaBelowGradeFinished: number | null;
  areaBelowGradeUnfinished: number | null;
  areaOther: number | null;
  /** Lot size normalised to square feet. */
  lotSizeSqft: number | null;
  /** Lot size as entered, with its units, for display. */
  lotSizeRaw: number | null;
  lotSizeUnits: LotUnits | null;
  bedrooms: number | null;
  fullBaths: number | null;
  halfBaths: number | null;
  yearBuilt: number | null;
  age: number | null;
  garageSpaces: number | null;
  carportSpaces: number | null;
  pool: boolean | null;
  quality: string | null;
  condition: string | null;
  renovation: string | null;
  remarks: string | null;
  waterfront: boolean | null;
  waterView: boolean | null;
  location: string | null;
  amenities: string | null;
  concessions: number | null;
  latitude: number | null;
  longitude: number | null;
  distance: number | null;
  photoUrl: string | null;
}

export interface ParseIssue {
  field: FieldKey | string;
  raw: string;
  message: string;
}

export interface Comparable {
  id: string;
  attrs: PropertyAttributes;
  /** The untouched source row (header -> cell) and where it came from. */
  raw: Record<string, string>;
  sourceFile: string;
  sourceRowIndex: number;
  parseIssues: ParseIssue[];
  included: boolean;
  /** Appraiser-written inclusion / exclusion note. */
  selectionNote: string;
  /** Non-negative weight. Defaults to 1 so untouched comps are weighted equally. */
  weight: number;
  /** Manual overrides keyed by adjustment rule id. An override replaces the computed adjustment. */
  overrides: Record<string, AdjustmentOverride>;
  /** Explicit resolutions for categories that would otherwise be unresolved. */
  resolutions: Record<string, CategoryResolution>;
  /** Free-form notes shown in the grid. */
  gridNote: string;
  /** True for synthetic demo rows. */
  synthetic?: boolean;
  /** Present when the record was extracted from a document rather than a CSV. */
  extraction?: ExtractionInfo;
}

export interface AdjustmentOverride {
  amount: number;
  reason: string;
  source: string;
  attachmentRef: string;
}

export type CategoryResolution = 'notApplicable' | 'confirmedZero';

export type RuleKind =
  | 'perUnit' // (subject qty - comp qty) x rate
  | 'flat' // fixed dollar amount applied when a boolean/feature differs
  | 'percent' // percentage of original sale price, signed by subject vs comp direction
  | 'categorical' // appraiser-defined dollar value per rating
  | 'market' // market conditions / time
  | 'concessions' // appraiser's concession adjustment
  | 'manual'; // only manual overrides, no formula

export type BuiltInCategory =
  | 'areaAboveGrade'
  | 'areaBelowGradeFinished'
  | 'areaOther'
  | 'lotSize'
  | 'fullBaths'
  | 'halfBaths'
  | 'bedrooms'
  | 'garageSpaces'
  | 'carportSpaces'
  | 'pool'
  | 'quality'
  | 'condition'
  | 'location'
  | 'view'
  | 'waterfront'
  | 'age'
  | 'marketConditions'
  | 'concessions';

export interface AdjustmentRule {
  id: string;
  /** Built-in category id or `custom:<slug>` */
  category: BuiltInCategory | string;
  label: string;
  kind: RuleKind;
  active: boolean;
  /** Per-unit rate, flat amount, or percent (as a number, e.g. 2.5 for 2.5%). */
  rate: number | null;
  /** Quantitative or boolean/text attribute compared between subject and comp. */
  field: keyof PropertyAttributes | null;
  /** For categorical rules: rating label -> dollar value. Adjustment = value(subject) - value(comp). */
  categoricalValues: Record<string, number>;
  /** Market conditions configuration. */
  market?: {
    mode: 'manual' | 'monthlyRate';
    monthlyRatePct: number | null;
    dateBasis: 'saleDate' | 'contractDate';
  };
  /** Concessions configuration: factor applied to the reported concession (1 = dollar for dollar). */
  concessionFactor?: number | null;
  reason: string;
  source: string;
  attachmentRef: string;
  /** Marks the rule as a custom (user-created) category. */
  custom?: boolean;
  /** Display order. */
  order: number;
}

export interface ReviewFlagSettings {
  netPct: number | null;
  grossPct: number | null;
  singleLinePct: number | null;
  /** Flag adjusted prices more than this percent away from the adjusted median. */
  outlierPct: number | null;
}

export interface ScenarioSettings {
  /** Rounding increment for displayed/exported dollar values (1, 10, 100, 1000). */
  rounding: number;
  flags: ReviewFlagSettings;
}

export interface SubjectProperty {
  attrs: PropertyAttributes;
  photoUrls: string[];
  notes: string;
  /** Where the subject data came from when imported from a CSV row or a document. */
  sourceFile?: string;
  sourceRow?: Record<string, string>;
  parseIssues?: ParseIssue[];
}

/** Provenance for records extracted from PDFs / images. */
export interface ExtractionInfo {
  method: 'ai' | 'local';
  model?: string;
  file: string;
  confidence: 'high' | 'medium' | 'low';
  evidence: { field: string; quote: string }[];
  notes: string;
}

export interface FinalOpinion {
  value: number | null;
  rationale: string;
}

export interface Scenario {
  id: string;
  name: string;
  createdAt: string;
  effectiveDate: string | null;
  subject: SubjectProperty;
  comparables: Comparable[];
  rules: AdjustmentRule[];
  settings: ScenarioSettings;
  finalOpinion: FinalOpinion;
  reconciliationNotes: string;
}

export interface MappingPreset {
  id: string;
  name: string;
  /** source header -> field key */
  mapping: Record<string, FieldKey>;
  options: MappingOptions;
}

export interface MappingOptions {
  dateOrder: 'MDY' | 'DMY' | 'YMD' | 'auto';
  lotUnitsDefault: LotUnits | 'auto' | null;
  bathsConvention: 'fullPointHalf' | 'decimal' | 'none' | null;
}

export interface Project {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  scenarios: Scenario[];
  activeScenarioId: string;
  mappingPresets: MappingPreset[];
  isDemo?: boolean;
  schemaVersion: number;
}

/* ---------- Calculation results ---------- */

export type LineStatus = 'computed' | 'override' | 'unresolved' | 'notApplicable' | 'confirmedZero' | 'inactive';

export interface AdjustmentLine {
  ruleId: string;
  category: string;
  label: string;
  status: LineStatus;
  /** Signed adjustment amount. null when unresolved or inactive. */
  amount: number | null;
  source: 'global' | 'manual' | 'none';
  formula: string;
  explanation: string;
  subjectValue: string | number | boolean | null;
  compValue: string | number | boolean | null;
  reason: string;
  sourceNote: string;
  attachmentRef: string;
}

export interface CompCalculation {
  compId: string;
  salePrice: number | null;
  lines: AdjustmentLine[];
  netAdjustment: number | null;
  grossAdjustment: number | null;
  netPct: number | null;
  grossPct: number | null;
  adjustedPrice: number | null;
  /** Whether this comp can contribute to the value indication. */
  calculable: boolean;
  /** Reasons the comp is not calculable. */
  blockers: string[];
  /** Unresolved rule ids. */
  unresolved: string[];
  flags: ReviewFlag[];
  normalizedWeight: number | null;
  pricePerSqft: number | null;
  pricePerSqftBasis: string;
  statusCategory: StatusCategory | null;
}

export interface ReviewFlag {
  code: 'netPct' | 'grossPct' | 'singleLine' | 'missingData' | 'outlier' | 'unresolved';
  message: string;
}

export interface Reconciliation {
  includedCount: number;
  /** Included comps that are calculable closed sales and contribute to the indication. */
  contributingCount: number;
  /** Included comps excluded from the indication because of unresolved or missing data. */
  excludedForUnresolved: number;
  unadjustedRange: [number, number] | null;
  adjustedRange: [number, number] | null;
  adjustedMean: number | null;
  adjustedMedian: number | null;
  weightedIndication: number | null;
  totalWeight: number;
  message: string | null;
}

export interface Warning {
  code: string;
  message: string;
}

export interface ScenarioCalculation {
  comps: Record<string, CompCalculation>;
  reconciliation: Reconciliation;
  warnings: Warning[];
  /** Explicit statement of calculation order for the report. */
  calculationOrder: string[];
}
