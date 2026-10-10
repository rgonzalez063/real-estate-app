import { z } from 'zod';
import type { Comparable, FieldKey, MappingOptions, ParseIssue, PropertyAttributes } from '../../types';
import { normalizeRows } from '../mapping';

/**
 * Structured output schema for document extraction. Every value is a string exactly as it appears
 * (or null when the document does not state it) so the app's own parsers normalise it and the
 * appraiser can see what was read. Nothing is inferred or invented.
 */
const nullableText = z.string().nullable();

export const ExtractedFieldsSchema = z.object({
  mlsNumber: nullableText,
  address: nullableText,
  city: nullableText,
  zip: nullableText,
  neighborhood: nullableText,
  status: nullableText,
  salePrice: nullableText,
  listPrice: nullableText,
  saleDate: nullableText,
  contractDate: nullableText,
  daysOnMarket: nullableText,
  areaAboveGrade: nullableText,
  areaBelowGradeFinished: nullableText,
  areaBelowGradeUnfinished: nullableText,
  areaOther: nullableText,
  lotSize: nullableText,
  lotSizeUnits: nullableText,
  bedrooms: nullableText,
  fullBaths: nullableText,
  halfBaths: nullableText,
  bathsCombined: nullableText,
  yearBuilt: nullableText,
  age: nullableText,
  garageSpaces: nullableText,
  carportSpaces: nullableText,
  pool: nullableText,
  quality: nullableText,
  condition: nullableText,
  renovation: nullableText,
  remarks: nullableText,
  waterfront: nullableText,
  waterView: nullableText,
  location: nullableText,
  view: nullableText,
  design: nullableText,
  tenure: nullableText,
  functionalUtility: nullableText,
  heatingCooling: nullableText,
  energyEfficient: nullableText,
  porchPatioDeck: nullableText,
  amenities: nullableText,
  concessions: nullableText,
  latitude: nullableText,
  longitude: nullableText,
  distance: nullableText,
  photoUrl: nullableText,
});

export const ExtractedRecordSchema = z.object({
  recordType: z.enum(['subject', 'comparable', 'unknown']),
  label: z.string(),
  fields: ExtractedFieldsSchema,
  evidence: z.array(z.object({ field: z.string(), quote: z.string() })),
  confidence: z.enum(['high', 'medium', 'low']),
  notes: z.string(),
});

export const ExtractionResultSchema = z.object({
  documentSummary: z.string(),
  records: z.array(ExtractedRecordSchema),
});

export type ExtractedFields = z.infer<typeof ExtractedFieldsSchema>;
export type ExtractedRecord = z.infer<typeof ExtractedRecordSchema>;
export type ExtractionResult = z.infer<typeof ExtractionResultSchema>;

export const EXTRACTED_FIELD_KEYS = Object.keys(ExtractedFieldsSchema.shape) as FieldKey[];

export function emptyExtractedFields(): ExtractedFields {
  return Object.fromEntries(EXTRACTED_FIELD_KEYS.map((k) => [k, null])) as ExtractedFields;
}

/** Identity mapping so the normal CSV normaliser can process an extracted record. */
const IDENTITY_MAPPING: Record<string, FieldKey> = Object.fromEntries(EXTRACTED_FIELD_KEYS.map((k) => [k, k]));

export const EXTRACTION_OPTIONS: MappingOptions = { dateOrder: 'auto', lotUnitsDefault: null, bathsConvention: null };

/** Turn extracted text fields into normalised attributes using the same parsers as the CSV importer. */
export function extractedToAttributes(fields: ExtractedFields, sourceFile: string, options: MappingOptions = EXTRACTION_OPTIONS): { attrs: PropertyAttributes; raw: Record<string, string>; issues: ParseIssue[]; comp: Comparable } {
  const row: Record<string, string> = {};
  for (const k of EXTRACTED_FIELD_KEYS) {
    const v = fields[k];
    if (v !== null && v !== undefined && String(v).trim() !== '') row[k] = String(v);
  }
  const res = normalizeRows([row], IDENTITY_MAPPING, options, sourceFile);
  const comp = res.comps[0];
  return { attrs: comp.attrs, raw: row, issues: comp.parseIssues, comp };
}
