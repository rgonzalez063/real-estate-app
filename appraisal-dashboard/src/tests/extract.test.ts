import { describe, expect, it } from 'vitest';
import { parseListingText } from '../engine/extract/local';
import { emptyExtractedFields, extractedToAttributes, ExtractionResultSchema } from '../engine/extract/schema';

describe('document extraction (local text scan)', () => {
  it('picks up labelled values and quotes evidence without inventing anything', () => {
    const text = `MLS #: 55501    Status: Closed
Address: 400 Harbor Way, Houston TX 77001
Sold Price: $412,000    List Price: $419,900
Close Date: 07/15/2026
Beds: 4    Baths Full: 2    Baths Half: 1
Year Built: 2006    Garage Spaces: 2
GLA: 2,150    Lot Size: 0.21 acres
Private Pool Y/N: N
Public Remarks: Updated kitchen, new roof 2024.`;
    const rec = parseListingText(text, 'sheet.pdf');
    expect(rec.fields.mlsNumber).toBe('55501');
    expect(rec.fields.status).toBe('Closed');
    expect(rec.fields.salePrice).toBe('$412,000');
    expect(rec.fields.listPrice).toBe('$419,900');
    expect(rec.fields.saleDate).toBe('07/15/2026');
    expect(rec.fields.bedrooms).toBe('4');
    expect(rec.fields.fullBaths).toBe('2');
    expect(rec.fields.halfBaths).toBe('1');
    expect(rec.fields.bathsCombined).toBeNull(); // "Baths Full: 2" must not leak into the combined field
    expect(rec.fields.yearBuilt).toBe('2006');
    expect(rec.fields.areaAboveGrade).toBe('2,150');
    expect(rec.fields.lotSize).toBe('0.21 acres');
    expect(rec.fields.pool).toBe('N');
    expect(rec.fields.latitude).toBeNull();
    expect(rec.fields.photoUrl).toBeNull();
    expect(rec.confidence).toBe('low');
    expect(rec.evidence.find((e) => e.field === 'salePrice')?.quote).toContain('$412,000');
  });

  it('returns an empty record with a note when nothing matches', () => {
    const rec = parseListingText('lorem ipsum dolor sit amet');
    expect(rec.evidence).toHaveLength(0);
    expect(Object.values(rec.fields).every((v) => v === null)).toBe(true);
    expect(rec.notes).toMatch(/no recognisable/);
  });
});

describe('extracted fields to attributes', () => {
  it('normalises through the shared parsers and keeps unknowns unknown', () => {
    const fields = { ...emptyExtractedFields(), salePrice: '$412,000', saleDate: '2026-07-15', areaAboveGrade: '2,150 sq ft', lotSize: '0.21 acres', pool: 'no', status: 'Sold', bathsCombined: '3.1' };
    const { attrs, raw, issues } = extractedToAttributes(fields, 'sheet.pdf');
    expect(attrs.salePrice).toBe(412000);
    expect(attrs.saleDate).toBe('2026-07-15');
    expect(attrs.areaAboveGrade).toBe(2150);
    expect(attrs.lotSizeUnits).toBe('acres');
    expect(attrs.lotSizeSqft).toBeCloseTo(0.21 * 43560);
    expect(attrs.pool).toBe(false);
    expect(attrs.status).toBe('closed');
    expect(attrs.fullBaths).toBeNull(); // combined baths are never split without a confirmed convention
    expect(attrs.yearBuilt).toBeNull();
    expect(raw.salePrice).toBe('$412,000');
    expect(issues.some((i) => i.field === 'bathsCombined')).toBe(true);
  });

  it('schema rejects records with invented numeric types', () => {
    const bad = { documentSummary: 'x', records: [{ recordType: 'comparable', label: 'a', fields: { ...emptyExtractedFields(), salePrice: 412000 }, evidence: [], confidence: 'high', notes: '' }] };
    expect(ExtractionResultSchema.safeParse(bad).success).toBe(false);
    const good = { documentSummary: 'x', records: [{ recordType: 'comparable', label: 'a', fields: emptyExtractedFields(), evidence: [], confidence: 'high', notes: '' }] };
    expect(ExtractionResultSchema.safeParse(good).success).toBe(true);
  });
});
