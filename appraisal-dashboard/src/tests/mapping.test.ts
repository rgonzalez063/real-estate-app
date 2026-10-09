import { describe, expect, it } from 'vitest';
import Papa from 'papaparse';
import { findDuplicates, normalizeRows, suggestMapping, defaultMappingOptions } from '../engine/mapping';

const CSV = `MLS #,Address,City,Zip Code,Subdivision,Status,Close Price,List Price,Close Date,Pending Date,DOM,GLA,Lot Size,Beds,Baths Full,Baths Half,Year Built,Garage Spaces,Private Pool Y/N,Condition,Seller Concessions,Latitude,Longitude,Photo URL
12345,"101 Main St",Houston,77001,Oak Park,Closed,"$300,000","$305,000",07/15/2026,06/20/2026,21,"1,800",0.21,3,2,1,2006,2,N,C3,"$0",29.7,-95.4,https://example.com/p.jpg
12346,"202 Elm St",Houston,77001,Oak Park,Active,,"$345,000",,,12,2050,9000,3,2,,2008,2,Y,,,,,`;

describe('CSV parsing and mapping', () => {
  const parsed = Papa.parse<Record<string, string>>(CSV, { header: true, skipEmptyLines: true });
  const headers = parsed.meta.fields ?? [];

  it('suggests mappings from HARMLS/Matrix style headers including legacy GLA', () => {
    const m = suggestMapping(headers);
    expect(m['MLS #']).toBe('mlsNumber');
    expect(m['Close Price']).toBe('salePrice');
    expect(m['Close Date']).toBe('saleDate');
    expect(m['Pending Date']).toBe('contractDate');
    expect(m['GLA']).toBe('areaAboveGrade');
    expect(m['Baths Full']).toBe('fullBaths');
    expect(m['Baths Half']).toBe('halfBaths');
    expect(m['Private Pool Y/N']).toBe('pool');
    expect(m['Seller Concessions']).toBe('concessions');
    expect(m['Zip Code']).toBe('zip');
    expect(m['Subdivision']).toBe('neighborhood');
    expect(m['Photo URL']).toBe('photoUrl');
  });

  it('normalises rows, preserves raw data and keeps missing values unknown', () => {
    const mapping = suggestMapping(headers);
    const res = normalizeRows(parsed.data, mapping, { ...defaultMappingOptions(), lotUnitsDefault: 'auto', dateOrder: 'MDY' }, 'export.csv');
    expect(res.comps).toHaveLength(2);
    const [a, b] = res.comps;
    expect(a.attrs.salePrice).toBe(300000);
    expect(a.attrs.saleDate).toBe('2026-07-15');
    expect(a.attrs.areaAboveGrade).toBe(1800);
    expect(a.attrs.lotSizeUnits).toBe('acres');
    expect(a.attrs.lotSizeSqft).toBeCloseTo(0.21 * 43560);
    expect(a.attrs.pool).toBe(false);
    expect(a.attrs.status).toBe('closed');
    expect(a.attrs.concessions).toBe(0);
    expect(a.attrs.photoUrl).toBe('https://example.com/p.jpg');
    expect(a.raw['Close Price']).toBe('$300,000');
    expect(a.sourceFile).toBe('export.csv');
    expect(b.attrs.salePrice).toBeNull();
    expect(b.attrs.saleDate).toBeNull();
    expect(b.attrs.halfBaths).toBeNull();
    expect(b.attrs.concessions).toBeNull();
    expect(b.attrs.latitude).toBeNull();
    expect(b.attrs.photoUrl).toBeNull();
    expect(b.attrs.status).toBe('active');
    expect(b.attrs.lotSizeSqft).toBe(9000);
    expect(res.missingByField.salePrice).toBe(1);
  });

  it('asks for lot units when ambiguous and date order when ambiguous', () => {
    const mapping = suggestMapping(headers);
    const res = normalizeRows(parsed.data, mapping, defaultMappingOptions(), 'export.csv');
    expect(res.lotNeedsUnits).toBe(true);
    expect(res.comps[0].attrs.lotSizeSqft).toBeNull();
    expect(res.comps[0].attrs.lotSizeRaw).toBe(0.21);
    const ambiguous = Papa.parse<Record<string, string>>('Close Date\n03/04/2024', { header: true });
    const r2 = normalizeRows(ambiguous.data, { 'Close Date': 'saleDate' }, defaultMappingOptions(), 'x.csv');
    expect(r2.datesAmbiguous).toBe(true);
  });

  it('does not split combined baths without a confirmed convention', () => {
    const p = Papa.parse<Record<string, string>>('Baths Total\n3.1', { header: true });
    const r = normalizeRows(p.data, { 'Baths Total': 'bathsCombined' }, defaultMappingOptions(), 'x.csv');
    expect(r.bathsNeedsConvention).toBe(true);
    expect(r.comps[0].attrs.fullBaths).toBeNull();
    const r2 = normalizeRows(p.data, { 'Baths Total': 'bathsCombined' }, { ...defaultMappingOptions(), bathsConvention: 'fullPointHalf' }, 'x.csv');
    expect(r2.comps[0].attrs.fullBaths).toBe(3);
    expect(r2.comps[0].attrs.halfBaths).toBe(1);
  });

  it('flags duplicates by MLS number or address plus date', () => {
    const p = Papa.parse<Record<string, string>>('MLS #,Address,Close Date\n1,101 Main St,2026-01-01\n1,101 Main Street,2026-01-01\n2,202 Elm St,2026-02-02\n3,202 Elm St.,2026-02-02', { header: true });
    const r = normalizeRows(p.data, { 'MLS #': 'mlsNumber', Address: 'address', 'Close Date': 'saleDate' }, defaultMappingOptions(), 'x.csv');
    const groups = findDuplicates(r.comps);
    expect(groups).toHaveLength(2);
    expect(groups.find((g) => g.reason === 'mls')!.compIds).toHaveLength(2);
    expect(groups.find((g) => g.reason === 'addressDate')!.compIds).toHaveLength(2);
  });
});
