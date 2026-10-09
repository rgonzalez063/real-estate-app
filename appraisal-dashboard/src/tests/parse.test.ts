import { describe, expect, it } from 'vitest';
import { parseBoolean, parseCurrency, parseDate, parseLot, parseNumber, parseStatus, sanitizeText, splitCombinedBaths, validatePhotoUrl } from '../engine/parse';

describe('numbers and currency', () => {
  it('parses currency symbols, commas and quoted text', () => {
    expect(parseCurrency('$300,000').value).toBe(300000);
    expect(parseCurrency('"$1,250,000.50"'.replace(/"/g, '')).value).toBe(1250000.5);
    expect(parseCurrency('(5,000)').value).toBe(-5000);
    expect(parseCurrency('-2,500').value).toBe(-2500);
    expect(parseNumber('2,100 sqft').value).toBe(2100);
  });
  it('keeps empty and placeholder values unknown', () => {
    expect(parseNumber('').value).toBeNull();
    expect(parseNumber('N/A').value).toBeNull();
    expect(parseNumber(null).value).toBeNull();
    expect(parseNumber('abc').value).toBeNull();
    expect(parseNumber('abc').issue).toBeTruthy();
  });
});

describe('dates', () => {
  it('parses common formats to ISO', () => {
    expect(parseDate('2024-03-05').value).toBe('2024-03-05');
    expect(parseDate('03/05/2024').value).toBe('2024-03-05');
    expect(parseDate('3/5/24').value).toBe('2024-03-05');
    expect(parseDate('Mar 5, 2024').value).toBe('2024-03-05');
    expect(parseDate('5 March 2024').value).toBe('2024-03-05');
    expect(parseDate('2024-03-05T14:22:00Z').value).toBe('2024-03-05');
    expect(parseDate('20240305').value).toBe('2024-03-05');
  });
  it('flags ambiguous day/month order and honours an explicit order', () => {
    const auto = parseDate('03/04/2024', 'auto');
    expect(auto.ambiguous).toBe(true);
    expect(auto.value).toBe('2024-03-04');
    expect(parseDate('03/04/2024', 'DMY').value).toBe('2024-04-03');
    expect(parseDate('25/04/2024', 'auto').value).toBe('2024-04-25');
    expect(parseDate('25/04/2024', 'auto').ambiguous).toBeFalsy();
  });
  it('rejects invalid dates', () => {
    expect(parseDate('02/30/2024').value).toBeNull();
    expect(parseDate('not a date').issue).toBeTruthy();
  });
});

describe('lot size and unit conversion', () => {
  it('converts acres to square feet', () => {
    const r = parseLot('0.25', 'acres');
    expect(r.units).toBe('acres');
    expect(r.sqft).toBe(10890);
    expect(parseLot('0.5 ac', null).sqft).toBe(21780);
  });
  it('keeps square feet as is', () => {
    expect(parseLot('9,000', 'sqft').sqft).toBe(9000);
    expect(parseLot('9000 sf', null).sqft).toBe(9000);
  });
  it('requires units when they cannot be determined', () => {
    const r = parseLot('9000', null);
    expect(r.sqft).toBeNull();
    expect(r.raw).toBe(9000);
    expect(r.needsUnits).toBe(true);
  });
  it('auto mode uses magnitude', () => {
    expect(parseLot('0.3', 'auto').units).toBe('acres');
    expect(parseLot('12000', 'auto').units).toBe('sqft');
  });
});

describe('baths', () => {
  it('only splits combined baths under a confirmed convention', () => {
    expect(splitCombinedBaths('3.1', 'none').full).toBeNull();
    expect(splitCombinedBaths('3.1', 'fullPointHalf')).toMatchObject({ full: 3, half: 1 });
    expect(splitCombinedBaths('3.5', 'decimal')).toMatchObject({ full: 3, half: 1 });
    expect(splitCombinedBaths('3.1', 'decimal').full).toBeNull();
  });
});

describe('booleans, status, text, urls', () => {
  it('parses yes/no variants and leaves blanks unknown', () => {
    expect(parseBoolean('Y').value).toBe(true);
    expect(parseBoolean('No').value).toBe(false);
    expect(parseBoolean('').value).toBeNull();
    expect(parseBoolean('In Ground').value).toBe(true);
    expect(parseBoolean('In Ground').issue).toBeTruthy();
  });
  it('classifies statuses', () => {
    expect(parseStatus('Closed').category).toBe('closed');
    expect(parseStatus('Sold').category).toBe('closed');
    expect(parseStatus('Active').category).toBe('active');
    expect(parseStatus('Pending').category).toBe('pending');
    expect(parseStatus('Active Under Contract').category).toBe('pending');
    expect(parseStatus('Expired').category).toBe('other');
    expect(parseStatus('').category).toBeNull();
  });
  it('sanitizes control characters and whitespace', () => {
    expect(sanitizeText('  Hello\u0000  world \n')).toBe('Hello world');
    expect(sanitizeText('')).toBeNull();
  });
  it('validates photo URLs', () => {
    expect(validatePhotoUrl('https://example.com/a.jpg').value).toBe('https://example.com/a.jpg');
    expect(validatePhotoUrl('javascript:alert(1)').value).toBeNull();
    expect(validatePhotoUrl('not a url').value).toBeNull();
    expect(validatePhotoUrl('').value).toBeNull();
  });
});
