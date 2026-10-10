import { FIELD_DEFS } from '../fields';
import type { FieldKey } from '../../types';
import { emptyExtractedFields, type ExtractedRecord } from './schema';

/**
 * Offline, best-effort label/value scan of document text. It only picks up values that sit
 * next to a recognised label ("Sold Price: $412,000", "Year Built 2006"). Everything it finds is
 * marked low confidence and must be reviewed; it never guesses.
 */
interface Alias {
  key: FieldKey;
  alias: string;
  re: RegExp;
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

const ALIASES: Alias[] = FIELD_DEFS.flatMap((d) =>
  [...d.aliases, d.label]
    .filter((a) => a.length >= 3)
    .map((alias) => ({
      key: d.key,
      alias,
      // label, optional punctuation, then the value up to the end of line or a double space / tab.
      re: new RegExp(`(?:^|[\\s|])${escapeRe(alias).replace(/\\ /g, '\\s+')}\\s*[:#=]?\\s*([^\\n\\t|]{1,80}?)(?=\\s{2,}|\\t|\\n|\\||$)`, 'i'),
    })),
).sort((a, b) => b.alias.length - a.alias.length);

const VALUE_OK = /[0-9a-z]/i;

export function parseListingText(text: string, label = 'Document'): ExtractedRecord {
  const fields = emptyExtractedFields();
  const evidence: { field: string; quote: string }[] = [];
  const normalized = text.replace(/\r/g, '').replace(/[ \u00a0]{2,}/g, '  ');
  for (const a of ALIASES) {
    if (fields[a.key] !== null) continue;
    const m = normalized.match(a.re);
    if (!m) continue;
    const value = m[1].trim().replace(/^[:#=\-–]\s*/, '');
    if (!VALUE_OK.test(value)) continue;
    // Reject values that are really the continuation of a longer label ("Baths Full: 2" seen via "Baths").
    if (/^[A-Za-z][A-Za-z /#]{0,30}[:#=]/.test(value)) continue;
    // Reject values that are just another label (e.g. "Beds  Baths").
    if (ALIASES.some((o) => o.key !== a.key && o.alias.toLowerCase() === value.toLowerCase())) continue;
    fields[a.key] = value;
    evidence.push({ field: a.key, quote: m[0].trim().slice(0, 120) });
  }
  const found = evidence.length;
  return {
    recordType: 'unknown',
    label,
    fields,
    evidence,
    confidence: 'low',
    notes: found ? `Local text scan matched ${found} labelled value${found === 1 ? '' : 's'}. Review every field; values are taken literally from label/value pairs.` : 'Local text scan found no recognisable label/value pairs. Use AI extraction or enter values manually.',
  };
}
