import type { AdjustmentLine, AdjustmentRule, Comparable, PropertyAttributes, ReviewFlag, ReviewFlagSettings, SubjectProperty } from '../types';
import { ATTR_LABELS } from './fields';
import { elapsedMonths, formatDate } from './dates';
import { fmtMoney, fmtNumber } from './format';

type AttrValue = string | number | boolean | null;

function attr(a: PropertyAttributes, key: keyof PropertyAttributes | null): AttrValue {
  if (!key) return null;
  const v = a[key];
  return v === undefined ? null : (v as AttrValue);
}

function describe(v: AttrValue): string {
  if (v === null) return 'unknown';
  if (typeof v === 'boolean') return v ? 'Yes' : 'No';
  if (typeof v === 'number') return fmtNumber(v, 2);
  return v;
}

function baseLine(rule: AdjustmentRule, subjectValue: AttrValue, compValue: AttrValue): AdjustmentLine {
  return {
    ruleId: rule.id,
    category: rule.category,
    label: rule.label,
    status: 'inactive',
    amount: null,
    source: 'none',
    formula: '',
    explanation: '',
    subjectValue,
    compValue,
    reason: rule.reason,
    sourceNote: rule.source,
    attachmentRef: rule.attachmentRef,
  };
}

export interface LineContext {
  effectiveDate: string | null;
}

/**
 * Compute a single adjustment line for one comparable.
 *
 * Precedence: inactive rule -> manual override -> explicit resolution -> computed / unresolved.
 * An override replaces the computed adjustment; it is never added on top of it.
 */
export function computeLine(rule: AdjustmentRule, subject: SubjectProperty, comp: Comparable, ctx: LineContext): AdjustmentLine {
  const s = subject.attrs;
  const c = comp.attrs;
  const field = rule.field;
  const subjectValue = attr(s, field);
  const compValue = attr(c, field);
  const line = baseLine(rule, subjectValue, compValue);

  if (!rule.active) {
    line.explanation = 'Rule inactive';
    return line;
  }

  const override = comp.overrides[rule.id];
  if (override) {
    line.status = 'override';
    line.amount = override.amount;
    line.source = 'manual';
    line.formula = `Manual override = ${fmtMoney(override.amount, { signed: true })}`;
    line.explanation = override.reason || 'Manual override replaces the global rule for this comparable.';
    line.reason = override.reason;
    line.sourceNote = override.source;
    line.attachmentRef = override.attachmentRef;
    return line;
  }

  const resolution = comp.resolutions[rule.id];
  if (resolution === 'notApplicable') {
    line.status = 'notApplicable';
    line.amount = null;
    line.explanation = 'Marked not applicable for this comparable (excluded from net and gross).';
    return line;
  }
  if (resolution === 'confirmedZero') {
    line.status = 'confirmedZero';
    line.amount = 0;
    line.source = 'manual';
    line.formula = 'Confirmed $0';
    line.explanation = 'Appraiser confirmed no adjustment for this comparable.';
    return line;
  }

  const unresolved = (why: string) => {
    line.status = 'unresolved';
    line.amount = null;
    line.explanation = why;
    return line;
  };

  switch (rule.kind) {
    case 'perUnit': {
      if (!field) return unresolved('No attribute selected for this rule.');
      if (rule.rate === null) return unresolved('Rate not set. Enter a supported rate or mark not applicable.');
      if (typeof subjectValue !== 'number') return unresolved(`Subject ${ATTR_LABELS[field]} is unknown.`);
      if (typeof compValue !== 'number') return unresolved(`Comparable ${ATTR_LABELS[field]} is unknown.`);
      const diff = subjectValue - compValue;
      const amount = diff * rule.rate;
      line.status = 'computed';
      line.source = 'global';
      line.amount = amount;
      line.formula = `(${fmtNumber(subjectValue, 2)} − ${fmtNumber(compValue, 2)}) × ${fmtMoney(rule.rate, { decimals: 2 })} = ${fmtMoney(amount, { signed: true })}`;
      line.explanation =
        diff === 0
          ? 'Subject and comparable are equal; no adjustment.'
          : diff > 0
            ? 'Comparable is inferior on this attribute; adjustment is added.'
            : 'Comparable is superior on this attribute; adjustment is subtracted.';
      return line;
    }
    case 'flat': {
      if (!field) return unresolved('No attribute selected for this rule.');
      if (rule.rate === null) return unresolved('Flat amount not set.');
      if (typeof subjectValue !== 'boolean') return unresolved(`Subject ${ATTR_LABELS[field]} is unknown.`);
      if (typeof compValue !== 'boolean') return unresolved(`Comparable ${ATTR_LABELS[field]} is unknown.`);
      const sign = subjectValue === compValue ? 0 : subjectValue ? 1 : -1;
      const amount = sign * rule.rate;
      line.status = 'computed';
      line.source = 'global';
      line.amount = amount;
      line.formula = `${describe(subjectValue)} vs ${describe(compValue)} → ${sign === 0 ? '0' : `${sign > 0 ? '+' : '−'}${fmtMoney(rule.rate)}`}`;
      line.explanation = sign === 0 ? 'Both have the same feature status.' : sign > 0 ? 'Subject has the feature and the comparable does not; added.' : 'Comparable has the feature and the subject does not; subtracted.';
      return line;
    }
    case 'percent': {
      if (!field) return unresolved('No attribute selected for this rule.');
      if (c.salePrice === null) return unresolved('Comparable sale price is unknown.');
      if (typeof subjectValue === 'boolean' || subjectValue === null) {
        if (rule.rate === null) return unresolved('Percentage not set.');
        if (typeof subjectValue !== 'boolean') return unresolved(`Subject ${ATTR_LABELS[field]} is unknown.`);
        if (typeof compValue !== 'boolean') return unresolved(`Comparable ${ATTR_LABELS[field]} is unknown.`);
        const sign = subjectValue === compValue ? 0 : subjectValue ? 1 : -1;
        const amount = (sign * rule.rate * c.salePrice) / 100;
        line.status = 'computed';
        line.source = 'global';
        line.amount = amount;
        line.formula = `${sign === 0 ? '0' : `${sign > 0 ? '+' : '−'}${rule.rate}%`} × ${fmtMoney(c.salePrice)} = ${fmtMoney(amount, { signed: true })}`;
        line.explanation = sign === 0 ? 'Both have the same feature status.' : 'Percentage of the comparable original sale price, signed by which property is superior.';
        return line;
      }
      // Rating-based percent: categoricalValues hold percentages
      const sv = rule.categoricalValues[String(subjectValue)];
      const cv = typeof compValue === 'string' ? rule.categoricalValues[compValue] : undefined;
      if (sv === undefined) return unresolved(`No percentage defined for subject rating "${subjectValue}".`);
      if (compValue === null) return unresolved(`Comparable ${ATTR_LABELS[field]} is unknown.`);
      if (cv === undefined) return unresolved(`No percentage defined for comparable rating "${compValue}".`);
      const amount = ((sv - cv) * c.salePrice) / 100;
      line.status = 'computed';
      line.source = 'global';
      line.amount = amount;
      line.formula = `(${sv}% − ${cv}%) × ${fmtMoney(c.salePrice)} = ${fmtMoney(amount, { signed: true })}`;
      line.explanation = 'Difference between appraiser-defined percentages for each rating, applied to the comparable original sale price.';
      return line;
    }
    case 'categorical': {
      if (!field) return unresolved('No attribute selected for this rule.');
      if (subjectValue === null) return unresolved(`Subject ${ATTR_LABELS[field]} is unknown.`);
      if (compValue === null) return unresolved(`Comparable ${ATTR_LABELS[field]} is unknown.`);
      const sv = rule.categoricalValues[String(subjectValue)];
      const cv = rule.categoricalValues[String(compValue)];
      if (sv === undefined) return unresolved(`No dollar value defined for subject rating "${describe(subjectValue)}".`);
      if (cv === undefined) return unresolved(`No dollar value defined for comparable rating "${describe(compValue)}".`);
      const amount = sv - cv;
      line.status = 'computed';
      line.source = 'global';
      line.amount = amount;
      line.formula = `${fmtMoney(sv)} (${describe(subjectValue)}) − ${fmtMoney(cv)} (${describe(compValue)}) = ${fmtMoney(amount, { signed: true })}`;
      line.explanation = 'Appraiser-defined dollar relationship between rating categories; intervals are not assumed equal.';
      return line;
    }
    case 'market': {
      const cfg = rule.market ?? { mode: 'manual', monthlyRatePct: null, dateBasis: 'saleDate' };
      if (cfg.mode === 'manual') return unresolved('Market conditions set to manual; enter an override or mark not applicable.');
      if (cfg.monthlyRatePct === null) return unresolved('Monthly market rate not set.');
      if (!ctx.effectiveDate) return unresolved('Effective date not set.');
      const startDate = cfg.dateBasis === 'contractDate' ? c.contractDate : c.saleDate;
      if (!startDate) return unresolved(`Comparable ${cfg.dateBasis === 'contractDate' ? 'contract' : 'sale'} date is unknown.`);
      if (c.salePrice === null) return unresolved('Comparable sale price is unknown.');
      const months = elapsedMonths(startDate, ctx.effectiveDate);
      const r = cfg.monthlyRatePct / 100;
      const amount = c.salePrice * (Math.pow(1 + r, months) - 1);
      line.status = 'computed';
      line.source = 'global';
      line.amount = amount;
      line.subjectValue = ctx.effectiveDate;
      line.compValue = startDate;
      line.formula = `${fmtMoney(c.salePrice)} × [(1 + ${cfg.monthlyRatePct}%)^${months.toFixed(3)} − 1] = ${fmtMoney(amount, { signed: true })}`;
      line.explanation = `Start ${formatDate(startDate)} (${cfg.dateBasis === 'contractDate' ? 'contract date' : 'sale date'}) to effective ${formatDate(ctx.effectiveDate)} = ${months.toFixed(3)} months (days ÷ 30.4375). Price basis: original sale price.${months < 0 ? ' Note: start date is after the effective date.' : ''}`;
      return line;
    }
    case 'concessions': {
      if (rule.concessionFactor === null || rule.concessionFactor === undefined) return unresolved('Concession factor not set.');
      if (c.concessions === null) return unresolved('Reported concessions are unknown for this comparable.');
      const amount = -c.concessions * rule.concessionFactor;
      line.status = 'computed';
      line.source = 'global';
      line.amount = amount;
      line.subjectValue = null;
      line.compValue = c.concessions;
      line.formula = `−(${fmtMoney(c.concessions)} reported × ${rule.concessionFactor}) = ${fmtMoney(amount, { signed: true })}`;
      line.explanation = 'Appraiser-selected factor applied to the reported concession. Reported concessions are source data, not an automatic deduction.';
      return line;
    }
    case 'manual':
    default:
      return unresolved('Manual category: enter an override, confirm $0, or mark not applicable.');
  }
}

export interface CompTotals {
  net: number | null;
  gross: number | null;
  netPct: number | null;
  grossPct: number | null;
  adjustedPrice: number | null;
  unresolved: string[];
}

/** Sum signed and absolute adjustments. Inactive / N/A lines are skipped; unresolved lines block the result. */
export function totalLines(salePrice: number | null, lines: AdjustmentLine[]): CompTotals {
  const unresolved = lines.filter((l) => l.status === 'unresolved').map((l) => l.ruleId);
  const counted = lines.filter((l) => l.amount !== null && (l.status === 'computed' || l.status === 'override' || l.status === 'confirmedZero'));
  const net = counted.reduce((a, l) => a + (l.amount as number), 0);
  const gross = counted.reduce((a, l) => a + Math.abs(l.amount as number), 0);
  if (salePrice === null || unresolved.length > 0) {
    return { net: unresolved.length ? null : net, gross: unresolved.length ? null : gross, netPct: null, grossPct: null, adjustedPrice: null, unresolved };
  }
  return {
    net,
    gross,
    netPct: salePrice !== 0 ? (net / salePrice) * 100 : null,
    grossPct: salePrice !== 0 ? (gross / salePrice) * 100 : null,
    adjustedPrice: salePrice + net,
    unresolved,
  };
}

export function reviewFlags(
  settings: ReviewFlagSettings,
  salePrice: number | null,
  totals: CompTotals,
  lines: AdjustmentLine[],
  missingFields: string[],
  adjustedMedian: number | null,
): ReviewFlag[] {
  const flags: ReviewFlag[] = [];
  if (totals.unresolved.length) flags.push({ code: 'unresolved', message: `${totals.unresolved.length} unresolved adjustment${totals.unresolved.length > 1 ? 's' : ''}` });
  if (settings.netPct !== null && totals.netPct !== null && Math.abs(totals.netPct) > settings.netPct) flags.push({ code: 'netPct', message: `Net adjustment ${totals.netPct.toFixed(1)}% exceeds ${settings.netPct}%` });
  if (settings.grossPct !== null && totals.grossPct !== null && totals.grossPct > settings.grossPct) flags.push({ code: 'grossPct', message: `Gross adjustment ${totals.grossPct.toFixed(1)}% exceeds ${settings.grossPct}%` });
  if (settings.singleLinePct !== null && salePrice) {
    for (const l of lines) {
      if (l.amount !== null && Math.abs(l.amount) / salePrice > settings.singleLinePct / 100) flags.push({ code: 'singleLine', message: `${l.label} adjustment is ${((Math.abs(l.amount) / salePrice) * 100).toFixed(1)}% of sale price` });
    }
  }
  if (missingFields.length) flags.push({ code: 'missingData', message: `Missing: ${missingFields.join(', ')}` });
  if (settings.outlierPct !== null && totals.adjustedPrice !== null && adjustedMedian) {
    const dev = Math.abs(totals.adjustedPrice - adjustedMedian) / adjustedMedian;
    if (dev > settings.outlierPct / 100) flags.push({ code: 'outlier', message: `Adjusted price is ${(dev * 100).toFixed(1)}% from the adjusted median` });
  }
  return flags;
}
