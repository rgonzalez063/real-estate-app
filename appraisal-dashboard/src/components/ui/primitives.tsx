import { useEffect, useId, useRef, useState, type ReactNode } from 'react';

/* ---------- Number input that preserves "unknown" (null) ---------- */
export function NumberInput({
  value,
  onChange,
  placeholder = '—',
  step,
  min,
  className = '',
  id,
  ariaLabel,
  disabled,
  suffix,
}: {
  value: number | null;
  onChange: (v: number | null) => void;
  placeholder?: string;
  step?: number | 'any';
  min?: number;
  className?: string;
  id?: string;
  ariaLabel?: string;
  disabled?: boolean;
  suffix?: string;
}) {
  const [text, setText] = useState(value === null ? '' : String(value));
  const focused = useRef(false);
  useEffect(() => {
    if (!focused.current) setText(value === null ? '' : String(value));
  }, [value]);
  return (
    <div className={`relative ${className}`}>
      <input
        id={id}
        aria-label={ariaLabel}
        type="text"
        inputMode="decimal"
        className={`input ${suffix ? 'pr-10' : ''}`}
        placeholder={placeholder}
        value={text}
        disabled={disabled}
        onFocus={() => (focused.current = true)}
        onBlur={() => {
          focused.current = false;
          setText(value === null ? '' : String(value));
        }}
        onChange={(e) => {
          const t = e.target.value;
          setText(t);
          const cleaned = t.replace(/[$,%\s]/g, '');
          if (cleaned === '' || cleaned === '-' || cleaned === '.') {
            onChange(null);
            return;
          }
          const n = Number(cleaned);
          if (Number.isFinite(n)) {
            if (min !== undefined && n < min) return;
            onChange(n);
          }
        }}
        step={step}
      />
      {suffix && <span className="pointer-events-none absolute inset-y-0 right-2.5 flex items-center text-xs text-slate-400">{suffix}</span>}
    </div>
  );
}

export function TextInput({ value, onChange, placeholder, id, ariaLabel, className = '', type = 'text', disabled }: { value: string | null; onChange: (v: string | null) => void; placeholder?: string; id?: string; ariaLabel?: string; className?: string; type?: string; disabled?: boolean }) {
  return (
    <input
      id={id}
      aria-label={ariaLabel}
      type={type}
      className={`input ${className}`}
      placeholder={placeholder ?? '—'}
      value={value ?? ''}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value === '' ? null : e.target.value)}
    />
  );
}

export function TextArea({ value, onChange, placeholder, id, ariaLabel, rows = 3, className = '' }: { value: string; onChange: (v: string) => void; placeholder?: string; id?: string; ariaLabel?: string; rows?: number; className?: string }) {
  return <textarea id={id} aria-label={ariaLabel} rows={rows} className={`input ${className}`} placeholder={placeholder} value={value} onChange={(e) => onChange(e.target.value)} />;
}

export function Select<T extends string>({ value, onChange, options, id, ariaLabel, className = '', placeholder }: { value: T | null; onChange: (v: T | null) => void; options: { value: T; label: string }[]; id?: string; ariaLabel?: string; className?: string; placeholder?: string }) {
  return (
    <select id={id} aria-label={ariaLabel} className={`input ${className}`} value={value ?? ''} onChange={(e) => onChange((e.target.value || null) as T | null)}>
      <option value="">{placeholder ?? '—'}</option>
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function TriState({ value, onChange, id, ariaLabel }: { value: boolean | null; onChange: (v: boolean | null) => void; id?: string; ariaLabel?: string }) {
  return (
    <select id={id} aria-label={ariaLabel} className="input" value={value === null ? '' : value ? 'yes' : 'no'} onChange={(e) => onChange(e.target.value === '' ? null : e.target.value === 'yes')}>
      <option value="">Unknown</option>
      <option value="yes">Yes</option>
      <option value="no">No</option>
    </select>
  );
}

export function Field({ label, children, hint, htmlFor }: { label: string; children: ReactNode; hint?: string; htmlFor?: string }) {
  const autoId = useId();
  const id = htmlFor ?? autoId;
  return (
    <div>
      <label className="label" htmlFor={id}>
        {label}
      </label>
      {children}
      {hint && <p className="mt-1 text-[11px] text-slate-500">{hint}</p>}
    </div>
  );
}

export function Toggle({ checked, onChange, label, id }: { checked: boolean; onChange: (v: boolean) => void; label: string; id?: string }) {
  return (
    <label className="inline-flex cursor-pointer items-center gap-2 text-sm" htmlFor={id}>
      <input id={id} type="checkbox" role="switch" aria-checked={checked} className="peer sr-only" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="relative h-5 w-9 rounded-full bg-slate-300 transition peer-checked:bg-teal-600 peer-focus-visible:ring-2 peer-focus-visible:ring-teal-500">
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition ${checked ? 'left-[18px]' : 'left-0.5'}`} />
      </span>
      <span>{label}</span>
    </label>
  );
}

export function Badge({ children, tone = 'slate', title }: { children: ReactNode; tone?: 'slate' | 'teal' | 'navy' | 'amber' | 'red' | 'green'; title?: string }) {
  const tones = {
    slate: 'bg-slate-100 text-slate-700',
    teal: 'bg-teal-100 text-teal-800',
    navy: 'bg-navy-100 text-navy-800',
    amber: 'bg-amber-100 text-amber-800',
    red: 'bg-red-100 text-red-800',
    green: 'bg-green-100 text-green-800',
  };
  return (
    <span className={`badge ${tones[tone]}`} title={title}>
      {children}
    </span>
  );
}

export function Modal({ open, onClose, title, children, wide, footer }: { open: boolean; onClose: () => void; title: string; children: ReactNode; wide?: boolean; footer?: ReactNode }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-navy-950/60 p-3 sm:p-6" role="dialog" aria-modal="true" aria-label={title} onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`card my-4 flex max-h-[calc(100vh-2rem)] w-full flex-col ${wide ? 'max-w-6xl' : 'max-w-2xl'}`}>
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
          <h2 className="text-base font-semibold text-navy-900">{title}</h2>
          <button className="btn-secondary btn-sm" onClick={onClose} aria-label="Close dialog">
            ✕
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">{children}</div>
        {footer && <div className="border-t border-slate-200 px-4 py-3">{footer}</div>}
      </div>
    </div>
  );
}

export function Section({ title, children, defaultOpen = true, right, id }: { title: string; children: ReactNode; defaultOpen?: boolean; right?: ReactNode; id?: string }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="border-b border-slate-200" id={id}>
      <div className="flex items-center justify-between px-3 py-2">
        <button className="flex items-center gap-2 text-left" onClick={() => setOpen(!open)} aria-expanded={open}>
          <span className={`text-slate-400 transition ${open ? 'rotate-90' : ''}`} aria-hidden>
            ▸
          </span>
          <span className="section-title">{title}</span>
        </button>
        {right}
      </div>
      {open && <div className="px-3 pb-3">{children}</div>}
    </section>
  );
}

export function Stat({ label, value, sub, tone = 'default' }: { label: string; value: string; sub?: string; tone?: 'default' | 'accent' | 'muted' }) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">{label}</div>
      <div className={`truncate text-base font-semibold tabular-nums ${tone === 'accent' ? 'text-teal-300' : tone === 'muted' ? 'text-slate-300' : 'text-white'}`}>{value}</div>
      {sub && <div className="truncate text-[11px] text-slate-400">{sub}</div>}
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="card flex flex-col items-center justify-center gap-2 p-8 text-center text-sm text-slate-500">
      <div className="text-base font-medium text-slate-700">{title}</div>
      {children}
    </div>
  );
}
