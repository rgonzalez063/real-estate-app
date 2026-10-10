import { useEffect, useMemo, useState } from 'react';
import { useAppStore } from '../../store/useAppStore';
import type { Comparable, PropertyAttributes } from '../../types';
import { ATTR_LABELS, FIELD_BY_KEY } from '../../engine/fields';
import type { FieldKey } from '../../types';

const fieldLabel = (k: string) => ATTR_LABELS[k as keyof typeof ATTR_LABELS] ?? FIELD_BY_KEY[k as FieldKey]?.label ?? k;
import { extractedToAttributes, type ExtractedRecord } from '../../engine/extract/schema';
import { parseListingText } from '../../engine/extract/local';
import { describeApiError, EXTRACTION_MODELS, extractWithClaude, type DocumentInput, type ExtractionModel } from '../../engine/extract/claude';
import { loadAiSettings, saveAiSettings, type AiSettings } from '../../store/aiSettings';
import { Badge, Field, Modal, Select, TextInput } from '../ui/primitives';
import { AttributeForm } from '../subject/AttributeForm';

type Method = 'ai' | 'local';
type Target = 'comparable' | 'subject' | 'skip';
type Step = 'setup' | 'running' | 'review';

interface DocFile {
  id: string;
  file: File;
  mime: DocumentInput['mime'] | null;
}

interface ReviewRecord {
  id: string;
  file: string;
  record: ExtractedRecord;
  method: Method;
  model?: string;
  attrs: PropertyAttributes;
  raw: Record<string, string>;
  issues: { field: string; raw: string; message: string }[];
  target: Target;
  text?: string;
}

const MIME: Record<string, DocumentInput['mime']> = { 'application/pdf': 'application/pdf', 'image/jpeg': 'image/jpeg', 'image/png': 'image/png', 'image/gif': 'image/gif', 'image/webp': 'image/webp' };

function sniffMime(f: File): DocumentInput['mime'] | null {
  if (MIME[f.type]) return MIME[f.type];
  const ext = f.name.toLowerCase().split('.').pop();
  if (ext === 'pdf') return 'application/pdf';
  if (ext === 'jpg' || ext === 'jpeg') return 'image/jpeg';
  if (ext === 'png') return 'image/png';
  if (ext === 'gif') return 'image/gif';
  if (ext === 'webp') return 'image/webp';
  return null;
}

async function toBase64(file: File): Promise<string> {
  const buf = await file.arrayBuffer();
  let binary = '';
  const bytes = new Uint8Array(buf);
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  return btoa(binary);
}

export function DocumentImport() {
  const { ui, setUi, addComparables, setSubjectFromImport, toast } = useAppStore();
  const [files, setFiles] = useState<DocFile[]>([]);
  const [method, setMethod] = useState<Method>('ai');
  const [hint, setHint] = useState<'auto' | 'subject' | 'comparables'>('auto');
  const [ai, setAi] = useState<AiSettings>(() => loadAiSettings());
  const [showKey, setShowKey] = useState(false);
  const [step, setStep] = useState<Step>('setup');
  const [progress, setProgress] = useState<string[]>([]);
  const [records, setRecords] = useState<ReviewRecord[]>([]);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (ui.docImportOpen) setAi(loadAiSettings());
  }, [ui.docImportOpen]);

  const reset = () => {
    setFiles([]);
    setStep('setup');
    setProgress([]);
    setRecords([]);
    setHint('auto');
  };
  const close = () => {
    setUi({ docImportOpen: false });
    reset();
  };

  const addFiles = (list: FileList | File[]) => {
    const next: DocFile[] = Array.from(list).map((f) => ({ id: `${f.name}-${f.size}-${f.lastModified}`, file: f, mime: sniffMime(f) }));
    const bad = next.filter((f) => !f.mime);
    if (bad.length) toast(`Skipped ${bad.length} unsupported file${bad.length > 1 ? 's' : ''} (PDF, JPG, PNG, GIF, WEBP only).`, 'error');
    setFiles((prev) => [...prev, ...next.filter((f) => f.mime && !prev.some((p) => p.id === f.id))]);
  };

  const canRun = files.length > 0 && (method === 'local' || ai.apiKey.trim().length > 0);

  const run = async () => {
    setStep('running');
    setProgress([]);
    const out: ReviewRecord[] = [];
    const log = (m: string) => setProgress((p) => [...p, m]);
    if (method === 'ai') saveAiSettings(ai);
    for (const f of files) {
      try {
        if (method === 'local') {
          if (f.mime !== 'application/pdf') {
            log(`${f.file.name}: local scan only reads PDFs with a text layer. Use AI extraction for images.`);
            continue;
          }
          log(`${f.file.name}: reading text…`);
          const { extractPdfText } = await import('../../engine/extract/pdfText');
          const { text, pages } = await extractPdfText(await f.file.arrayBuffer());
          if (text.trim().length < 20) {
            log(`${f.file.name}: no text layer found (${pages} page${pages === 1 ? '' : 's'}; probably a scan). Use AI extraction.`);
            continue;
          }
          const rec = parseListingText(text, f.file.name);
          const n = extractedToAttributes(rec.fields, f.file.name);
          out.push({ id: `${f.id}-0`, file: f.file.name, record: rec, method: 'local', attrs: n.attrs, raw: n.raw, issues: n.issues, target: hint === 'subject' ? 'subject' : 'comparable', text });
          log(`${f.file.name}: ${rec.evidence.length} labelled value${rec.evidence.length === 1 ? '' : 's'} found on ${pages} page${pages === 1 ? '' : 's'}.`);
        } else {
          log(`${f.file.name}: sending to ${ai.model}…`);
          const base64 = await toBase64(f.file);
          const result = await extractWithClaude({ apiKey: ai.apiKey.trim(), model: ai.model, document: { name: f.file.name, mime: f.mime as DocumentInput['mime'], base64 }, hint });
          result.records.forEach((rec, i) => {
            const n = extractedToAttributes(rec.fields, f.file.name);
            const target: Target = hint === 'subject' || rec.recordType === 'subject' ? 'subject' : 'comparable';
            out.push({ id: `${f.id}-${i}`, file: f.file.name, record: rec, method: 'ai', model: result.model, attrs: n.attrs, raw: n.raw, issues: n.issues, target });
          });
          log(`${f.file.name}: ${result.records.length} record${result.records.length === 1 ? '' : 's'} extracted. ${result.documentSummary}`);
        }
      } catch (err) {
        log(`${f.file.name}: ${describeApiError(err)}`);
      }
    }
    // Only one record can be the subject.
    let subjectSeen = false;
    for (const r of out) {
      if (r.target === 'subject') {
        if (subjectSeen) r.target = 'comparable';
        subjectSeen = true;
      }
    }
    setRecords(out);
    setStep('review');
  };

  const setTarget = (id: string, target: Target) =>
    setRecords((prev) => prev.map((r) => (r.id === id ? { ...r, target } : target === 'subject' && r.target === 'subject' ? { ...r, target: 'comparable' } : r)));
  const patchAttrs = (id: string, patch: Partial<PropertyAttributes>) => setRecords((prev) => prev.map((r) => (r.id === id ? { ...r, attrs: { ...r.attrs, ...patch } } : r)));

  const finish = () => {
    const comps: Comparable[] = [];
    let subject: ReviewRecord | null = null;
    for (const r of records) {
      if (r.target === 'skip') continue;
      if (r.target === 'subject') {
        subject = r;
        continue;
      }
      comps.push({
        id: `comp_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
        attrs: r.attrs,
        raw: r.raw,
        sourceFile: r.file,
        sourceRowIndex: 0,
        parseIssues: r.issues,
        included: false,
        selectionNote: '',
        weight: 1,
        overrides: {},
        resolutions: {},
        gridNote: '',
        extraction: { method: r.method, model: r.model, file: r.file, confidence: r.record.confidence, evidence: r.record.evidence, notes: r.record.notes },
      });
    }
    if (comps.length) addComparables(comps);
    if (subject) setSubjectFromImport(subject.attrs, { file: subject.file, row: subject.raw, issues: subject.issues });
    toast(`Added ${comps.length} comparable${comps.length === 1 ? '' : 's'}${subject ? ' and updated the subject' : ''} from documents.`);
    close();
  };

  const counts = useMemo(() => ({ comps: records.filter((r) => r.target === 'comparable').length, subject: records.some((r) => r.target === 'subject') }), [records]);

  return (
    <Modal
      open={ui.docImportOpen}
      onClose={close}
      title="Import from PDF or image documents"
      wide
      footer={
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-slate-500">
            {step === 'setup' && `${files.length} file${files.length === 1 ? '' : 's'} selected`}
            {step === 'running' && 'Working…'}
            {step === 'review' && `${records.length} record${records.length === 1 ? '' : 's'} to review`}
          </span>
          <div className="flex gap-2">
            {step === 'review' && (
              <button className="btn-secondary" onClick={() => setStep('setup')}>
                Back
              </button>
            )}
            {step === 'setup' && (
              <button className="btn-primary" disabled={!canRun} onClick={() => void run()}>
                {method === 'ai' ? 'Extract with Claude' : 'Scan text locally'}
              </button>
            )}
            {step === 'review' && (
              <button className="btn-primary" disabled={!counts.comps && !counts.subject} onClick={finish}>
                Add {counts.comps} comparable{counts.comps === 1 ? '' : 's'}
                {counts.subject ? ' + subject' : ''}
              </button>
            )}
          </div>
        </div>
      }
    >
      {step === 'setup' && (
        <div className="space-y-4">
          <div
            className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 text-center ${dragging ? 'border-teal-500 bg-teal-50' : 'border-slate-300'}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files); }}
          >
            <p className="text-sm font-medium text-slate-700">Drop MLS sheets, closed-sale reports, or photos of listings</p>
            <p className="mb-3 text-xs text-slate-500">PDF, JPG, PNG, GIF, WEBP. Several properties in one document are fine.</p>
            <label className="btn-primary cursor-pointer">
              Choose files
              <input type="file" accept=".pdf,application/pdf,image/*" multiple className="sr-only" onChange={(e) => e.target.files && addFiles(e.target.files)} />
            </label>
          </div>
          {files.length > 0 && (
            <ul className="space-y-1 text-sm">
              {files.map((f) => (
                <li key={f.id} className="flex items-center justify-between rounded border border-slate-200 px-3 py-1.5">
                  <span>
                    {f.file.name} <span className="text-xs text-slate-500">· {(f.file.size / 1024).toFixed(0)} KB · {f.mime}</span>
                  </span>
                  <button className="btn-danger btn-sm" onClick={() => setFiles(files.filter((x) => x.id !== f.id))}>
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          )}

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <fieldset className={`rounded-md border p-3 ${method === 'ai' ? 'border-teal-400 bg-teal-50/40' : 'border-slate-200'}`}>
              <label className="flex items-start gap-2">
                <input type="radio" name="doc-method" checked={method === 'ai'} onChange={() => setMethod('ai')} className="mt-1" />
                <span>
                  <span className="block text-sm font-medium text-slate-800">AI extraction with Claude</span>
                  <span className="block text-xs text-slate-600">Reads PDFs and photos, including scanned pages, and quotes the text each value came from. Uses your own Anthropic API key; the document is sent to the Claude API for this request only.</span>
                </span>
              </label>
              {method === 'ai' && (
                <div className="mt-3 space-y-2">
                  <Field label="Anthropic API key" htmlFor="ai-key" hint="Create one at console.anthropic.com. Usage is billed to your account.">
                    <div className="flex gap-1">
                      <TextInput id="ai-key" type={showKey ? 'text' : 'password'} value={ai.apiKey} onChange={(v) => setAi({ ...ai, apiKey: v ?? '' })} placeholder="sk-ant-…" className="font-mono" />
                      <button className="btn-secondary btn-sm" onClick={() => setShowKey(!showKey)} aria-label={showKey ? 'Hide key' : 'Show key'}>
                        {showKey ? 'Hide' : 'Show'}
                      </button>
                    </div>
                  </Field>
                  <Field label="Model" htmlFor="ai-model">
                    <Select<ExtractionModel> id="ai-model" value={ai.model} onChange={(v) => v && setAi({ ...ai, model: v })} options={EXTRACTION_MODELS.map((m) => ({ value: m.id, label: m.label }))} />
                  </Field>
                  <label className="flex items-center gap-2 text-xs text-slate-600">
                    <input type="checkbox" checked={ai.remember} onChange={(e) => setAi({ ...ai, remember: e.target.checked })} /> Remember the key on this device (stored in this browser only)
                  </label>
                </div>
              )}
            </fieldset>
            <fieldset className={`rounded-md border p-3 ${method === 'local' ? 'border-teal-400 bg-teal-50/40' : 'border-slate-200'}`}>
              <label className="flex items-start gap-2">
                <input type="radio" name="doc-method" checked={method === 'local'} onChange={() => setMethod('local')} className="mt-1" />
                <span>
                  <span className="block text-sm font-medium text-slate-800">Local text scan (offline, no account)</span>
                  <span className="block text-xs text-slate-600">Reads the PDF's text layer in your browser and picks up values next to recognised labels (Sold Price, Year Built, GLA…). One property per document. Does not work on scanned images.</span>
                </span>
              </label>
            </fieldset>
          </div>

          <Field label="What do these documents describe?" htmlFor="doc-hint">
            <Select<'auto' | 'subject' | 'comparables'> id="doc-hint" value={hint} onChange={(v) => v && setHint(v)} options={[{ value: 'auto', label: 'Let the extractor decide per record' }, { value: 'comparables', label: 'Comparable sales / listings' }, { value: 'subject', label: 'The subject property' }]} />
          </Field>
          <p className="text-[11px] text-slate-500">Nothing is added to the project until you review each extracted record on the next screen. Values are copied from the document, never estimated; anything the document does not state stays unknown.</p>
        </div>
      )}

      {step === 'running' && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm text-slate-700">
            <span className="inline-block h-3 w-3 animate-pulse rounded-full bg-teal-500" aria-hidden /> Processing {files.length} file{files.length === 1 ? '' : 's'}…
          </div>
          <ul className="space-y-1 text-xs text-slate-600">
            {progress.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        </div>
      )}

      {step === 'review' && (
        <div className="space-y-3">
          {progress.length > 0 && (
            <details className="rounded-md border border-slate-200 bg-slate-50 p-2 text-xs text-slate-600">
              <summary className="cursor-pointer font-medium">Processing log ({progress.length})</summary>
              <ul className="mt-1 space-y-0.5">
                {progress.map((p, i) => (
                  <li key={i}>{p}</li>
                ))}
              </ul>
            </details>
          )}
          {records.length === 0 && <p className="text-sm text-slate-600">No records were extracted. Check the log above, then go back and try AI extraction or a different file.</p>}
          {records.map((r) => (
            <ReviewCard key={r.id} r={r} onTarget={(t) => setTarget(r.id, t)} onAttrs={(p) => patchAttrs(r.id, p)} />
          ))}
        </div>
      )}
    </Modal>
  );
}

function ReviewCard({ r, onTarget, onAttrs }: { r: ReviewRecord; onTarget: (t: Target) => void; onAttrs: (p: Partial<PropertyAttributes>) => void }) {
  const [edit, setEdit] = useState(false);
  const [showEvidence, setShowEvidence] = useState(false);
  const [showText, setShowText] = useState(false);
  const filled = Object.entries(r.raw);
  const tone = r.record.confidence === 'high' ? 'green' : r.record.confidence === 'medium' ? 'amber' : 'slate';
  return (
    <div className={`rounded-md border p-3 ${r.target === 'skip' ? 'border-slate-200 opacity-60' : r.target === 'subject' ? 'border-navy-400 bg-navy-50/40' : 'border-teal-300'}`}>
      <div className="flex flex-wrap items-start gap-2">
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold text-navy-900">{r.attrs.address ?? r.record.label ?? 'Unnamed record'}</div>
          <div className="flex flex-wrap items-center gap-1 text-[11px] text-slate-500">
            <span>{r.file}</span>
            <Badge tone={r.method === 'ai' ? 'teal' : 'slate'}>{r.method === 'ai' ? `AI · ${r.model ?? ''}` : 'Local scan'}</Badge>
            <Badge tone={tone}>{r.record.confidence} confidence</Badge>
            <Badge tone="slate">{r.record.recordType}</Badge>
            <span>{filled.length} fields read</span>
            {r.issues.length > 0 && <Badge tone="amber">{r.issues.length} parsing note{r.issues.length > 1 ? 's' : ''}</Badge>}
          </div>
        </div>
        <div role="radiogroup" aria-label="Use this record as" className="flex gap-1 text-xs">
          {(['comparable', 'subject', 'skip'] as Target[]).map((t) => (
            <button key={t} role="radio" aria-checked={r.target === t} className={`rounded-full border px-2.5 py-1 font-medium ${r.target === t ? 'border-navy-800 bg-navy-800 text-white' : 'border-slate-300 bg-white text-slate-600'}`} onClick={() => onTarget(t)}>
              {t === 'comparable' ? 'Comparable' : t === 'subject' ? 'Subject' : 'Skip'}
            </button>
          ))}
        </div>
      </div>
      {r.record.notes && <p className="mt-1 text-[11px] text-slate-600">{r.record.notes}</p>}
      <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-0.5 text-[11px] sm:grid-cols-3 lg:grid-cols-4">
        {filled.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="truncate text-slate-500">{fieldLabel(k)}</dt>
            <dd className="truncate font-medium text-slate-800" title={v}>
              {v}
            </dd>
          </div>
        ))}
      </dl>
      {r.issues.length > 0 && (
        <ul className="mt-1 list-disc pl-5 text-[11px] text-amber-800">
          {r.issues.map((i, idx) => (
            <li key={idx}>
              {fieldLabel(i.field)}: {i.message}
            </li>
          ))}
        </ul>
      )}
      <div className="mt-2 flex flex-wrap gap-1">
        <button className="btn-secondary btn-sm" onClick={() => setEdit(!edit)} aria-expanded={edit}>
          {edit ? 'Hide editor' : 'Edit values'}
        </button>
        {r.record.evidence.length > 0 && (
          <button className="btn-secondary btn-sm" onClick={() => setShowEvidence(!showEvidence)} aria-expanded={showEvidence}>
            {showEvidence ? 'Hide evidence' : `Evidence (${r.record.evidence.length})`}
          </button>
        )}
        {r.text && (
          <button className="btn-secondary btn-sm" onClick={() => setShowText(!showText)} aria-expanded={showText}>
            {showText ? 'Hide document text' : 'Document text'}
          </button>
        )}
      </div>
      {showEvidence && (
        <ul className="mt-2 max-h-48 space-y-0.5 overflow-auto rounded border border-slate-200 bg-slate-50 p-2 text-[11px]">
          {r.record.evidence.map((e, i) => (
            <li key={i}>
              <span className="font-medium text-slate-700">{fieldLabel(e.field)}:</span> <span className="text-slate-600">“{e.quote}”</span>
            </li>
          ))}
        </ul>
      )}
      {showText && <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap rounded border border-slate-200 bg-slate-50 p-2 text-[10px] text-slate-700">{r.text}</pre>}
      {edit && (
        <div className="mt-2 rounded border border-slate-200 bg-white p-2">
          <AttributeForm attrs={r.attrs} onChange={onAttrs} mode={r.target === 'subject' ? 'subject' : 'comp'} idPrefix={`doc-${r.id}`} />
        </div>
      )}
    </div>
  );
}
