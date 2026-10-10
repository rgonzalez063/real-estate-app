import { useCallback, useMemo, useState } from 'react';
import Papa from 'papaparse';
import { useAppStore, activeScenarioOf } from '../../store/useAppStore';
import type { Comparable, FieldKey, MappingOptions, MappingPreset } from '../../types';
import { FIELD_DEFS, ATTR_LABELS } from '../../engine/fields';
import { coverage, defaultMappingOptions, findDuplicates, normalizeRows, RECOMMENDED_FIELDS, suggestMapping, type DuplicateGroup } from '../../engine/mapping';
import { makeId } from '../../engine/ids';
import { fmtMoney } from '../../engine/format';
import { Badge, Modal, Select } from '../ui/primitives';

interface ParsedFile {
  id: string;
  name: string;
  headers: string[];
  rows: Record<string, string>[];
  errors: string[];
  mapping: Record<string, FieldKey>;
  options: MappingOptions;
}

type Step = 'files' | 'mapping' | 'preview' | 'duplicates';

const GROUP_LABEL: Record<string, string> = { identity: 'Identity', transaction: 'Transaction', area: 'Area', site: 'Site', rooms: 'Rooms', age: 'Age', parking: 'Parking', ratings: 'Ratings', location: 'Location', other: 'Other' };

function parseFile(file: File): Promise<ParsedFile> {
  return new Promise((resolve) => {
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: 'greedy',
      transformHeader: (h) => h.trim(),
      complete: (res) => {
        const headers = (res.meta.fields ?? []).filter((h) => h !== '');
        const rows = res.data.map((r) => {
          const out: Record<string, string> = {};
          for (const h of headers) out[h] = r[h] === undefined || r[h] === null ? '' : String(r[h]);
          return out;
        });
        resolve({
          id: makeId('file'),
          name: file.name,
          headers,
          rows,
          errors: res.errors.map((e) => `Row ${e.row !== undefined ? e.row + 2 : '?'}: ${e.message}`),
          mapping: suggestMapping(headers),
          options: defaultMappingOptions(),
        });
      },
      error: (err) => resolve({ id: makeId('file'), name: file.name, headers: [], rows: [], errors: [err.message], mapping: {}, options: defaultMappingOptions() }),
    });
  });
}

export function ImportWizard() {
  const { ui, setUi, addComparables, setSubjectFromImport, globalPresets, savePreset, deletePreset, toast } = useAppStore();
  const [target, setTarget] = useState<'comparables' | 'subject'>('comparables');
  const [subjectRowId, setSubjectRowId] = useState<string | null>(null);
  const scenario = useAppStore((s) => activeScenarioOf(s.project));
  const [files, setFiles] = useState<ParsedFile[]>([]);
  const [active, setActive] = useState(0);
  const [step, setStep] = useState<Step>('files');
  const [keep, setKeep] = useState<Record<string, boolean>>({});
  const [dragging, setDragging] = useState(false);

  const reset = useCallback(() => {
    setFiles([]);
    setActive(0);
    setStep('files');
    setKeep({});
    setTarget('comparables');
    setSubjectRowId(null);
  }, []);
  const close = useCallback(() => {
    setUi({ importOpen: false });
    reset();
  }, [setUi, reset]);

  const onFiles = async (list: FileList | File[]) => {
    const arr = Array.from(list).filter((f) => /\.(csv|txt|tsv)$/i.test(f.name) || f.type.includes('csv') || f.type.includes('text'));
    if (!arr.length) {
      toast('Please choose CSV files.', 'error');
      return;
    }
    const parsed = await Promise.all(arr.map(parseFile));
    setFiles((prev) => [...prev, ...parsed]);
    setStep('mapping');
  };

  const current = files[active];
  const updateFile = (id: string, patch: Partial<ParsedFile>) => setFiles((prev) => prev.map((f) => (f.id === id ? { ...f, ...patch } : f)));

  const normalized = useMemo(() => files.map((f) => normalizeRows(f.rows, f.mapping, f.options, f.name)), [files]);
  const currentNorm = normalized[active];

  const allNew: Comparable[] = useMemo(() => normalized.flatMap((n) => n.comps), [normalized]);
  const duplicateGroups: DuplicateGroup[] = useMemo(() => {
    const existing = scenario?.comparables ?? [];
    return findDuplicates([...existing, ...allNew]).filter((g) => g.compIds.some((id) => allNew.some((c) => c.id === id)));
  }, [allNew, scenario]);
  const existingIds = useMemo(() => new Set((scenario?.comparables ?? []).map((c) => c.id)), [scenario]);

  const blockers = (f: ParsedFile, idx: number): string[] => {
    const n = normalized[idx];
    const out: string[] = [];
    if (!n) return out;
    if (n.datesAmbiguous && f.options.dateOrder === 'auto') out.push('Some dates are ambiguous (e.g. 03/04/2024). Choose a day/month order.');
    if (n.lotNeedsUnits && !f.options.lotUnitsDefault) out.push('Lot sizes have no units. Choose square feet, acres, or auto-detect.');
    if (n.bathsNeedsConvention && !f.options.bathsConvention) out.push('A combined baths column is mapped. Confirm how to split it (or choose not to).');
    const vals = Object.values(f.mapping);
    const dup = vals.filter((v, i) => v && vals.indexOf(v) !== i);
    if (dup.length) out.push(`Fields mapped more than once: ${[...new Set(dup)].map((k) => ATTR_LABELS[k as keyof typeof ATTR_LABELS] ?? k).join(', ')}.`);
    return out;
  };
  const allBlockers = files.flatMap((f, i) => blockers(f, i).map((b) => `${f.name}: ${b}`));

  const finishSubject = () => {
    const row = allNew.find((c) => c.id === subjectRowId);
    if (!row) return;
    setSubjectFromImport(row.attrs, { file: row.sourceFile, row: row.raw, issues: row.parseIssues });
    toast(`Subject property updated from ${row.sourceFile} (row ${row.sourceRowIndex + 1}).`);
    close();
  };

  const finish = () => {
    const comps = allNew.filter((c) => keep[c.id] !== false);
    addComparables(comps);
    toast(`Imported ${comps.length} record${comps.length === 1 ? '' : 's'} from ${files.length} file${files.length === 1 ? '' : 's'}.`);
    close();
  };

  const stepIndex: Record<Step, number> = { files: 0, mapping: 1, preview: 2, duplicates: 3 };

  return (
    <Modal
      open={ui.importOpen}
      onClose={close}
      title={target === 'subject' ? 'Import subject property from CSV' : 'Import comparable CSV files'}
      wide
      footer={
        <div className="flex items-center justify-between gap-2">
          <ol className="flex gap-3 text-xs text-slate-500">
            {(['files', 'mapping', 'preview', 'duplicates'] as Step[]).map((s, i) => (
              <li key={s} className={step === s ? 'font-semibold text-teal-700' : stepIndex[step] > i ? 'text-slate-700' : ''}>
                {i + 1}. {s === 'files' ? 'Files' : s === 'mapping' ? 'Map fields' : s === 'preview' ? 'Preview' : 'Duplicates'}
              </li>
            ))}
          </ol>
          <div className="flex gap-2">
            {step !== 'files' && (
              <button className="btn-secondary" onClick={() => setStep(step === 'mapping' ? 'files' : step === 'preview' ? 'mapping' : 'preview')}>
                Back
              </button>
            )}
            {step === 'mapping' && (
              <button className="btn-primary" disabled={allBlockers.length > 0 || !files.length} onClick={() => setStep('preview')} title={allBlockers.join('\n')}>
                Preview
              </button>
            )}
            {step === 'preview' && target === 'comparables' && (
              <button className="btn-primary" onClick={() => setStep('duplicates')}>
                Check duplicates
              </button>
            )}
            {step === 'preview' && target === 'subject' && (
              <button className="btn-primary" disabled={!subjectRowId} onClick={finishSubject}>
                Use selected row as subject
              </button>
            )}
            {step === 'duplicates' && (
              <button className="btn-primary" onClick={finish}>
                Import {allNew.filter((c) => keep[c.id] !== false).length} records
              </button>
            )}
          </div>
        </div>
      }
    >
      {step === 'files' && (
        <div>
          <div
            className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-10 text-center ${dragging ? 'border-teal-500 bg-teal-50' : 'border-slate-300'}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); void onFiles(e.dataTransfer.files); }}
          >
            <p className="text-sm font-medium text-slate-700">Drop one or more CSV exports here</p>
            <p className="mb-3 text-xs text-slate-500">HARMLS / Matrix style exports are recognised. Headers are mapped on the next step; nothing is imported until you confirm.</p>
            <label className="btn-primary relative cursor-pointer">
              Choose files
              <input type="file" accept=".csv,text/csv,.txt,.tsv" multiple className="sr-only" onChange={(e) => e.target.files && void onFiles(e.target.files)} />
            </label>
          </div>
          {files.length > 0 && (
            <ul className="mt-3 space-y-1 text-sm">
              {files.map((f) => (
                <li key={f.id} className="flex items-center justify-between rounded border border-slate-200 px-3 py-1.5">
                  <span>
                    {f.name} <span className="text-xs text-slate-500">· {f.rows.length} rows · {f.headers.length} columns</span>
                  </span>
                  <button className="btn-danger btn-sm" onClick={() => setFiles(files.filter((x) => x.id !== f.id))}>
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          )}
          <fieldset className="mt-3 rounded-md border border-slate-200 p-3">
            <legend className="px-1 text-xs font-medium text-slate-600">Import as</legend>
            <div className="flex flex-wrap gap-4 text-sm">
              <label className="flex items-center gap-2">
                <input type="radio" name="csv-target" checked={target === 'comparables'} onChange={() => setTarget('comparables')} /> Comparable sales / listings <span className="text-xs text-slate-500">(added to the existing comps; import as many files as you like)</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="radio" name="csv-target" checked={target === 'subject'} onChange={() => setTarget('subject')} /> Subject property <span className="text-xs text-slate-500">(pick one row on the preview step)</span>
              </label>
            </div>
          </fieldset>
          {files.length > 0 && (
            <button className="btn-primary mt-3" onClick={() => setStep('mapping')}>
              Continue to mapping
            </button>
          )}
          <p className="mt-3 text-[11px] text-slate-500">Files are processed entirely in your browser. Original rows and file names are kept with each record.</p>
        </div>
      )}

      {step === 'mapping' && current && (
        <MappingStep
          files={files}
          active={active}
          setActive={setActive}
          file={current}
          onChange={(patch) => updateFile(current.id, patch)}
          blockers={blockers(current, active)}
          norm={currentNorm}
          presets={globalPresets}
          onSavePreset={async (name) => {
            await savePreset({ id: makeId('preset'), name, mapping: current.mapping, options: current.options });
            toast(`Preset "${name}" saved.`);
          }}
          onDeletePreset={(id) => void deletePreset(id)}
        />
      )}

      {step === 'preview' && (
        <div className="space-y-3">
          {files.map((f, i) => {
            const n = normalized[i];
            return (
              <div key={f.id} className="rounded-md border border-slate-200">
                <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 bg-slate-50 px-3 py-2 text-sm">
                  <span className="font-medium">{f.name}</span>
                  <Badge tone="slate">{n.comps.length} rows</Badge>
                  <Badge tone={n.totalIssues ? 'amber' : 'green'}>{n.totalIssues} parsing notes</Badge>
                  {f.errors.length > 0 && <Badge tone="red">{f.errors.length} CSV errors</Badge>}
                  <span className="ml-auto text-[11px] text-slate-500">
                    Closed {n.comps.filter((c) => c.attrs.status === 'closed').length} · Active {n.comps.filter((c) => c.attrs.status === 'active').length} · Pending {n.comps.filter((c) => c.attrs.status === 'pending').length} · Other/unknown {n.comps.filter((c) => c.attrs.status !== 'closed' && c.attrs.status !== 'active' && c.attrs.status !== 'pending').length}
                  </span>
                </div>
                {Object.keys(n.missingByField).length > 0 && (
                  <div className="px-3 py-1 text-[11px] text-slate-600">
                    Missing values (kept as unknown):{' '}
                    {Object.entries(n.missingByField)
                      .map(([k, v]) => `${ATTR_LABELS[k as keyof typeof ATTR_LABELS] ?? k} (${v})`)
                      .join(', ')}
                  </div>
                )}
                {f.errors.length > 0 && (
                  <ul className="list-disc px-6 py-1 text-[11px] text-red-700">
                    {f.errors.slice(0, 10).map((e, j) => (
                      <li key={j}>{e}</li>
                    ))}
                  </ul>
                )}
                <div className="max-h-64 overflow-auto">
                  <table className="w-full text-[11px]">
                    <thead className="sticky top-0 bg-white">
                      <tr className="text-left text-slate-500">
                        {target === 'subject' && <th className="px-2 py-1">Subject</th>}
                        <th className="px-2 py-1">#</th>
                        <th className="px-2 py-1">Address</th>
                        <th className="px-2 py-1">Status</th>
                        <th className="px-2 py-1">Sale price</th>
                        <th className="px-2 py-1">Sale date</th>
                        <th className="px-2 py-1">Area (SF)</th>
                        <th className="px-2 py-1">Lot (SF)</th>
                        <th className="px-2 py-1">Baths F/H</th>
                        <th className="px-2 py-1">Notes</th>
                      </tr>
                    </thead>
                    <tbody>
                      {n.comps.slice(0, 50).map((c) => (
                        <tr key={c.id} className={`border-t border-slate-100 ${subjectRowId === c.id ? 'bg-navy-50' : ''}`}>
                          {target === 'subject' && (
                            <td className="px-2 py-1">
                              <input type="radio" name="subject-row" checked={subjectRowId === c.id} onChange={() => setSubjectRowId(c.id)} aria-label={`Use row ${c.sourceRowIndex + 1} as subject`} />
                            </td>
                          )}
                          <td className="px-2 py-1 text-slate-400">{c.sourceRowIndex + 1}</td>
                          <td className="px-2 py-1">{c.attrs.address ?? <span className="text-slate-400">unknown</span>}</td>
                          <td className="px-2 py-1">{c.attrs.statusRaw ?? '—'}</td>
                          <td className="px-2 py-1 tabular-nums">{fmtMoney(c.attrs.salePrice)}</td>
                          <td className="px-2 py-1">{c.attrs.saleDate ?? '—'}</td>
                          <td className="px-2 py-1 tabular-nums">{c.attrs.areaAboveGrade ?? '—'}</td>
                          <td className="px-2 py-1 tabular-nums">{c.attrs.lotSizeSqft !== null ? Math.round(c.attrs.lotSizeSqft) : '—'}</td>
                          <td className="px-2 py-1">{c.attrs.fullBaths ?? '—'} / {c.attrs.halfBaths ?? '—'}</td>
                          <td className="px-2 py-1 text-amber-700" title={c.parseIssues.map((p) => `${p.field}: ${p.message}`).join('\n')}>
                            {c.parseIssues.length ? `${c.parseIssues.length} note${c.parseIssues.length > 1 ? 's' : ''}` : ''}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {n.comps.length > 50 && <p className="px-3 py-1 text-[11px] text-slate-500">Showing first 50 of {n.comps.length}.</p>}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {step === 'duplicates' && (
        <div className="space-y-3">
          {duplicateGroups.length === 0 ? (
            <p className="text-sm text-slate-600">No potential duplicates found by MLS number or address + transaction date. Ready to import {allNew.length} records.</p>
          ) : (
            <>
              <p className="text-sm text-slate-600">{duplicateGroups.length} potential duplicate group{duplicateGroups.length > 1 ? 's' : ''}. Untick any new record you do not want to keep. Records already in the scenario are shown for reference and are not changed.</p>
              {duplicateGroups.map((g) => (
                <div key={g.key} className="rounded-md border border-amber-200 bg-amber-50/40 p-2">
                  <div className="mb-1 text-xs font-medium text-amber-800">Matched by {g.reason === 'mls' ? 'MLS number' : 'address + date'}</div>
                  <ul className="space-y-1 text-xs">
                    {g.compIds.map((id) => {
                      const c = allNew.find((x) => x.id === id) ?? scenario?.comparables.find((x) => x.id === id);
                      if (!c) return null;
                      const isExisting = existingIds.has(id);
                      return (
                        <li key={id} className="flex items-center gap-2">
                          {isExisting ? <Badge tone="slate">Existing</Badge> : <input type="checkbox" checked={keep[id] !== false} onChange={(e) => setKeep({ ...keep, [id]: e.target.checked })} aria-label={`Keep ${c.attrs.address ?? id}`} />}
                          <span className="font-medium">{c.attrs.address ?? 'Unknown address'}</span>
                          <span className="text-slate-500">MLS {c.attrs.mlsNumber ?? '—'} · {c.attrs.saleDate ?? c.attrs.contractDate ?? 'no date'} · {fmtMoney(c.attrs.salePrice)} · {c.sourceFile} row {c.sourceRowIndex + 1}</span>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </Modal>
  );
}

function MappingStep({ files, active, setActive, file, onChange, blockers, norm, presets, onSavePreset, onDeletePreset }: { files: ParsedFile[]; active: number; setActive: (i: number) => void; file: ParsedFile; onChange: (p: Partial<ParsedFile>) => void; blockers: string[]; norm: ReturnType<typeof normalizeRows> | undefined; presets: MappingPreset[]; onSavePreset: (name: string) => void; onDeletePreset: (id: string) => void }) {
  const covered = coverage(file.mapping);
  const missingRecommended = RECOMMENDED_FIELDS.filter((k) => !covered.has(k));
  const fieldOptions = FIELD_DEFS.map((d) => ({ value: d.key, label: `${GROUP_LABEL[d.group]}: ${d.label}` }));
  const sample = (h: string) => file.rows.slice(0, 3).map((r) => r[h]).filter((v) => v !== '').join(' · ');
  const setMapping = (h: string, key: FieldKey | null) => {
    const next = { ...file.mapping };
    if (key) next[h] = key;
    else delete next[h];
    onChange({ mapping: next });
  };
  const opt = (p: Partial<MappingOptions>) => onChange({ options: { ...file.options, ...p } });
  const showDate = covered.has('saleDate') || covered.has('contractDate');
  const showLot = covered.has('lotSize') && !covered.has('lotSizeUnits');
  const showBaths = covered.has('bathsCombined');
  const [appliedPresetId, setAppliedPresetId] = useState<string | null>(null);
  const appliedPreset = presets.find((p) => p.id === appliedPresetId);
  return (
    <div className="space-y-3">
      {files.length > 1 && (
        <div className="flex flex-wrap gap-1" role="tablist">
          {files.map((f, i) => (
            <button key={f.id} role="tab" aria-selected={i === active} className={`rounded-full border px-3 py-1 text-xs ${i === active ? 'border-teal-600 bg-teal-600 text-white' : 'border-slate-300 bg-white text-slate-700'}`} onClick={() => setActive(i)}>
              {f.name}
            </button>
          ))}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="font-medium text-slate-700">{file.name}</span>
        <Badge tone="slate">{file.rows.length} rows</Badge>
        <Badge tone="slate">{Object.keys(file.mapping).length} of {file.headers.length} columns mapped</Badge>
        {missingRecommended.length > 0 && <Badge tone="amber">Not mapped: {missingRecommended.map((k) => ATTR_LABELS[k as keyof typeof ATTR_LABELS]).join(', ')}</Badge>}
        <div className="ml-auto flex items-center gap-1">
          <Select<string>
            ariaLabel="Apply mapping preset"
            value={appliedPresetId}
            onChange={(id) => {
              const p = presets.find((x) => x.id === id);
              setAppliedPresetId(p ? p.id : null);
              if (!p) return;
              const mapping: Record<string, FieldKey> = {};
              for (const h of file.headers) if (p.mapping[h]) mapping[h] = p.mapping[h];
              onChange({ mapping, options: { ...p.options } });
            }}
            options={presets.map((p) => ({ value: p.id, label: p.name }))}
            placeholder={presets.length ? 'Apply preset…' : 'No presets saved'}
            className="w-44"
          />
          <button className="btn-secondary btn-sm" onClick={() => { const n = window.prompt('Preset name', 'HARMLS export'); if (n) onSavePreset(n); }}>
            Save as preset
          </button>
          {appliedPreset && (
            <button className="btn-danger btn-sm" onClick={() => { if (window.confirm(`Delete preset "${appliedPreset.name}"?`)) { onDeletePreset(appliedPreset.id); setAppliedPresetId(null); } }}>
              Delete preset
            </button>
          )}
          <button className="btn-secondary btn-sm" onClick={() => onChange({ mapping: suggestMapping(file.headers) })}>
            Re-suggest
          </button>
        </div>
      </div>

      {(showDate || showLot || showBaths) && (
        <div className="grid grid-cols-1 gap-2 rounded-md border border-slate-200 bg-slate-50 p-2 text-xs sm:grid-cols-3">
          {showDate && (
            <div>
              <div className="label">Date order {norm?.datesAmbiguous && file.options.dateOrder === 'auto' && <Badge tone="amber">ambiguous</Badge>}</div>
              <select aria-label="Date order" className="input" value={file.options.dateOrder} onChange={(e) => opt({ dateOrder: e.target.value as MappingOptions['dateOrder'] })}>
                <option value="auto">Auto-detect (requires unambiguous values)</option>
                <option value="MDY">Month / Day / Year (US)</option>
                <option value="DMY">Day / Month / Year</option>
                <option value="YMD">Year / Month / Day</option>
              </select>
            </div>
          )}
          {showLot && (
            <div>
              <div className="label">Lot size units {norm?.lotNeedsUnits && !file.options.lotUnitsDefault && <Badge tone="amber">required</Badge>}</div>
              <select aria-label="Lot size units" className="input" value={file.options.lotUnitsDefault ?? ''} onChange={(e) => opt({ lotUnitsDefault: (e.target.value || null) as MappingOptions['lotUnitsDefault'] })}>
                <option value="">Choose…</option>
                <option value="sqft">Square feet</option>
                <option value="acres">Acres</option>
                <option value="auto">Auto (values under 100 = acres)</option>
              </select>
            </div>
          )}
          {showBaths && (
            <div>
              <div className="label">Combined baths convention {!file.options.bathsConvention && <Badge tone="amber">required</Badge>}</div>
              <select aria-label="Baths convention" className="input" value={file.options.bathsConvention ?? ''} onChange={(e) => opt({ bathsConvention: (e.target.value || null) as MappingOptions['bathsConvention'] })}>
                <option value="">Choose…</option>
                <option value="fullPointHalf">3.1 = 3 full + 1 half</option>
                <option value="decimal">3.5 = 3 full + 1 half (0.5 = half bath)</option>
                <option value="none">Do not split (keep baths unknown)</option>
              </select>
            </div>
          )}
        </div>
      )}

      {blockers.length > 0 && (
        <ul className="list-disc rounded-md border border-amber-200 bg-amber-50 px-6 py-2 text-xs text-amber-800">
          {blockers.map((b) => (
            <li key={b}>{b}</li>
          ))}
        </ul>
      )}

      <div className="max-h-[50vh] overflow-auto rounded-md border border-slate-200">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-slate-50 text-left text-slate-600">
            <tr>
              <th className="px-2 py-1.5">Source header</th>
              <th className="px-2 py-1.5">Sample values</th>
              <th className="px-2 py-1.5">Maps to</th>
            </tr>
          </thead>
          <tbody>
            {file.headers.map((h) => (
              <tr key={h} className="border-t border-slate-100">
                <td className="px-2 py-1 font-medium text-slate-800">{h}</td>
                <td className="max-w-xs truncate px-2 py-1 text-slate-500" title={sample(h)}>
                  {sample(h) || <span className="italic text-slate-400">empty</span>}
                </td>
                <td className="px-2 py-1">
                  <Select<FieldKey> ariaLabel={`Map ${h}`} value={file.mapping[h] ?? null} onChange={(v) => setMapping(h, v)} options={fieldOptions} placeholder="Ignore" className="!py-1" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-slate-500">Legacy labels such as GLA map to above-grade finished area. Keep full and half baths separate when the export provides them; use the combined-baths field only when it does not.</p>
    </div>
  );
}
