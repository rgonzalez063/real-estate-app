import { useState } from 'react';
import { useAppStore } from '../../store/useAppStore';
import type { SubjectProperty } from '../../types';
import { validatePhotoUrl } from '../../engine/parse';
import { Field, TextArea, TextInput } from '../ui/primitives';
import { AttributeForm } from './AttributeForm';

export function SubjectForm({ subject, effectiveDate }: { subject: SubjectProperty; effectiveDate: string | null }) {
  const setSubjectAttrs = useAppStore((s) => s.setSubjectAttrs);
  const setSubjectNotes = useAppStore((s) => s.setSubjectNotes);
  const setSubjectPhotos = useAppStore((s) => s.setSubjectPhotos);
  const setEffectiveDate = useAppStore((s) => s.setEffectiveDate);
  const [photoDraft, setPhotoDraft] = useState('');
  const [photoError, setPhotoError] = useState<string | null>(null);

  return (
    <div className="space-y-4">
      <Field label="Appraisal effective date" htmlFor="subject-effective-date" hint="Drives market conditions (time) adjustments.">
        <TextInput id="subject-effective-date" type="date" value={effectiveDate} onChange={setEffectiveDate} />
      </Field>
      {subject.sourceFile && (
        <div className="rounded-md border border-slate-200 bg-slate-50 p-2 text-[11px] text-slate-600">
          <div>
            Imported from <span className="font-medium">{subject.sourceFile}</span>. Edits below override the imported values.
          </div>
          {subject.parseIssues && subject.parseIssues.length > 0 && (
            <ul className="mt-1 list-disc pl-4 text-amber-800">
              {subject.parseIssues.map((p, i) => (
                <li key={i}>
                  {p.field}: {p.message}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      <AttributeForm attrs={subject.attrs} onChange={setSubjectAttrs} mode="subject" idPrefix="subject" />
      <div>
        <div className="section-title mb-2">Photos (optional)</div>
        {subject.photoUrls.length > 0 && (
          <ul className="mb-2 space-y-1">
            {subject.photoUrls.map((u) => (
              <li key={u} className="flex items-center gap-2 text-xs">
                <img src={u} alt="Subject" className="h-10 w-14 rounded object-cover" />
                <span className="min-w-0 flex-1 truncate text-slate-600">{u}</span>
                <button className="btn-danger btn-sm" onClick={() => setSubjectPhotos(subject.photoUrls.filter((x) => x !== u))} aria-label={`Remove photo ${u}`}>
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
        <div className="flex gap-2">
          <TextInput ariaLabel="Photo URL" value={photoDraft} onChange={(v) => setPhotoDraft(v ?? '')} placeholder="https://… image URL" />
          <button
            className="btn-secondary btn-sm"
            onClick={() => {
              const r = validatePhotoUrl(photoDraft);
              if (!r.value) {
                setPhotoError(r.issue ?? 'Enter an http(s) image URL');
                return;
              }
              setPhotoError(null);
              setSubjectPhotos([...subject.photoUrls, r.value]);
              setPhotoDraft('');
            }}
          >
            Add
          </button>
        </div>
        {photoError && <p className="mt-1 text-[11px] text-red-700">{photoError}</p>}
      </div>
      <Field label="Appraiser notes" htmlFor="subject-notes">
        <TextArea id="subject-notes" value={subject.notes} onChange={setSubjectNotes} rows={3} />
      </Field>
    </div>
  );
}
