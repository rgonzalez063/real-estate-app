import type { ExtractionModel } from '../engine/extract/claude';

export interface AiSettings {
  apiKey: string;
  model: ExtractionModel;
  remember: boolean;
}

const KEY = 'gac.appraisal.aiSettings';
let memory: AiSettings = { apiKey: '', model: 'claude-opus-5-5', remember: false };

/** API key and model for document extraction. Stored only in this browser (localStorage) when "remember" is on. */
export function loadAiSettings(): AiSettings {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<AiSettings>;
      memory = { apiKey: parsed.apiKey ?? '', model: (parsed.model as ExtractionModel) ?? 'claude-opus-5-5', remember: true };
    }
  } catch {
    /* storage unavailable */
  }
  return memory;
}

export function saveAiSettings(s: AiSettings): void {
  memory = s;
  try {
    if (s.remember) localStorage.setItem(KEY, JSON.stringify({ apiKey: s.apiKey, model: s.model }));
    else localStorage.removeItem(KEY);
  } catch {
    /* storage unavailable */
  }
}
