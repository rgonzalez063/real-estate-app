import Anthropic from '@anthropic-ai/sdk';
import { betaZodOutputFormat } from '@anthropic-ai/sdk/helpers/beta/zod';
import { ExtractionResultSchema, type ExtractionResult } from './schema';

export const EXTRACTION_MODELS = [
  { id: 'claude-opus-5-5', label: 'Claude Opus 5.5 (default)' },
  { id: 'claude-sonnet-5-5', label: 'Claude Sonnet 5.5' },
  { id: 'claude-haiku-5-5', label: 'Claude Haiku 5.5 (fastest, lowest cost)' },
] as const;

export type ExtractionModel = (typeof EXTRACTION_MODELS)[number]['id'];

export interface DocumentInput {
  name: string;
  mime: 'application/pdf' | 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp';
  base64: string;
}

const SYSTEM = `You are a data-entry assistant for a residential real-estate appraiser. You read MLS listing sheets, closed-sale reports, public-record printouts, and appraisal workfile documents, and you transcribe property facts into a fixed schema.

Rules:
- Transcribe only what the document states. Never estimate, infer, or fill in a value that is not written (for example, do not compute age from year built, do not guess lot units, do not invent coordinates or photo URLs).
- Copy values as they appear, including units and currency symbols (e.g. "$412,000", "0.21 acres", "2,150 sq ft"). Dates may be normalised to YYYY-MM-DD when the day and month are unambiguous; otherwise copy the text as written.
- Keep full baths and half baths separate. If the document only gives a combined figure such as "3.1" or "2.5", put it in bathsCombined and leave fullBaths/halfBaths null.
- status is the listing status text (Closed, Sold, Active, Pending, Expired...). salePrice is the closed/sold price only; listPrice is the asking price.
- lotSize must include its units when the document shows them.
- pool / waterfront / waterView: "yes", "no", or null when not stated.
- One record per property. A document can contain several properties (a comp sheet, a multi-listing report); create one record for each. Mark a record "subject" only when the document explicitly identifies it as the subject property of an appraisal; otherwise "comparable" for sales/listings and "unknown" when unclear.
- For every non-null field, add an evidence entry quoting the exact text you read it from.
- confidence: "high" when the document is a clean, machine-readable listing; "medium" when some values were hard to read; "low" when the document is partially illegible or ambiguous.
- Put anything the appraiser should double-check in notes.`;

export class ExtractionRefusal extends Error {}

export async function extractWithClaude(opts: { apiKey: string; model: ExtractionModel; document: DocumentInput; hint: 'auto' | 'subject' | 'comparables' }): Promise<ExtractionResult & { model: string }> {
  const client = new Anthropic({ apiKey: opts.apiKey, dangerouslyAllowBrowser: true, maxRetries: 2 });
  const hintText =
    opts.hint === 'subject'
      ? 'The appraiser says this document describes the SUBJECT property of the appraisal.'
      : opts.hint === 'comparables'
        ? 'The appraiser says this document contains COMPARABLE sales or listings.'
        : 'Decide from the document whether each property is the subject, a comparable, or unknown.';
  const block: Anthropic.Beta.BetaContentBlockParam =
    opts.document.mime === 'application/pdf'
      ? { type: 'document', source: { type: 'base64', media_type: 'application/pdf', data: opts.document.base64 }, title: opts.document.name }
      : { type: 'image', source: { type: 'base64', media_type: opts.document.mime, data: opts.document.base64 } };

  const response = await client.beta.messages.parse({
    model: opts.model,
    max_tokens: 16000,
    betas: ['server-side-fallback-2026-07-01'],
    fallbacks: 'default',
    system: SYSTEM,
    output_config: { effort: 'medium', format: betaZodOutputFormat(ExtractionResultSchema) },
    messages: [
      {
        role: 'user',
        content: [block, { type: 'text', text: `Document file name: ${opts.document.name}. ${hintText} Extract every property record in this document.` }],
      },
    ],
  });

  if (response.stop_reason === 'refusal') {
    throw new ExtractionRefusal(`The model declined to process "${opts.document.name}"${response.stop_details?.explanation ? `: ${response.stop_details.explanation}` : '.'}`);
  }
  if (response.stop_reason === 'max_tokens') {
    throw new Error(`The response for "${opts.document.name}" was cut off (too long). Try a smaller document or fewer pages.`);
  }
  const parsed = response.parsed_output;
  if (!parsed) throw new Error(`The model returned output that did not match the expected schema for "${opts.document.name}".`);
  return { ...parsed, model: response.model };
}

/** Map common API failures to plain-language messages. */
export function describeApiError(err: unknown): string {
  if (err instanceof ExtractionRefusal) return err.message;
  if (err instanceof Anthropic.AuthenticationError) return 'The API key was rejected. Check the key in AI settings.';
  if (err instanceof Anthropic.PermissionDeniedError) return 'This API key does not have permission to use the selected model.';
  if (err instanceof Anthropic.RateLimitError) return 'Rate limit reached. Wait a moment and try again.';
  if (err instanceof Anthropic.BadRequestError) return `The request was rejected: ${err.message}`;
  if (err instanceof Anthropic.APIConnectionError) return 'Could not reach the Claude API. Check your internet connection.';
  if (err instanceof Anthropic.APIError) return `Claude API error (${err.status ?? 'unknown'}): ${err.message}`;
  return err instanceof Error ? err.message : String(err);
}
