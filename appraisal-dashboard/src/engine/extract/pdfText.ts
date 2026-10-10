import * as pdfjs from 'pdfjs-dist';
import workerSource from 'pdfjs-dist/build/pdf.worker.min.mjs?raw';

let workerReady = false;
function ensureWorker() {
  if (workerReady) return;
  const blob = new Blob([workerSource], { type: 'text/javascript' });
  pdfjs.GlobalWorkerOptions.workerSrc = URL.createObjectURL(blob);
  workerReady = true;
}

/** Extract text from a PDF entirely in the browser. Scanned (image-only) PDFs return little or no text. */
export async function extractPdfText(bytes: ArrayBuffer): Promise<{ text: string; pages: number }> {
  ensureWorker();
  const doc = await pdfjs.getDocument({ data: new Uint8Array(bytes) }).promise;
  const parts: string[] = [];
  for (let p = 1; p <= doc.numPages; p++) {
    const page = await doc.getPage(p);
    const content = await page.getTextContent();
    let line = '';
    let lastY: number | null = null;
    for (const item of content.items) {
      if (!('str' in item)) continue;
      const y = item.transform[5];
      if (lastY !== null && Math.abs(y - lastY) > 2) {
        parts.push(line);
        line = '';
      }
      line += (line && !line.endsWith(' ') ? ' ' : '') + item.str;
      lastY = y;
    }
    if (line) parts.push(line);
    parts.push('');
  }
  return { text: parts.join('\n'), pages: doc.numPages };
}
