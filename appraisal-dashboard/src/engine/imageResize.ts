/** Resize a dropped image in the browser and return a compact JPEG data URL for local storage. */
export async function fileToResizedDataUrl(file: File, maxWidth = 480, quality = 0.75): Promise<string> {
  if (!file.type.startsWith('image/')) throw new Error('Not an image file');
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, maxWidth / bitmap.width);
  const w = Math.round(bitmap.width * scale);
  const h = Math.round(bitmap.height * scale);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas unavailable');
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();
  return canvas.toDataURL('image/jpeg', quality);
}
