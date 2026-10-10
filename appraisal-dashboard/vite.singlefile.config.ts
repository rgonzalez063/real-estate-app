import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { viteSingleFile } from 'vite-plugin-singlefile';

/** Builds one self-contained HTML file (all JS/CSS inlined) that runs from disk by double-clicking. */
export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: {
    outDir: 'release',
    emptyOutDir: true,
    cssCodeSplit: false,
    assetsInlineLimit: 100_000_000,
    chunkSizeWarningLimit: 5000,
    rollupOptions: { output: { inlineDynamicImports: true } },
  },
});
