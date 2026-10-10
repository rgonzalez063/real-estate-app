import fs from 'node:fs';
const src = 'release/index.html';
const dst = 'release/Gonzalez-Appraisal-Dashboard.html';
fs.renameSync(src, dst);
for (const f of fs.readdirSync('release')) if (f !== 'Gonzalez-Appraisal-Dashboard.html') fs.rmSync(`release/${f}`, { recursive: true, force: true });
console.log(`Wrote ${dst} (${(fs.statSync(dst).size / 1024).toFixed(0)} KB)`);
