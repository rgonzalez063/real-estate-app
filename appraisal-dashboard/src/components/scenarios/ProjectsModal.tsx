import { useRef } from 'react';
import { useAppStore } from '../../store/useAppStore';
import { importProjectJson } from '../../engine/export';
import { Badge, Modal } from '../ui/primitives';

export function ProjectsModal() {
  const { ui, setUi, projects, project, openProject, createProject, loadDemo, removeProject, importProject, toast } = useAppStore();
  const fileRef = useRef<HTMLInputElement>(null);
  return (
    <Modal open={ui.projectsOpen} onClose={() => setUi({ projectsOpen: false })} title="Projects (stored locally in this browser)">
      <div className="mb-3 flex flex-wrap gap-2">
        <button className="btn-primary btn-sm" onClick={() => { const n = window.prompt('Project name', 'New appraisal'); if (n) createProject(n); }}>
          New project
        </button>
        <button className="btn-secondary btn-sm" onClick={() => fileRef.current?.click()}>
          Restore from backup (JSON)
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          className="hidden"
          aria-label="Restore project backup"
          onChange={async (e) => {
            const f = e.target.files?.[0];
            if (!f) return;
            try {
              const p = importProjectJson(await f.text());
              if (projects.some((x) => x.id === p.id) && !window.confirm('A project with this id already exists locally. Replace it?')) {
                p.id = `${p.id}_restored_${Date.now().toString(36)}`;
              }
              importProject(p);
              toast(`Restored "${p.name}".`);
            } catch (err) {
              toast((err as Error).message, 'error');
            } finally {
              e.target.value = '';
            }
          }}
        />
        <button className="btn-secondary btn-sm" onClick={loadDemo}>
          Load synthetic demo
        </button>
      </div>
      <p className="mb-2 text-[11px] text-slate-500">Projects are saved in your browser's IndexedDB. Nothing leaves this device unless you export a file. The demo project uses clearly labelled synthetic data.</p>
      {projects.length === 0 ? (
        <p className="text-sm text-slate-500">No saved projects yet.</p>
      ) : (
        <ul className="divide-y divide-slate-200 rounded-md border border-slate-200">
          {projects.map((p) => (
            <li key={p.id} className="flex items-center gap-2 px-3 py-2 text-sm">
              <div className="min-w-0 flex-1">
                <div className="truncate font-medium text-slate-800">
                  {p.name} {p.isDemo && <Badge tone="amber">Demo</Badge>} {project?.id === p.id && <Badge tone="teal">Open</Badge>}
                </div>
                <div className="text-[11px] text-slate-500">
                  {p.scenarioCount} scenario{p.scenarioCount === 1 ? '' : 's'} · updated {new Date(p.updatedAt).toLocaleString()}
                </div>
              </div>
              <button className="btn-secondary btn-sm" onClick={() => void openProject(p.id)} disabled={project?.id === p.id}>
                Open
              </button>
              <button className="btn-danger btn-sm" onClick={() => { if (window.confirm(`Delete "${p.name}" from this browser? Export a backup first if you need it.`)) void removeProject(p.id); }}>
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </Modal>
  );
}
