import { useEffect, useState, type FormEvent } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

type Project = {
  id: string
  name: string
  source_path: string
  working_copy: string
  import_info?: { skipped_episodes?: number[] }
  summary: {
    fps: number
    episode_count: number
    frame_count: number
    robot_type: string | null
    tasks: string[]
    cameras: { key: string; width: number; height: number }[]
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api/projects${path}`, options)
  if (!response.ok) {
    const body = await response.json()
    throw new Error(typeof body.detail === 'string' ? body.detail : 'Please check the project name and source path.')
  }
  return response.status === 204 ? undefined as T : response.json()
}

function App() {
  const [projects, setProjects] = useState<Project[]>([])
  const [selected, setSelected] = useState<Project | null>(null)
  const [name, setName] = useState('')
  const [sourcePath, setSourcePath] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    request<Project[]>('').then(setProjects).catch((e: Error) => setError(e.message))
  }, [])

  async function createProject(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      const project = await request<Project>('', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, source_path: sourcePath }),
      })
      setProjects(await request<Project[]>(''))
      setSelected(project)
      setName('')
      setSourcePath('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Import failed.')
    } finally {
      setBusy(false)
    }
  }

  async function openProject(id: string) {
    setError('')
    try { setSelected(await request<Project>(`/${id}`)) }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not open project.') }
  }

  async function deleteProject(project: Project) {
    if (!window.confirm(`Delete project “${project.name}” and its working copy? Your source dataset will be kept.`)) return
    setBusy(true)
    setError('')
    try {
      await request<void>(`/${project.id}`, { method: 'DELETE' })
      setProjects(projects.filter((p) => p.id !== project.id))
      if (selected?.id === project.id) setSelected(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not delete project.')
    } finally { setBusy(false) }
  }

  return <main>
    <header><p className="eyebrow">LOCAL DATASET WORKSPACE</p><h1>Robot Data Studio</h1></header>
    {error && <p role="alert" className="error">{error}</p>}
    <div className="layout">
      <aside>
        <section>
          <h2>Create a project</h2>
          <p>Import a local LeRobot v2.1 or v3.0 dataset into a separate working copy.</p>
          <p>For v2.1 datasets, episodes with missing camera videos are skipped.</p>
          <form onSubmit={createProject}>
            <label htmlFor="name">Project name</label>
            <input id="name" required maxLength={120} value={name} onChange={(e) => setName(e.target.value)} disabled={busy} />
            <label htmlFor="source">Dataset path on this machine</label>
            <input id="source" required value={sourcePath} onChange={(e) => setSourcePath(e.target.value)} placeholder="/path/to/dataset" disabled={busy} />
            <button disabled={busy}>{busy ? 'Working…' : 'Import dataset'}</button>
            {busy && <p role="status">Copying and validating the dataset. Large imports may take a few minutes.</p>}
          </form>
        </section>
        <section>
          <h2>Projects</h2>
          {projects.length === 0 && <p>No projects yet.</p>}
          <ul className="projects">{projects.map((project) => <li key={project.id}>
            <button className="project" aria-pressed={selected?.id === project.id} disabled={busy} onClick={() => openProject(project.id)}>{project.name}</button>
            <button className="delete" aria-label={`Delete ${project.name}`} disabled={busy} onClick={() => deleteProject(project)}>Delete</button>
          </li>)}</ul>
        </section>
      </aside>
      <section className="summary">
        {selected ? <>
          <p className="eyebrow">PROJECT SUMMARY</p><h2>{selected.name}</h2>
          {(selected.import_info?.skipped_episodes?.length ?? 0) > 0 && <details className="import-note">
            <summary>Skipped {selected.import_info!.skipped_episodes!.length} episodes with missing camera videos</summary>
            <p>Source episode indices: {selected.import_info!.skipped_episodes!.join(', ')}</p>
            <p>All cameras are kept for the remaining episodes. The source dataset is unchanged.</p>
          </details>}
          <dl className="metrics">
            <div><dt>Episodes</dt><dd>{selected.summary.episode_count.toLocaleString()}</dd></div>
            <div><dt>Frames</dt><dd>{selected.summary.frame_count.toLocaleString()}</dd></div>
            <div><dt>FPS</dt><dd>{selected.summary.fps}</dd></div>
            <div><dt>Robot</dt><dd>{selected.summary.robot_type ?? 'Unspecified'}</dd></div>
          </dl>
          <h3>Cameras</h3>
          <ul>{selected.summary.cameras.map((camera) => <li key={camera.key}>{camera.key} · {camera.width} × {camera.height}</li>)}</ul>
          <h3>Tasks</h3><ul>{selected.summary.tasks.map((task) => <li key={task}>{task}</li>)}</ul>
          <h3>Dataset paths</h3>
          <p>Source: <code>{selected.source_path}</code></p>
          <p>Working copy: <code>{selected.working_copy}</code></p>
        </> : <div className="empty"><h2>Your dataset, ready to work on</h2><p>Create or open a project to see its cameras, tasks, and frame counts.</p></div>}
      </section>
    </div>
  </main>
}

createRoot(document.getElementById('root')!).render(<App />)
