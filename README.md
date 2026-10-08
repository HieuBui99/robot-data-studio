# Robot Data Studio

A local, single-user studio for LeRobot datasets. Ticket 01 provides project
creation, import, summaries, listing, opening, and deletion.

From this checkout, with [uv](https://docs.astral.sh/uv/), Python 3.12,
Node.js 20.19+ (or 22.12+), npm, and ffmpeg installed, run:

```sh
uv run --locked robot-data-studio
```

This installs the locked Python environment, installs and builds the React
frontend, then serves everything at **http://127.0.0.1:8000**. 

Enter a project name and the local dataset directory containing `meta/info.json`.
v2.1 imports run LeRobot's converter on a private copy; v3.0 imports copy the
dataset directly. Import checks the files for every episode and camera, and
validates the result with `LeRobotDataset` before publishing the project.
For v2.1 datasets, import skips episodes with missing camera videos, keeps every
camera for the remaining episodes, and reindexes episode, task and global frame
indices. Skipped source indices and the mapping back to source episodes are
recorded in `project.json`; the summary page shows the skipped count and indices.
Actions, observations, timestamps and camera video bytes for retained episodes
are preserved. Other invalid files still fail import. v3.0 imports remain strict.
The source stays unchanged. Full dataset copies need disk space comparable to
the original; v2.1 conversion temporarily also retains the unconverted copy.

Projects default to `./projects/`. To choose another location or port:

```sh
uv run --locked robot-data-studio --projects-dir /path/to/projects --port 8080
```

Each project has this layout:

```text
projects/<project-id>/
  project.json       # name, source path, import provenance, dataset summary
  dataset/           # independent v3.0 working copy
  annotations/       # episode sidecars (later tickets)
  previews/          # camera previews (later tickets)
  masks/             # cached masks (later tickets)
  runs/              # augmentation outputs (later tickets)
```


The HTTP API is `POST /api/projects` (`name`, `source_path`),
`GET /api/projects`, `GET /api/projects/{id}`, and `DELETE /api/projects/{id}`.
Interactive API documentation is available at `/docs`.

For frontend development, run the backend and `npm run dev --prefix frontend`
in separate terminals. Vite proxies `/api` to port 8000.

Checks:

```sh
uv run --locked mypy
uv run --locked ruff check src tests
uv run --locked pytest
npm ci --prefix frontend
npm run build --prefix frontend
```

API tests use a committed three-episode v2.1 OrderPicking fixture with distinct
tasks, the real import endpoint, the upstream converter, and the standard
dataset reader. They check source file hashes, v3.0 copies, persistence,
deletion, and invalid sources. No model services or GPU are required for tests.
