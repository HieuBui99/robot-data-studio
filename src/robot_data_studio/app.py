import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from robot_data_studio.lerobot_compat import (
    LEROBOT_COMMIT,
    import_dataset,
    read_json,
    validate_source,
)


class CreateProject(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    source_path: str = Field(min_length=1)

    @field_validator("name", "source_path")
    @classmethod
    def strip_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Must not be blank")
        return value.strip()


def create_app(projects_dir: Path | None = None, frontend_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Robot Data Studio")
    storage = (projects_dir if projects_dir is not None else Path.cwd() / "projects").resolve()

    def project_dir(project_id: str) -> Path:
        try:
            if str(UUID(project_id)) != project_id:
                raise ValueError
        except ValueError as error:
            raise HTTPException(404, "Project not found") from error
        path = storage / project_id
        if path.is_symlink() or not (path / "project.json").is_file():
            raise HTTPException(404, "Project not found")
        return path

    def load_project(path: Path) -> dict[str, Any]:
        config = read_json(path / "project.json")
        return {**config, "working_copy": str(path / "dataset")}

    @app.get("/api/projects")
    def list_projects() -> list[dict[str, Any]]:
        if not storage.exists():
            return []
        return [
            load_project(project_dir(path.name))
            for path in sorted(storage.iterdir())
            if not path.name.startswith(".") and (path / "project.json").is_file()
        ]

    @app.post("/api/projects", status_code=201)
    def create_project(request: CreateProject) -> dict[str, Any]:
        project_id = str(uuid4())
        staging = storage / f".import-{project_id}"
        try:
            source = Path(request.source_path).expanduser().resolve()
            if storage.is_relative_to(source):
                raise ValueError("Project storage must not be inside the source dataset")
            info = validate_source(source)
            staging.mkdir(parents=True)
            summary = import_dataset(source, staging / "dataset", info["codebase_version"])
            config = {
                "id": project_id,
                "name": request.name,
                "source_path": str(source),
                "created_at": datetime.now(UTC).isoformat(),
                "import_info": {
                    "source_version": info["codebase_version"],
                    "working_version": "v3.0",
                    "lerobot_commit": LEROBOT_COMMIT,
                },
                "summary": summary,
            }
            (staging / "project.json").write_text(json.dumps(config, indent=2) + "\n")
            for directory in ("annotations", "previews", "masks", "runs"):
                (staging / directory).mkdir()
            destination = storage / project_id
            staging.rename(destination)
            return load_project(destination)
        except Exception as error:
            shutil.rmtree(staging, ignore_errors=True)
            raise HTTPException(400, f"Import failed for {request.source_path}: {error}") from error

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, Any]:
        return load_project(project_dir(project_id))

    @app.delete("/api/projects/{project_id}", status_code=204)
    def delete_project(project_id: str) -> Response:
        shutil.rmtree(project_dir(project_id))
        return Response(status_code=204)

    frontend = frontend_dir or Path(__file__).resolve().parents[2] / "frontend/dist"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app
