import hashlib
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robot_data_studio.app import create_app


def file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_import_converts_a_working_copy_without_changing_source(
    client: TestClient, source: Path
) -> None:
    before = file_hashes(source)
    response = client.post(
        "/api/projects", json={"name": "Order picking", "source_path": str(source)}
    )
    assert response.status_code == 201, response.text
    project = response.json()
    assert project["summary"] == {
        "fps": 10,
        "episode_count": 3,
        "frame_count": 329,
        "robot_type": "ffw_bg2_rev4",
        "tasks": [
            "Put the black wrench into the crate.",
            "Put the yellow paint brush into the crate.",
            "Grasp the handle of the crate.",
        ],
        "cameras": [
            {"key": "observation.images.cam_head", "width": 672, "height": 376},
            {"key": "observation.images.cam_head_right", "width": 672, "height": 376},
            {"key": "observation.images.cam_wrist_left", "width": 424, "height": 240},
            {"key": "observation.images.cam_wrist_right", "width": 424, "height": 240},
        ],
    }
    working_copy = Path(project["working_copy"])
    assert json.loads((working_copy / "meta/info.json").read_text())["codebase_version"] == "v3.0"
    dataset = LeRobotDataset("studio/local", root=working_copy, video_backend="pyav")
    assert dataset.num_episodes == 3
    assert len(dataset) == 329
    assert dataset[0]["task"] == "Put the black wrench into the crate."
    assert dataset[139]["task"] == "Put the yellow paint brush into the crate."
    assert dataset[257]["task"] == "Grasp the handle of the crate."
    assert file_hashes(source) == before
    config = json.loads((working_copy.parent / "project.json").read_text())
    assert config["source_path"] == str(source.resolve())
    assert config["import_info"]["source_version"] == "v2.1"
    assert config["import_info"]["working_version"] == "v3.0"


@pytest.mark.parametrize("broken_symlink", [False, True])
def test_v21_import_skips_missing_camera_episodes_and_reindexes_kept_frames(
    client: TestClient, source: Path, tmp_path: Path, broken_symlink: bool
) -> None:
    incomplete = tmp_path / "incomplete"
    shutil.copytree(source, incomplete)
    video = incomplete / "videos/chunk-000/observation.images.cam_wrist_right/episode_000001.mp4"
    video.unlink()
    if broken_symlink:
        video.symlink_to(tmp_path / "missing.mp4")
    kept_video = incomplete / "videos/chunk-000/observation.images.cam_head/episode_000000.mp4"
    kept_video.rename(kept_video.with_suffix(".original"))
    kept_video.symlink_to("episode_000000.original")
    before = file_hashes(incomplete)
    response = client.post(
        "/api/projects", json={"name": "Complete episodes", "source_path": str(incomplete)}
    )
    assert response.status_code == 201, response.text
    project = response.json()
    assert project["summary"]["episode_count"] == 2
    assert project["summary"]["frame_count"] == 211
    assert len(project["summary"]["cameras"]) == 4
    assert project["summary"]["tasks"] == [
        "Put the black wrench into the crate.",
        "Grasp the handle of the crate.",
    ]
    assert project["import_info"]["skipped_episodes"] == [1]
    assert project["import_info"]["source_episode_indices"] == [0, 2]
    dataset = LeRobotDataset("studio/local", root=project["working_copy"], video_backend="pyav")
    assert len(dataset.hf_dataset) == 211
    assert dataset[139]["episode_index"].item() == 1
    assert dataset[139]["index"].item() == 139
    assert dataset[139]["frame_index"].item() == 2613
    assert dataset[139]["timestamp"].item() == 0
    assert dataset[139]["task"] == "Grasp the handle of the crate."
    kept_source = pq.read_table(incomplete / "data/chunk-000/episode_000002.parquet")
    assert dataset[139]["action"].tolist() == kept_source["action"][0].as_py()
    assert dataset[210]["task"] == "Grasp the handle of the crate."
    assert dataset.meta.stats["episode_index"]["max"].tolist() == [1]
    assert dataset.meta.stats["index"]["max"].tolist() == [210]
    assert file_hashes(incomplete) == before


def test_v21_import_rejects_a_dataset_with_no_complete_episodes(
    client: TestClient, source: Path, tmp_path: Path
) -> None:
    incomplete = tmp_path / "no-complete-episodes"
    shutil.copytree(source, incomplete)
    for video in incomplete.glob("videos/chunk-000/observation.images.cam_head/*.mp4"):
        video.unlink()
    response = client.post("/api/projects", json={"name": "Empty", "source_path": str(incomplete)})
    assert response.status_code == 400
    assert "No complete episodes remain" in response.json()["detail"]
    assert client.get("/api/projects").json() == []


def test_v30_copy_can_be_listed_reopened_and_deleted_without_touching_source(
    client: TestClient, source: Path, tmp_path: Path
) -> None:
    original = client.post(
        "/api/projects", json={"name": "Original", "source_path": str(source)}
    ).json()
    v30_source = Path(original["working_copy"])
    before = file_hashes(v30_source)
    response = client.post("/api/projects", json={"name": "Copy", "source_path": str(v30_source)})
    assert response.status_code == 201, response.text
    copy = response.json()
    assert copy["summary"] == original["summary"]
    assert copy["import_info"]["source_version"] == "v3.0"
    assert file_hashes(Path(copy["working_copy"])) == before
    dataset = LeRobotDataset("studio/local", root=copy["working_copy"], video_backend="pyav")
    assert len(dataset) == 329
    assert dataset[328]["task"] == "Grasp the handle of the crate."
    assert {p["id"] for p in client.get("/api/projects").json()} == {original["id"], copy["id"]}
    with TestClient(create_app(projects_dir=tmp_path / "projects")) as reopened:
        assert reopened.get(f"/api/projects/{copy['id']}").json() == copy
        assert reopened.delete(f"/api/projects/{copy['id']}").status_code == 204
        assert reopened.get(f"/api/projects/{copy['id']}").status_code == 404
        assert reopened.delete(f"/api/projects/{copy['id']}").status_code == 404
        assert [p["id"] for p in reopened.get("/api/projects").json()] == [original["id"]]
    assert not Path(copy["working_copy"]).parent.exists()
    assert file_hashes(v30_source) == before


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("missing_info", "meta/info.json"),
        ("bad_json", "meta/info.json"),
        ("unsupported_version", "Unsupported dataset version"),
        ("unreadable_file", "episode_000001.parquet"),
        ("corrupt_video", "cam_wrist_right/episode_000001.mp4"),
        ("corrupt_data", "episode_000001.parquet"),
        ("bad_tasks", "meta/tasks.jsonl"),
        ("missing_task_index", "meta/tasks.jsonl"),
        ("missing_stats", "meta/episodes_stats.jsonl"),
    ],
)
def test_invalid_import_names_problem_and_leaves_no_project(
    client: TestClient, source: Path, tmp_path: Path, problem: str, message: str
) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(source, broken)
    info_path = broken / "meta/info.json"
    video = broken / "videos/chunk-000/observation.images.cam_wrist_right/episode_000001.mp4"
    data = broken / "data/chunk-000/episode_000001.parquet"
    if problem == "missing_info":
        info_path.unlink()
    elif problem == "bad_json":
        info_path.write_text("{")
    elif problem == "unsupported_version":
        info = json.loads(info_path.read_text())
        info["codebase_version"] = "v1.0"
        info_path.write_text(json.dumps(info))
    elif problem == "unreadable_file":
        data.chmod(0)
    elif problem == "corrupt_video":
        video.write_bytes(b"not a video")
    elif problem == "corrupt_data":
        data.write_bytes(b"not parquet")
    elif problem == "bad_tasks":
        (broken / "meta/tasks.jsonl").write_text("{broken\n")
    elif problem == "missing_task_index":
        (broken / "meta/tasks.jsonl").write_text('{"task": "a"}\n')
    elif problem == "missing_stats":
        (broken / "meta/episodes_stats.jsonl").write_text('{"episode_index": 0}\n')
    try:
        response = client.post("/api/projects", json={"name": "Broken", "source_path": str(broken)})
        assert response.status_code == 400, response.text
        assert message in response.json()["detail"]
        assert client.get("/api/projects").json() == []
        assert not (tmp_path / "projects").exists() or not list((tmp_path / "projects").iterdir())
    finally:
        data.chmod(0o644)


def test_missing_v30_camera_is_reported_before_copying(
    client: TestClient, source: Path, tmp_path: Path
) -> None:
    original = client.post(
        "/api/projects", json={"name": "Original", "source_path": str(source)}
    ).json()
    broken = tmp_path / "broken-v30"
    shutil.copytree(original["working_copy"], broken)
    video = next(broken.glob("videos/observation.images.cam_head/chunk-*/*.mp4"))
    video.unlink()
    before = file_hashes(broken)
    response = client.post("/api/projects", json={"name": "Broken", "source_path": str(broken)})
    assert response.status_code == 400
    assert str(video) in response.json()["detail"]
    assert file_hashes(broken) == before
    assert [p["id"] for p in client.get("/api/projects").json()] == [original["id"]]


def test_project_storage_cannot_be_nested_in_source(source: Path, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    shutil.copytree(source, dataset)
    before = file_hashes(dataset)
    with TestClient(create_app(projects_dir=dataset / "projects")) as nested:
        response = nested.post(
            "/api/projects", json={"name": "Nested", "source_path": str(dataset)}
        )
    assert response.status_code == 400
    assert "inside the source dataset" in response.json()["detail"]
    assert file_hashes(dataset) == before


@pytest.mark.parametrize(
    "project_id", ["unknown", "%2e%2e", "00000000-0000-0000-0000-000000000000"]
)
def test_unknown_project_cannot_be_opened_or_deleted(client: TestClient, project_id: str) -> None:
    assert client.get(f"/api/projects/{project_id}").status_code == 404
    assert client.delete(f"/api/projects/{project_id}").status_code == 404


def test_v30_paths_cannot_reference_files_outside_the_working_copy(
    client: TestClient, source: Path, tmp_path: Path
) -> None:
    original = client.post(
        "/api/projects", json={"name": "Original", "source_path": str(source)}
    ).json()
    broken = tmp_path / "external-path"
    shutil.copytree(original["working_copy"], broken)
    info_path = broken / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["data_path"] = str(Path(original["working_copy"]) / info["data_path"])
    info_path.write_text(json.dumps(info))
    response = client.post("/api/projects", json={"name": "Broken", "source_path": str(broken)})
    assert response.status_code == 400
    assert "relative" in response.json()["detail"]


def test_failed_validation_after_conversion_removes_partial_project_and_preserves_source(
    client: TestClient, source: Path, tmp_path: Path
) -> None:
    broken = tmp_path / "wrong-frame-count"
    shutil.copytree(source, broken)
    info_path = broken / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = 999
    info_path.write_text(json.dumps(info))
    before = file_hashes(broken)
    response = client.post("/api/projects", json={"name": "Broken", "source_path": str(broken)})
    assert response.status_code == 400
    assert "counts do not match" in response.json()["detail"]
    assert file_hashes(broken) == before
    assert client.get("/api/projects").json() == []
    assert not list((tmp_path / "projects").iterdir())
