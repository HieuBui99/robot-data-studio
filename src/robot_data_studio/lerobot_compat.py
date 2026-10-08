"""The only boundary to pinned LeRobot internals; all imports are local-only."""

import json
import os
import shutil
from pathlib import Path
from typing import Any

# LeRobot falls back to Hub downloads on missing local files. Imports must fail instead.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import av
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

LEROBOT_COMMIT = "200ee53596d464bd28f6595cdb31be69a2b5e379"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error


def required_file(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Dataset file paths must be relative to the dataset: {relative}")
    path = root / relative
    if not path.is_file():
        raise ValueError(f"Missing or unreadable dataset file: {path}")
    with path.open("rb") as stream:
        stream.read(1)
    return path


def validate_source(root: Path) -> dict[str, Any]:
    info = read_json(root / "meta/info.json")
    version = info.get("codebase_version")
    if version not in ("v2.1", "v3.0"):
        raise ValueError(f"Unsupported dataset version {version!r}; expected v2.1 or v3.0")
    if info.get("storage_format") not in (None, "parquet"):
        raise ValueError("Import requires the LeRobot parquet/video storage format")
    try:
        cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
        if version == "v2.1":
            metadata: dict[str, list[dict[str, Any]]] = {}
            for name, required_keys in {
                "tasks": {"task_index", "task"},
                "episodes": {"episode_index", "tasks", "length"},
                "episodes_stats": {"episode_index", "stats"},
            }.items():
                path = required_file(root, f"meta/{name}.jsonl")
                try:
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    if not all(isinstance(row, dict) for row in rows):
                        raise ValueError("expected JSON objects")
                    for row in rows:
                        if missing := required_keys - row.keys():
                            raise ValueError(
                                f"Missing required fields: {', '.join(sorted(missing))}"
                            )
                    metadata[name] = rows
                except ValueError as error:
                    raise ValueError(f"Cannot read {path}: {error}") from error
            episodes = metadata["episodes"]
        else:
            required_file(root, "meta/tasks.parquet")
            required_file(root, "meta/stats.json")
            episode_files = sorted(root.glob("meta/episodes/*/*.parquet"))
            if not episode_files:
                raise ValueError(f"Missing episode metadata in {root / 'meta/episodes'}")
            episodes = [row for path in episode_files for row in pq.read_table(path).to_pylist()]
        if len(episodes) != info["total_episodes"] or not episodes:
            raise ValueError(
                "Episode metadata count does not match meta/info.json or dataset is empty"
            )
        if [episode["episode_index"] for episode in episodes] != list(range(len(episodes))):
            raise ValueError("Episode indices must be contiguous starting at 0")
        checked: set[Path] = set()
        for episode in episodes:
            index = episode["episode_index"]
            if version == "v2.1":
                fields = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
            else:
                fields = {
                    "chunk_index": episode["data/chunk_index"],
                    "file_index": episode["data/file_index"],
                }
            data_path = required_file(root, info["data_path"].format(**fields))
            if data_path not in checked:
                try:
                    pq.read_metadata(data_path)
                except Exception as error:
                    raise ValueError(f"Unreadable data file {data_path}: {error}") from error
                checked.add(data_path)
            for camera in cameras:
                if version == "v3.0":
                    fields = {
                        "chunk_index": episode[f"videos/{camera}/chunk_index"],
                        "file_index": episode[f"videos/{camera}/file_index"],
                    }
                video_path = required_file(
                    root, info["video_path"].format(video_key=camera, **fields)
                )
                if video_path not in checked:
                    try:
                        with av.open(str(video_path)) as container:
                            if not container.streams.video:
                                raise ValueError("no video stream")
                            next(container.decode(video=0))
                    except Exception as error:
                        raise ValueError(
                            f"Unreadable camera video {video_path}: {error}"
                        ) from error
                    checked.add(video_path)
    except (KeyError, TypeError, OSError, ValueError) as error:
        raise ValueError(f"Invalid dataset at {root}: {error}") from error
    return info


def import_dataset(source: Path, destination: Path, version: str) -> dict[str, Any]:
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".cache", ".git"))
    if version == "v2.1":
        # The converter renames its input and creates siblings; use the private copied root.
        convert_dataset("studio/local", root=destination, push_to_hub=False)
        shutil.rmtree(destination.with_name(f"{destination.name}_old"))
    dataset = LeRobotDataset("studio/local", root=destination, video_backend="pyav")
    info = read_json(destination / "meta/info.json")
    frame_count = len(dataset.hf_dataset)
    if frame_count != info["total_frames"] or dataset.num_episodes != info["total_episodes"]:
        raise ValueError("Dataset frame or episode counts do not match meta/info.json")
    # Verify each episode can read actions, tasks and all cameras from the working copy.
    assert dataset.meta.episodes is not None
    for episode in dataset.meta.episodes:
        dataset[episode["dataset_from_index"]]
        dataset[episode["dataset_to_index"] - 1]
    tasks = pq.read_table(destination / "meta/tasks.parquet").to_pandas()
    return {
        "fps": info["fps"],
        "episode_count": dataset.num_episodes,
        "frame_count": frame_count,
        "robot_type": info.get("robot_type"),
        "tasks": tasks.index.tolist(),
        "cameras": [
            {"key": key, "height": feature["shape"][0], "width": feature["shape"][1]}
            for key, feature in info["features"].items()
            if feature["dtype"] in ("video", "image")
        ],
    }
