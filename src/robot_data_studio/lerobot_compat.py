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
import pyarrow as pa
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


def dataset_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Dataset file paths must be relative to the dataset: {relative}")
    return root / relative


def required_file(root: Path, relative: str) -> Path:
    path = dataset_path(root, relative)
    if not path.is_file():
        raise FileNotFoundError(f"Missing or unreadable dataset file: {path}")
    with path.open("rb") as stream:
        stream.read(1)
    return path


def validate_source(root: Path) -> tuple[dict[str, Any], list[int]]:
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
        skipped: list[int] = []
        for episode in episodes:
            index = episode["episode_index"]
            if version == "v2.1":
                fields = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
            else:
                fields = {
                    "chunk_index": episode["data/chunk_index"],
                    "file_index": episode["data/file_index"],
                }
            data_relative = info["data_path"].format(**fields)
            video_paths: list[Path] = []
            try:
                for camera in cameras:
                    if version == "v3.0":
                        fields = {
                            "chunk_index": episode[f"videos/{camera}/chunk_index"],
                            "file_index": episode[f"videos/{camera}/file_index"],
                        }
                    video_paths.append(
                        required_file(root, info["video_path"].format(video_key=camera, **fields))
                    )
            except FileNotFoundError:
                if version != "v2.1":
                    raise
                skipped.append(index)
                continue
            data_path = required_file(root, data_relative)
            if data_path not in checked:
                try:
                    pq.read_metadata(data_path)
                except Exception as error:
                    raise ValueError(f"Unreadable data file {data_path}: {error}") from error
                checked.add(data_path)
            for video_path in video_paths:
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
        if len(skipped) == len(episodes):
            raise ValueError("No complete episodes remain; every episode has missing camera videos")
    except (KeyError, TypeError, OSError, ValueError) as error:
        raise ValueError(f"Invalid dataset at {root}: {error}") from error
    return info, skipped


def prepare_v21_subset(root: Path, skipped: list[int]) -> None:
    """Drop incomplete episodes and reindex the private copy before conversion."""
    info = read_json(root / "meta/info.json")
    episodes = [
        json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()
    ]
    stats = {
        row["episode_index"]: row
        for row in map(json.loads, (root / "meta/episodes_stats.jsonl").read_text().splitlines())
    }
    tasks = list(map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines()))
    cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    skipped_set = set(skipped)

    def data_path(index: int) -> Path:
        return dataset_path(
            root,
            info["data_path"].format(
                episode_index=index, episode_chunk=index // info["chunks_size"]
            ),
        )

    def video_path(index: int, camera: str) -> Path:
        return dataset_path(
            root,
            info["video_path"].format(
                episode_index=index, episode_chunk=index // info["chunks_size"], video_key=camera
            ),
        )

    for index in skipped:
        data_path(index).unlink(missing_ok=True)
        for camera in cameras:
            video_path(index, camera).unlink(missing_ok=True)
    kept = [episode for episode in episodes if episode["episode_index"] not in skipped_set]
    used_tasks = {
        task_index
        for episode in kept
        for task_index in pq.read_table(
            data_path(episode["episode_index"]), columns=["task_index"]
        )["task_index"].to_pylist()
    }
    task_map = {old: new for new, old in enumerate(sorted(used_tasks))}
    new_tasks = [
        {**task, "task_index": task_map[task["task_index"]]}
        for task in sorted(tasks, key=lambda task: task["task_index"])
        if task["task_index"] in task_map
    ]
    new_episodes = []
    new_stats = []
    offset = 0
    for new_index, episode in enumerate(kept):
        old_index = episode["episode_index"]
        old_data = data_path(old_index)
        table = pq.read_table(old_data)
        indices = {
            "episode_index": [new_index] * len(table),
            "index": list(range(offset, offset + len(table))),
            "task_index": [task_map[value] for value in table["task_index"].to_pylist()],
        }
        episode_stats = stats[old_index]
        for key, values in indices.items():
            field = table.schema.field(key)
            array = pa.array(values, type=field.type)
            table = table.set_column(table.schema.get_field_index(key), field, array)
            if key in episode_stats["stats"]:
                numbers = array.to_numpy()
                episode_stats["stats"][key] = {
                    "min": [int(numbers.min())],
                    "max": [int(numbers.max())],
                    "mean": [float(numbers.mean())],
                    "std": [float(numbers.std())],
                    "count": [len(table)],
                }
        new_data = data_path(new_index)
        new_data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, new_data)
        if old_data != new_data:
            old_data.unlink()
        for camera in cameras:
            old_video, new_video = video_path(old_index, camera), video_path(new_index, camera)
            if old_video != new_video:
                new_video.parent.mkdir(parents=True, exist_ok=True)
                old_video.rename(new_video)
        new_episodes.append({**episode, "episode_index": new_index, "length": len(table)})
        new_stats.append({**episode_stats, "episode_index": new_index})
        offset += len(table)
    splits = {}
    for name, selection in info.get("splits", {}).items():
        start, end = map(int, selection.split(":"))
        new_start = sum(episode["episode_index"] < start for episode in kept)
        new_end = sum(episode["episode_index"] < end for episode in kept)
        splits[name] = f"{new_start}:{new_end}"
    info.update(
        total_episodes=len(kept),
        total_frames=offset,
        total_tasks=len(new_tasks),
        total_videos=len(kept) * len(cameras),
        total_chunks=(len(kept) + info["chunks_size"] - 1) // info["chunks_size"],
        splits=splits,
    )
    (root / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    for name, rows in (
        ("episodes", new_episodes),
        ("episodes_stats", new_stats),
        ("tasks", new_tasks),
    ):
        (root / f"meta/{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def import_dataset(
    source: Path, destination: Path, version: str, skipped: list[int]
) -> dict[str, Any]:
    def ignore_files(directory: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in (".cache", ".git")
            or ((Path(directory) / name).is_symlink() and not (Path(directory) / name).exists())
        }

    shutil.copytree(source, destination, ignore=ignore_files)
    if version == "v2.1":
        if skipped:
            prepare_v21_subset(destination, skipped)
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
