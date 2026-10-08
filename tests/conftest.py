from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from robot_data_studio.app import create_app


@pytest.fixture
def source() -> Path:
    return Path(__file__).parent / "fixtures" / "orderpicking-v21"


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(projects_dir=tmp_path / "projects")) as test_client:
        yield test_client
