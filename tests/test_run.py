"""Test integracyjny: uruchamia caly domyslny pipeline Kedro.

Jest ciezki (dziala na pelnych danych, kilka minut), dlatego domyslnie jest
pomijany - zarowno lokalnie, jak i w CI. Aby go uruchomic, ustaw zmienna
srodowiskowa RUN_E2E=1, np.:

    RUN_E2E=1 pytest tests/test_run.py
"""
import os
from pathlib import Path

import pytest
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_E2E") != "1",
    reason="test e2e pomijany domyslnie (ustaw RUN_E2E=1, aby uruchomic)",
)


class TestKedroRun:
    def test_kedro_run(self):
        bootstrap_project(Path.cwd())

        with KedroSession.create(project_path=Path.cwd()) as session:
            assert session.run() is not None
