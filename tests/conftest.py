import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedrf_diff.alignment import agree_schema  # noqa: E402
from fedrf_diff.config import CONFIG  # noqa: E402
from fedrf_diff.pipeline import build_federation  # noqa: E402
from fedrf_diff.server import Limits  # noqa: E402


@pytest.fixture
def small_cfg(tmp_path):
    """The real data with a short, cheap federation."""
    return dict(CONFIG, out_dir=tmp_path / "run", clients_dir=tmp_path / "clients", initial_trees=4,
                supplementary_trees=4, local_budget=6, global_budget=10, max_rounds=3,
                dp_sweep=[2.0, (2.0, 3.0), None], make_figures=False, local_tuning=None)


@pytest.fixture
def schema():
    return agree_schema()


@pytest.fixture
def limits():
    return Limits.from_config(CONFIG)


@pytest.fixture
def federation(small_cfg):
    return build_federation(small_cfg)
