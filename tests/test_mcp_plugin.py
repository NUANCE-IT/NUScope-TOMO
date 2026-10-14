"""NUANCE-MCP plugin tools, on a fake server (NUANCE-MCP itself is not needed)."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from nuscope_tomo import Volume, VoxelSize
from nuscope_tomo import mcp as plugin


class FakeMCP:
    def __init__(self):
        self.tools = {}

    def tool(self, name):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco


@pytest.fixture
def tools():
    ctx = SimpleNamespace(mcp=FakeMCP())
    plugin.register(ctx)
    return ctx.mcp.tools


def test_project_table_volume(tmp_path, tools):
    proj = tmp_path / "proj"
    Volume(np.zeros((4, 8, 8), np.uint8), VoxelSize(20, 5, 5), name="raw").to_zarr(proj / "raw.zarr")
    (proj / "tables").mkdir()
    (proj / "tables" / "phases.csv").write_text("phase,fraction\npore,0.1\nsolid,0.9\n")
    p = tools["tomo_project"](str(proj))
    assert p["volumes"][0]["id"] == "raw" and p["volumes"][0]["voxel_nm"] == [20.0, 5.0, 5.0]
    assert tools["tomo_table"](str(proj), "tables/phases.csv")["rows"] == 2
    with pytest.raises(ValueError, match="outside"):
        tools["tomo_table"](str(proj), "../secrets.csv")
    v = tools["tomo_volume"](str(proj / "raw.zarr"))
    assert v["size_um"] == [0.08, 0.04, 0.04]
    assert "preprocess.denoise_nlm" in tools["tomo_ops"]()


def test_run_job_reports_errors(tmp_path, tools):
    j = tools["tomo_run"](str(tmp_path / "missing.yaml"))
    for _ in range(50):
        st = tools["tomo_job"](j["job"])
        if st["status"] != "running":
            break
        time.sleep(0.05)
    assert st["status"] == "error" and "missing.yaml" in st["error"]
    assert "no job" in tools["tomo_job"]("nope")["error"]


def test_phase_fractions(tmp_path, tools):
    lab = np.zeros((16, 32, 32), np.uint8)
    lab[:, :, 8:] = 1
    lab[:, :, 24:] = 2
    Volume(lab, VoxelSize(20, 10, 10), name="phases", kind="labels",
           meta={"classes": {"1": "solid", "2": "binder"}}).to_zarr(tmp_path / "phases.zarr")
    r = tools["tomo_phase_fractions"](str(tmp_path / "phases.zarr"))
    f = {p["name"]: p["fraction"] for p in r["phases"]}
    assert f == {"solid": 0.5, "binder": 0.25} and r["voxel_nm"] == [20.0, 10.0, 10.0]
