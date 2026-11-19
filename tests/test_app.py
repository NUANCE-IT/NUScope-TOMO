"""The desktop launcher, without opening a window."""
import json
import urllib.request

import numpy as np
import pytest

from nuscope_tomo import Volume, VoxelSize


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("NUSCOPE_TOMO_HOME", str(tmp_path / "home"))
    import importlib

    import nuscope_tomo.app as mod
    return importlib.reload(mod)


def test_recent_projects_newest_first_and_only_existing(app, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    app.remember(a)
    app.remember(b)
    app.remember(a)
    assert app.recent() == [str(a.resolve()), str(b.resolve())]
    b.rmdir()
    assert app.recent() == [str(a.resolve())]


def test_studio_serves_a_project_then_stops(app, tmp_path):
    proj = tmp_path / "proj"
    Volume(np.zeros((4, 8, 8), np.uint8), VoxelSize(20, 10, 10)).to_zarr(proj / "raw.zarr")
    s = app.Studio()
    url = s.open(proj)
    info = json.loads(urllib.request.urlopen(url + "api/project", timeout=5).read())
    assert info["name"] == "proj" and "raw" in json.dumps(info)
    assert app.recent() == [str(proj.resolve())]
    s.close()
    with pytest.raises(OSError):
        urllib.request.urlopen(url + "api/project", timeout=2)


def test_studio_refuses_a_missing_folder(app, tmp_path):
    with pytest.raises(FileNotFoundError):
        app.Studio().open(tmp_path / "nope")


def test_start_page_escapes_paths(app, tmp_path):
    page = app.start_page([str(tmp_path / 'x"><script>')], error="<b>bad</b>")
    assert "<script>" not in page and "&lt;b&gt;bad" in page and "Open project folder" in page
