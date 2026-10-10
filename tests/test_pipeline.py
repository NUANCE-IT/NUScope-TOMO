"""Ground-truth tests on synthetic phantoms (no real data needed)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile
from scipy import ndimage as ndi

from nuscope_tomo import Volume, VoxelSize
from nuscope_tomo.analyze import measure_objects, phase_metrics
from nuscope_tomo.chem import chemical_phases, composition, denoise_counts, register_to_se, \
    simulate_eds
from nuscope_tomo.core.recipe import run_recipe
from nuscope_tomo.preprocess import align, destripe_slice, detect_sample_roi, estimate_drift
from nuscope_tomo.segment import PixelClassifier, auto_annotations, separate


def phantom(shape=(40, 192, 192), seed=0, corr=(2, 8, 8), fractions=(40, 75)):
    rng = np.random.default_rng(seed)
    f = ndi.gaussian_filter(rng.random(shape), corr)
    t = np.percentile(f, fractions)
    return (np.digitize(f, t) + 1).astype(np.uint8)


def test_volume_zarr_roundtrip(tmp_path):
    a = (np.random.default_rng(0).random((10, 32, 48)) * 255).astype(np.uint8)
    v = Volume(a, VoxelSize(20, 4, 4), "t", meta={"note": 1})
    w = v.to_zarr(tmp_path / "t.zarr")
    assert w.shape == a.shape and np.array_equal(w.numpy(), a)
    assert w.voxel_size.as_tuple() == (20, 4, 4) and w.meta["note"] == 1
    p = w.preview(max_voxels=10 * 16 * 24)
    assert p.shape == (5, 16, 24) and p.voxel_size.z == 40


def test_detect_roi_excludes_cap_and_walls():
    z, h, w = 20, 300, 400
    rng = np.random.default_rng(1)
    img = np.zeros((z, h, w), np.float32)
    img[:, 60:280, 80:330] = 100 + rng.normal(0, 8, (z, 220, 250))   # sample
    img[:, 20:60, 80:330] = 230                                       # Pt cap
    roi = detect_sample_roi(Volume(img.clip(0, 255).astype(np.uint8)), margin=0)
    assert 58 <= roi["y"][0] <= 70 and abs(roi["x"][0] - 80) <= 4 and abs(roi["x"][1] - 330) <= 4


def test_multibaseline_alignment_beats_pairwise():
    rng = np.random.default_rng(0)
    base = ndi.gaussian_filter(rng.random((30, 256, 256)), (3, 4, 4)).astype(np.float32)
    n = 120
    true = np.cumsum(rng.normal(0, 0.8, (n, 2)), 0)
    true -= true[0]
    st = np.stack([ndi.shift(base[int(i * 0.2)], true[i], order=1) + rng.normal(0, .01, (256, 256))
                   for i in range(n)]).astype(np.float32)
    v = Volume(st)
    errs = {}
    for off in [(1,), (1, 2, 4, 8)]:
        d = estimate_drift(v, patch=200, offsets=off)
        e = -d[["dy", "dx"]].to_numpy() - true
        errs[off] = np.sqrt(((e - e.mean(0)) ** 2).mean())
    assert errs[(1, 2, 4, 8)] < 1.0
    assert errs[(1, 2, 4, 8)] < 0.5 * errs[(1,)]


def test_align_skips_registered_stack():
    st = np.repeat(phantom((1, 128, 128))[0:1] * 60, 20, 0).astype(np.uint8)
    out = align(Volume(st), patch=128)
    assert out["volume"].meta["aligned"] == "already registered"


def test_destripe_removes_vertical_stripes():
    rng = np.random.default_rng(0)
    clean = ndi.gaussian_filter(rng.random((256, 256)), 6) * 255
    stripes = np.tile(rng.normal(0, 12, 256), (256, 1))
    out = destripe_slice((clean + stripes).astype(np.float32), level=4, sigma=2)
    col_err = lambda a: np.std((a - clean).mean(0))  # noqa: E731
    assert col_err(out) < 0.3 * col_err(clean + stripes)


def test_object_measurements_on_spheres():
    vs = VoxelSize(10, 5, 5)
    z, y, x = np.ogrid[:60, :200, :200]
    m = np.zeros((60, 200, 200), np.uint32)
    for i, (r, c) in enumerate([(200, (300, 250, 250)), (120, (250, 750, 300))], 1):
        m[((z * 10 - c[0]) ** 2 + (y * 5 - c[1]) ** 2 + (x * 5 - c[2]) ** 2) <= r * r] = i
    df = measure_objects(Volume(m, vs, kind="labels")).set_index("label")
    for i, r in [(1, 200), (2, 120)]:
        assert df.loc[i, "eq_diameter_nm"] == pytest.approx(2 * r, rel=0.01)
        assert df.loc[i, "sphericity"] == pytest.approx(1.0, abs=0.03)
        assert df.loc[i, "surface_area_um2"] == pytest.approx(4 * np.pi * r * r / 1e6, rel=0.02)


def test_phase_metrics_percolation_and_fraction():
    lab = np.ones((20, 40, 40), np.uint8)
    lab[:, 18:22, 18:22] = 2  # a straight channel along z
    out = phase_metrics(Volume(lab, VoxelSize(10, 10, 10), kind="labels"))["phases"]
    ch = out.set_index("class").loc[2]
    assert ch["volume_fraction"] == pytest.approx(16 / 1600)
    assert ch["percolates_z"] and not ch["percolates_x"]
    assert ch["tortuosity_z"] == pytest.approx(1.0, abs=0.02)


def test_random_forest_segments_noisy_phantom():
    ph = phantom()
    rng = np.random.default_rng(2)
    img = Volume((ph * 50 + rng.normal(0, 22, ph.shape)).clip(0, 255).astype(np.uint8))
    ann = auto_annotations(img, classes=3, n_slices=4, erode_px=3)
    clf = PixelClassifier(sigmas=(1, 2, 4), n_estimators=40).fit(img, ann)
    pred = clf.predict(img).numpy()
    assert (pred == ph).mean() > 0.9


def test_watershed_separates_touching_spheres():
    z, y, x = np.ogrid[:40, :80, :120]
    m = (((z - 20) ** 2 + (y - 40) ** 2 + (x - 40) ** 2) <= 15 ** 2) | \
        (((z - 20) ** 2 + (y - 40) ** 2 + (x - 66) ** 2) <= 15 ** 2)
    lab = separate(Volume(m.astype(np.uint8), VoxelSize(10, 10, 10)), min_distance_nm=150, h_nm=10)
    assert lab.meta["n_objects"] == 2


def test_eds_registration_and_phase_recovery():
    from sklearn.metrics import adjusted_rand_score
    ph = phantom((40, 384, 384), corr=(3, 24, 24))
    rng = np.random.default_rng(3)
    se = Volume((ph * 60 + rng.normal(0, 8, ph.shape)).clip(0, 255).astype(np.uint8),
                VoxelSize(20, 5, 5))
    comp = {1: {"Ni": .6, "Mn": .2, "O": .2}, 2: {"Ni": .1, "Mn": .1, "O": .8},
            3: {"Ni": .05, "Mn": .75, "O": .2}}
    eds, truth = simulate_eds(ph, comp, se.voxel_size, bin_xy=8, every=2,
                              counts_per_pixel=80, shift_px=(1.2, -0.8))
    reg = register_to_se(eds, se)
    est = -np.median(reg.shifts_px, 0)
    assert np.allclose(est, truth["shift_px"], atol=0.3)
    lab, _ = chemical_phases(composition(denoise_counts(reg, 30)), 3)
    assert adjusted_rand_score(ph[reg.z_index].ravel(), lab.ravel()) > 0.6


def test_recipe_end_to_end(tmp_path):
    ph = phantom((12, 96, 128))
    folder = tmp_path / "slices"
    folder.mkdir()
    noisy = ph * 70.0 + np.random.default_rng(0).normal(0, 6, ph.shape)
    for i, s in enumerate(noisy):
        tifffile.imwrite(folder / f"img_{i + 1:03d}.tif", s.clip(0, 255).astype(np.uint8))
    recipe = tmp_path / "r.yaml"
    recipe.write_text(f"""
name: t
input: {{tiff_folder: {folder}, voxel_size_nm: [20, 5, 5]}}
steps:
  - op: preprocess.median
  - op: segment.multi_otsu
    id: seg
    params: {{classes: 3}}
    save: true
  - op: analyze.phases
    inputs: {{labels: seg}}
""")
    pipe = run_recipe(recipe, tmp_path / "proj", log=lambda *_: None)
    prov = json.loads((tmp_path / "proj" / "provenance.json").read_text())
    assert [s["op"] for s in prov["steps"]] == ["preprocess.median", "segment.multi_otsu",
                                               "analyze.phases"]
    assert (tmp_path / "proj" / "volumes" / "seg.zarr").exists()
    assert (pipe.results["seg"].numpy() == ph).mean() > 0.99


def test_studio_api(tmp_path):
    from fastapi.testclient import TestClient
    from nuscope_tomo.studio.server import create_app
    Volume(phantom((8, 64, 64)) * 60, VoxelSize(20, 5, 5), "img").to_zarr(tmp_path / "img.zarr")
    c = TestClient(create_app(tmp_path))
    p = c.get("/api/project").json()
    assert [v["id"] for v in p["volumes"]] == ["img"]
    assert c.get("/api/volume/img/info").json()["shape"] == [8, 64, 64]
    r = c.get("/api/volume/img/slice?axis=1&index=3")
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    b = c.get("/api/volume/img/preview3d").content
    n = int.from_bytes(b[:4], "little")
    hdr = json.loads(b[4:4 + n])
    assert len(b) - 4 - n == int(np.prod(hdr["shape"]))
    assert any(o["name"] == "segment.random_forest" for o in c.get("/api/ops").json())


def test_triple_phase_boundary_straight_line():
    from nuscope_tomo.analyze import triple_phase_boundary
    lab = np.zeros((30, 40, 40), np.uint8)
    lab[:, :, :20], lab[:, :20, 20:], lab[:, 20:, 20:] = 1, 2, 3
    stats, _ = triple_phase_boundary(lab, (20, 10, 10), (1, 2, 3))
    assert stats["tpb_length_um"] == pytest.approx(0.6)
    assert stats["active_fraction"] == pytest.approx(1.0)
    lab[:, 30:35, 5:10] = 3  # an isolated inclusion of phase 3 inside phase 1 adds no TPB
    assert triple_phase_boundary(lab, (20, 10, 10), (1, 2, 3))[0]["tpb_length_um"] == \
        pytest.approx(0.6)


def test_transport_channel_and_block():
    from nuscope_tomo.analyze import effective_transport
    m = np.zeros((20, 30, 30), bool)
    m[:, 10:20, 10:20] = True
    r = effective_transport(m, (10, 10, 10), axis=0)
    assert r["D_eff_ratio"] == pytest.approx(r["porosity"], rel=1e-6)
    assert r["tortuosity_factor"] == pytest.approx(1.0, rel=1e-6)
    assert effective_transport(m, (10, 10, 10), axis=2)["D_eff_ratio"] == 0.0  # no path in x
    assert effective_transport(np.ones((12, 20, 20), bool), (20, 10, 10), 1)["D_eff_ratio"] == \
        pytest.approx(1.0)


def test_network_two_spheres_neck():
    from nuscope_tomo.analyze import extract_network
    z, y, x = np.ogrid[:40, :80, :120]
    m = (((z - 20) ** 2 + (y - 40) ** 2 + (x - 40) ** 2) <= 15 ** 2) | \
        (((z - 20) ** 2 + (y - 40) ** 2 + (x - 66) ** 2) <= 15 ** 2)
    lab = separate(Volume(m.astype(np.uint8), VoxelSize(10, 10, 10)), min_distance_nm=150,
                   h_nm=10).numpy()
    nodes, links = extract_network(lab, (10, 10, 10))
    assert list(nodes.coordination) == [1, 1] and len(links) == 1
    neck = 2 * np.sqrt(150 ** 2 - 130 ** 2)  # analytical neck diameter (nm)
    assert links.throat_diameter_nm[0] == pytest.approx(neck, abs=15)


def test_calibration_check_detects_wrong_slice_thickness():
    from nuscope_tomo.analyze import calibration_check
    rng = np.random.default_rng(0)
    iso = ndi.gaussian_filter(rng.random((120, 120, 120)), 6)[::2]  # true slices = 2 × pixel
    ok = calibration_check(Volume(iso, VoxelSize(20, 10, 10)))
    bad = calibration_check(Volume(iso, VoxelSize(50, 10, 10)))
    assert ok["warning"] == "" and abs(ok["z_over_xy"] - 1) < 0.15
    assert bad["warning"] and abs(bad["z_over_xy"] - 2.5) < 0.3


def test_integral_range_predicts_sampling_error():
    from nuscope_tomo.analyze import integral_range_cov
    fr, A, thr = [], [], None
    for seed in range(12):
        f = ndi.gaussian_filter(np.random.default_rng(seed).random((136,) * 3), 2,
                                mode="wrap")[20:-20, 20:-20, 20:-20]
        thr = np.percentile(f, 70) if thr is None else thr
        m = f > thr
        fr.append(m.mean())
        A.append(integral_range_cov(m, (10, 10, 10)))
    p, V = np.mean(fr), 96 ** 3 * 1e3 / 1e9
    predicted = np.sqrt(p * (1 - p) * np.median(A) / V)
    assert 0.6 < predicted / np.std(fr, ddof=1) < 1.6


def _noisy_phantom(noise, seed=0, shape=(12, 128, 128)):
    rng = np.random.default_rng(seed)
    f = ndi.gaussian_filter(rng.random(shape), (2, 5, 5))
    ph = (np.digitize(f, np.percentile(f, [40, 75])) + 1).astype(np.uint8)
    clean = ph * 60.0 + 20
    return ph, clean, (clean + rng.normal(0, noise, ph.shape)).clip(0, 255).astype(np.uint8)


def test_noise2void_beats_input_noise():
    pytest.importorskip("torch")
    from nuscope_tomo.ai import denoise_n2v
    _, clean, noisy = _noisy_phantom(30)
    out = denoise_n2v(Volume(noisy), steps=300, batch=32).numpy().astype(float)
    psnr = lambda a: 10 * np.log10(255 ** 2 / np.mean((a - clean) ** 2))  # noqa: E731
    assert psnr(out) > psnr(noisy.astype(float)) + 6


def test_unet_from_scribbles_matches_forest():
    pytest.importorskip("torch")
    from nuscope_tomo.ai import unet_segment
    ph, _, noisy = _noisy_phantom(40)
    v = Volume(noisy)
    ann = auto_annotations(v, classes=3, n_slices=3, erode_px=3)
    rf = (PixelClassifier(n_estimators=40).fit(v, ann).predict(v).numpy() == ph).mean()
    un = (unet_segment(v, annotations=ann, classes=3, steps=300)["volume"].numpy() == ph).mean()
    assert un > rf - 0.03


def test_live_session_ingests_and_segments(tmp_path):
    from nuscope_tomo.live import LiveSession
    rng = np.random.default_rng(0)
    nz, h, w = 40, 200, 260
    f = ndi.gaussian_filter(rng.random((nz, 140, 200)), (2, 5, 5))
    ph = (np.digitize(f, np.percentile(f, [50])) + 1).astype(np.uint8)  # 2 phases
    frames = np.zeros((nz, h, w), np.uint8)
    frames[:, 40:180, 30:230] = (ph * 50 + 30 + rng.normal(0, 6, ph.shape)).clip(0, 255)
    frames[:, 15:40, 30:230] = 235  # Pt cap
    # model trained on processed-like data (same intensities, binned 2x)
    small = frames[:, 40:180, 30:230].reshape(nz, 70, 2, 100, 2).mean((2, 4)).astype(np.uint8)
    sv = Volume(small)
    clf = PixelClassifier(sigmas=(1, 2, 4), n_estimators=30,
                          class_names={1: "dark", 2: "bright"}).fit(
        sv, auto_annotations(sv, classes=2, n_slices=3, erode_px=2))
    clf.save(tmp_path / "m.pkl")
    acq = tmp_path / "acq"
    acq.mkdir()
    sess = LiveSession(acq, tmp_path / "proj", VoxelSize(20, 5, 5), tmp_path / "m.pkl",
                       stats_every=10, log=lambda *_: None)
    for i, fr in enumerate(frames):
        tifffile.imwrite(acq / f"slice_{i:03d}.tif", fr)
        sess.step()  # first poll sees the file, second confirms its size is stable
        sess.step()
    assert sess.stores["raw"].shape[0] == nz
    assert sess.stores["labels"].shape[0] == nz
    doc = json.loads((tmp_path / "proj" / "live" / "live.json").read_text())
    fr = {p["name"]: p["fraction"] for p in doc["phases"]}
    assert abs(fr["dark"] - 0.5) < 0.12 and doc["slices"] == 40


def test_advisor_finds_oversampling():
    from nuscope_tomo.advisor import advise, chord_lengths
    m = np.zeros((5, 50, 60), bool)
    m[:, :, 20:30] = True
    assert set(chord_lengths(m, 2, 10.0)) == {100.0}
    f = ndi.gaussian_filter(np.random.default_rng(0).random((48, 160, 160)), 8)
    lab = (f > np.median(f)).astype(np.uint8) + 1  # features ~100 nm, sampled at 5 nm / 10 nm
    rec = advise(Volume(lab, VoxelSize(10, 5, 5), kind="labels"))["recommendation"].iloc[0]
    assert rec.recommended_pixel_nm >= 10 and rec.recommended_slice_nm >= 20
    assert rec.max_change_at_recommendation <= 0.05


def test_cohort_detects_large_difference_not_noise():
    from nuscope_tomo.cohort import compare, summarize

    def sample(seed, q):
        f = ndi.gaussian_filter(np.random.default_rng(seed).random((60, 120, 120)), 2)
        return Volume((f > np.quantile(f, q)).astype(np.uint8) + 1, VoxelSize(10, 10, 10),
                      kind="labels")

    res = compare({"a": summarize(sample(0, 0.6)), "b": summarize(sample(1, 0.6)),
                   "c": summarize(sample(2, 0.3))})["tests"]
    frac = res[(res.metric == "fraction") & (res.phase == "2")]
    ab = frac[(frac.a == "a") & (frac.b == "b")].iloc[0]
    ac = frac[(frac.a == "a") & (frac.b == "c")].iloc[0]
    assert not ab.significant and ac.significant
    assert ab.min_detectable_rel > abs(ab.rel_difference)


def test_sparse_dkl_beats_interpolation():
    pytest.importorskip("torch")
    from nuscope_tomo.sparse import active_sparse, reconstruct_interpolate
    rng = np.random.default_rng(0)
    B, ny, nx = 4, 40, 40
    f = ndi.gaussian_filter(rng.random((ny * B, nx * B)), 3)
    ph = (f > np.median(f)).astype(int)
    se = (ph * 80 + 60 + rng.normal(0, 6, ph.shape)).astype(np.float32)
    truth = ph.reshape(ny, B, nx, B).mean((1, 3))[..., None]  # composition follows the phase
    meas = (truth + rng.normal(0, 0.15, truth.shape)).reshape(-1, 1)
    r = active_sparse(se, (ny, nx), B, lambda i: meas[i], budget=0.08, init=0.03, rounds=2,
                      patch=12)
    itp = reconstruct_interpolate((ny, nx), r["measured_idx"], r["measured_values"])
    rmse = lambda a: np.sqrt(np.mean((a - truth) ** 2))  # noqa: E731
    assert rmse(r["mean"]) < 0.8 * rmse(itp)


def test_autoscript_dry_run_records_calls_and_guards():
    from nuscope_tomo.autonomy import AutoScriptFIBSEM, SafetyError, SafetyLimits
    fib = AutoScriptFIBSEM(dry_run=True, limits=SafetyLimits(max_slices=2, max_slice_nm=50))
    fib.mill(20)
    fib.image(10, 2.0)
    fib.autofocus()
    fib.beam_shift(5, -3)
    calls = [c["call"] for c in fib.calls.entries]
    assert calls == ["patterning.create_cleaning_cross_section+run", "imaging.grab_frame",
                     "auto_functions.run_auto_focus", "beams.electron_beam.beam_shift"]
    assert fib.calls.entries[0]["height_m"] == pytest.approx(20e-9)
    assert fib.calls.entries[1]["hfw_m"] == pytest.approx(10e-9 * 1536)
    with pytest.raises(SafetyError):
        fib.mill(80)  # thicker than allowed
    with pytest.raises(SafetyError):
        fib.arm("yes")  # the arming phrase must be exact
    fib.mill(20)
    with pytest.raises(SafetyError):
        fib.mill(20)  # slice limit


def test_agent_on_twin_corrects_drift_and_focus():
    from nuscope_tomo.autonomy import DigitalTwin, Policy, SliceAndViewAgent
    rng = np.random.default_rng(0)
    f = ndi.gaussian_filter(rng.random((80, 96, 96)), (2, 4, 4))
    ph = (np.digitize(f, np.percentile(f, [50])) + 1).astype(np.uint8)
    clean = Volume((ph * 70 + 40).astype(np.float32), VoxelSize(20, 10, 10))
    labels = Volume(ph, VoxelSize(20, 10, 10), kind="labels")
    noisy = Volume((ph * 70 + 40 + rng.normal(0, 8, ph.shape)).clip(0, 255).astype(np.uint8))
    clf = PixelClassifier(sigmas=(1, 2), n_estimators=20, class_names={1: "a", 2: "b"}).fit(
        noisy, auto_annotations(noisy, classes=2, n_slices=3, erode_px=2))
    twin = DigitalTwin(clean, labels, {1: {"X": 1.0}, 2: {"Y": 1.0}}, noise_at_1us=8,
                       drift_nm_per_slice=15, defocus_px_per_slice=0.08)
    pol = Policy(max_depth_nm=1000, eds_every=0, stop_on_precision=False, stats_every=0)
    res = SliceAndViewAgent(twin, clf, pol).run()
    acts = res["decisions"].action.value_counts()
    assert res["slices"] >= 20 and acts.get("beam_shift", 0) > 0 and acts.get("autofocus", 0) > 0
    assert np.hypot(*twin.drift) < 60  # drift kept in check (would random-walk otherwise)


def test_cnmf_pansharpening_beats_upsampling():
    from nuscope_tomo.sparse import cnmf_pansharpen, fusion_metrics
    rng = np.random.default_rng(0)
    H = W = 128
    B, E = 8, 12
    f = ndi.gaussian_filter(rng.random((H, W)), 4)
    lab = np.digitize(f, np.percentile(f, [33, 66]))
    spec = np.abs(rng.normal(0, 1, (3, E))) + 0.1
    truth = spec[lab]
    X = truth + rng.normal(0, 0.05, truth.shape)
    YH = X.reshape(H // B, B, W // B, B, E).mean((1, 3))  # coarse full-spectrum map
    sel = [0, 3, 7]                                        # three fixed-wavenumber maps
    R = np.zeros((3, E))
    R[range(3), sel] = 1
    out = cnmf_pansharpen(YH, X[..., sel], R, B, n_components=3)["cube"]
    up = np.repeat(np.repeat(YH, B, 0), B, 1)
    m, u = fusion_metrics(out, truth), fusion_metrics(up, truth)
    assert m["cc"] > u["cc"] + 0.1 and m["sam"] < 0.6 * u["sam"]
