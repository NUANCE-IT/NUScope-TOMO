"""``nuscope`` command line.

    nuscope info   <volume.zarr | tiff folder>
    nuscope ops                       # list recipe operations and parameters
    nuscope run    recipe.yaml [-p project_dir]
    nuscope studio [project_dir] [--port 8765]
    nuscope watch  <acquisition folder> -p project --voxel 20 5 5 [--model rf.pkl]
    nuscope compare NAME=project_dir [NAME=project_dir ...] [--out dir]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nuscope", description="NuScope-TOMO — FIB-SEM tomography")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("info", help="describe a volume")
    a.add_argument("path")
    sub.add_parser("ops", help="list operations")
    r = sub.add_parser("run", help="run a YAML recipe")
    r.add_argument("recipe")
    r.add_argument("-p", "--project")
    s = sub.add_parser("studio", help="launch the browser UI")
    s.add_argument("project", nargs="?", default=".")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--host", default="127.0.0.1")
    w = sub.add_parser("watch", help="live mode: analyse slices while the FIB mills")
    w.add_argument("folder")
    w.add_argument("-p", "--project", required=True)
    w.add_argument("--voxel", nargs=3, type=float, required=True, metavar=("Z", "Y", "X"),
                   help="voxel size in nm (slice thickness, pixel y, pixel x)")
    w.add_argument("--model", help="trained PixelClassifier (.pkl) for live segmentation")
    w.add_argument("--target", type=float, default=0.05, help="relative precision to reach")
    w.add_argument("--idle", type=float, default=60, help="stop after this many idle seconds")
    c = sub.add_parser("compare", help="compare samples with uncertainty-aware statistics")
    c.add_argument("samples", nargs="+", help="NAME=project_dir (a recipe output folder)")
    c.add_argument("--out", default="cohort")
    args = ap.parse_args(argv)

    if args.cmd == "info":
        from .core.volume import Volume
        from .io.stack import read_tiff_folder
        p = Path(args.path)
        v = Volume.from_zarr(p) if p.suffix == ".zarr" else read_tiff_folder(p)
        print(v)
        for k, val in v.meta.items():
            if not isinstance(val, list) or len(val) < 20:
                print(f"  {k}: {val}")
    elif args.cmd == "ops":
        from .core.registry import list_ops
        for name, spec in list_ops().items():
            print(f"{name:<34s} {spec.summary}")
            for k, d in spec.params().items():
                print(f"    {k} = {d!r}")
    elif args.cmd == "run":
        from .core.recipe import run_recipe
        pipe = run_recipe(args.recipe, args.project)
        print(f"done → {pipe.project}")
    elif args.cmd == "watch":
        from .core.volume import VoxelSize
        from .live import LiveSession
        LiveSession(args.folder, args.project, VoxelSize(*args.voxel), args.model,
                    target_rel_error=args.target).run(idle_timeout=args.idle)
    elif args.cmd == "compare":
        from .cohort import compare, load_project, plot
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        smp = {}
        for item in args.samples:
            name, _, path = item.partition("=")
            smp[name] = load_project(path)
        res = compare(smp)
        res["tests"].to_csv(out / "tests.csv", index=False)
        res["summary"].to_csv(out / "summary.csv", index=False)
        plot(res, str(out / "cohort.png"))
        print(res["tests"][["a", "b", "phase", "metric", "difference", "p_holm",
                            "significant"]].to_string())
    elif args.cmd == "studio":
        from .studio.server import serve
        serve(args.project, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
