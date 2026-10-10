"""``nuscope`` command line.

    nuscope info   <volume.zarr | tiff folder>
    nuscope ops                       # list recipe operations and parameters
    nuscope run    recipe.yaml [-p project_dir]
    nuscope studio [project_dir] [--port 8765]
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
    elif args.cmd == "studio":
        from .studio.server import serve
        serve(args.project, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
