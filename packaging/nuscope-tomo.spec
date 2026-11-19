# PyInstaller build for the NUScope-TOMO desktop app (macOS .app, Windows folder for the installer).
#   pyinstaller packaging/nuscope-tomo.spec --noconfirm
# Leaves out PyTorch (deep-learning extras) and PyVista/VTK (offline 3D renders): the Studio needs neither.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

HERE = Path(SPECPATH)
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from nuscope_tomo import __version__  # noqa: E402

datas = collect_data_files("nuscope_tomo")                     # Studio front end, recipes
datas += collect_data_files("skimage", includes=["**/*.pyi"])   # scikit-image's lazy loader reads its .pyi stubs
for dist in ("zarr", "numcodecs", "imageio", "dask", "fastapi", "uvicorn", "pywebview", "scikit-image",
             "nuscope-tomo"):
    try:
        datas += copy_metadata(dist)                            # entry points (zarr codecs) and version lookups
    except Exception:
        pass

hidden = collect_submodules("nuscope_tomo", filter=lambda m: not m.startswith(("nuscope_tomo.ai", "nuscope_tomo.viz.render")))
hidden += collect_submodules("uvicorn") + collect_submodules("numcodecs") + collect_submodules("zarr.codecs")

excludes = ["torch", "torchvision", "pyvista", "vtk", "vtkmodules", "IPython", "jupyter", "notebook",
            "pytest", "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6"]

a = Analysis([str(HERE / "launch.py")], pathex=[str(ROOT)], datas=datas, hiddenimports=hidden,
             excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)
icon = HERE / "icons" / ("nuscope-tomo.icns" if sys.platform == "darwin" else "nuscope-tomo.ico")
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="NUScope-TOMO", console=False, icon=str(icon),
          upx=False, argv_emulation=False)
coll = COLLECT(exe, a.binaries, a.datas, name="NUScope-TOMO", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(coll, name="NUScope-TOMO.app", icon=str(icon), bundle_identifier="edu.northwestern.nuance.nuscope-tomo",
                 version=__version__,
                 info_plist={"CFBundleShortVersionString": __version__, "CFBundleVersion": __version__,
                             "NSHighResolutionCapable": True, "LSMinimumSystemVersion": "12.0",
                             "NSHumanReadableCopyright": "Roberto dos Reis, NUANCE Center, Northwestern University"})
