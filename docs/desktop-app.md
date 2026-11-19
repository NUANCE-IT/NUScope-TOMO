# Desktop app (macOS and Windows)

NUScope-TOMO Studio also comes as an app you install like any other: no Python, no terminal.

| System | Download | Install |
|---|---|---|
| macOS 12+, Apple silicon | `NUScope-TOMO-<version>-macos-arm64.dmg` | Open the disk image and drag NUScope-TOMO to Applications |
| macOS 12+, Intel | `NUScope-TOMO-<version>-macos-x64.dmg` | Same |
| Windows 10/11, 64-bit | `NUScope-TOMO-<version>-windows-x64-setup.exe` | Run it. It installs for your user only; no administrator rights needed |

Downloads are attached to each [GitHub release](https://github.com/NUANCE-IT/NUScope-TOMO/releases).

## Using it

- With no project open, the start page offers **Open project folder…** and your recent projects.
- **File > Open Project…** switches project at any time. **File > Start Page** goes back.
- The Studio runs inside the app on this computer only (127.0.0.1, a free port). Nothing is sent anywhere.
- The first time you open a project in Documents, Desktop or Downloads, macOS asks to allow access to
  that folder. Allow it, or the project can't be read.
- Problems are written to `~/.nuscope-tomo/app.log` (on Windows, `C:\Users\<you>\.nuscope-tomo\app.log`).

## What the app leaves out

To keep the download near 250 MB instead of over 1 GB, the app doesn't include:

- **PyTorch**: the two deep-learning operations (Noise2Void denoising, U-Net segmentation) aren't listed.
- **PyVista/VTK**: offline 3D renders and turntable animations from `nuscope_tomo.viz`. The Studio's own 3D
  view runs in the window and is included.

For those, install the Python package: `pip install "nuscope-tomo[ai]"`.

## First launch of an unsigned build

The installers are not code-signed yet, so the system warns the first time:

- **macOS:** "cannot be opened because the developer cannot be verified". Right-click (or Control-click)
  NUScope-TOMO in Applications, choose **Open**, then **Open** again. Only needed once.
- **Windows:** SmartScreen shows "Windows protected your PC". Click **More info**, then **Run anyway**.

Signing removes these warnings. It needs an Apple Developer ID (for signing and notarization) and a
Windows code-signing certificate, both issued to the university or the developer. The build workflow has
places for both once they exist.

## Building it yourself

```bash
pip install -e ".[app]" pyinstaller
pyinstaller packaging/nuscope-tomo.spec --noconfirm --distpath build/dist --workpath build/work
build/dist/NUScope-TOMO.app/Contents/MacOS/NUScope-TOMO --self-test     # macOS
build\dist\NUScope-TOMO\NUScope-TOMO.exe --self-test                    # Windows
```

`--self-test` starts the bundled Studio without a window, calls its API, reads a slice and exits with 0 when
everything answered. GitHub Actions (`.github/workflows/app.yml`) builds all three installers on every version
tag, runs the self-test on each, and attaches them to the release. It can also be started by hand from the
Actions tab.

From a Python install, `nuscope-tomo app [project]` opens the same window.
