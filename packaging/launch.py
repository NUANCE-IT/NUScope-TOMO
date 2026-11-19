"""Entry point for the packaged NUScope-TOMO app (PyInstaller)."""
import sys

from nuscope_tomo.app import main

if __name__ == "__main__":
    sys.exit(main())
