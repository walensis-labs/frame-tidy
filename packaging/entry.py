"""PyInstaller entry point for Frame Tidy."""

import multiprocessing

from aura_mcp.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
