import sys

from .cli import main

# Guarded: worker processes (green/verify.py) import the main module and must not run the CLI.
if __name__ == "__main__":
    sys.exit(main())
