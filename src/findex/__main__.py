"""`python -m findex ...` runs the same CLI as the `findex` script."""

from findex.cli import app

# The guard matters since Lab 5: under the `spawn` start method (Windows, macOS) every
# worker process re-imports the parent's main module as `__mp_main__`. Without the guard
# each worker would start the CLI again, which would start more workers, and so on.
if __name__ == "__main__":
    app(prog_name="findex")
