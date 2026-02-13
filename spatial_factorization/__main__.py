"""Main entry point for python -m spatial_factorization.

This module enables subprocess invocation using:
    python -m spatial_factorization <command> <args>
"""

from .cli import cli

if __name__ == "__main__":
    cli()
