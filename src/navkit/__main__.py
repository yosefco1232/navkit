"""Allows `python -m navkit ...` as an alternative to the `navkit` command."""

from navkit.cli import main

raise SystemExit(main())
