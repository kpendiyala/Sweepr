"""Lets `python -m sweepr run -- ...` work even when `supervise` isn't on your PATH."""

import sys

from sweepr.cli import main

sys.exit(main())
