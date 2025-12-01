#!/usr/bin/env python
"""CLI wrapper for building SFT data."""

import sys
import os

# Add parent directory to path to import sft_training package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import and run the main function from the tools module
from sft_training.tools.build_sft_data import main

if __name__ == "__main__":
    main()
