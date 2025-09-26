#!/usr/bin/env python3
"""
Main download script - Simple interface for multiprocess downloader
This is now the PRIMARY way to download Binance futures data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.download_multiprocess import main

if __name__ == "__main__":
    # Simply redirect to multiprocess downloader
    main()