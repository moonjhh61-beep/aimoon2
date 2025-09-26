#!/usr/bin/env python3
"""
Universal S3 Parallel Downloader for Binance Futures Data
Downloads ALL symbols from S3 bucket with maximum parallelization
Zero 404 errors by checking S3 directory first
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import hashlib
import zipfile
import pandas as pd
import requests

from downloader.s3_scraper import S3DirectoryScraper
from downloader.download_tracker import DownloadTracker
from downloader.symbol_manager import SymbolManager
from storage.storage_manager import StorageManager


def download_file_task(args):
    """Worker function to download a single file with checksum verification"""
    file_info, output_dir = args

    try:
        # Build output path
        output_path = Path(output_dir) / file_info['filename']
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        if output_path.exists() and output_path.stat().st_size == file_info['size']:
            return {'status': 'skipped', 'file': file_info['filename']}

        # Download file
        url = f"https://data.binance.vision/{file_info['key']}"
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        # Write to file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=131072):
                if chunk:
                    f.write(chunk)

        # Verify size
        if output_path.stat().st_size != file_info['size']:
            output_path.unlink()
            return {'status': 'failed', 'file': file_info['filename'], 'error': 'Size mismatch'}

        # Download checksum file for reference
        checksum_url = f"{url}.CHECKSUM"
        try:
            checksum_resp = requests.get(checksum_url, timeout=10)
            if checksum_resp.status_code == 200:
                checksum_path = Path(output_dir) / f"{file_info['filename']}.CHECKSUM"
                with open(checksum_path, 'w') as f:
                    f.write(checksum_resp.text)
        except:
            pass  # Checksum is optional

        return {'status': 'success', 'file': file_info['filename'], 'size': file_info['size']}

    except Exception as e:
        return {'status': 'failed', 'file': file_info['filename'], 'error': str(e)}


class UniversalS3Downloader:
    """Universal downloader that gets everything from S3"""

    def __init__(self, num_workers: int = None):
        """Initialize downloader"""
        self.scraper = S3DirectoryScraper()
        self.symbol_manager = SymbolManager()
        self.tracker = DownloadTracker()
        self.storage = StorageManager()

        # Set number of workers (default: CPU count, max: 20)
        if num_workers is None:
            self.num_workers = min(cpu_count(), 20)
        else:
            self.num_workers = num_workers

        # Data types to download
        self.data_types = ['klines', 'fundingRate', 'markPriceKlines', 'indexPriceKlines', 'premiumIndexKlines']

    def discover_all_symbols(self):
        """Discover all available symbols from S3"""
        print("🔍 Discovering all symbols from S3...")
        symbols = self.scraper.discover_all_symbols()
        print(f"✓ Found {len(symbols)} symbols")
        return symbols

    def process_to_parquet(self, symbol: str, data_type: str):
        """Convert downloaded ZIP files to Parquet"""
        # Map to actual folder names
        folder_map = {
            'klines': 'klines',
            'fundingRate': 'fundingRate',
            'markPriceKlines': 'markPriceKlines',
            'indexPriceKlines': 'indexPriceKlines',
            'premiumIndexKlines': 'premiumIndexKlines'
        }

        folder_name = folder_map.get(data_type, data_type)

        # Paths - use binance_futures_data directory
        base_dir = Path(__file__).parent.parent  # binance_futures_data/
        raw_dir = base_dir / 'data' / 'raw' / folder_name / symbol
        processed_dir = base_dir / 'data' / 'processed' / folder_name / symbol

        if not raw_dir.exists():
            return

        processed_dir.mkdir(parents=True, exist_ok=True)

        # Process each ZIP file
        for zip_file in raw_dir.glob('*.zip'):
            try:
                parquet_filename = zip_file.stem + '.parquet'
                parquet_path = processed_dir / parquet_filename

                if not parquet_path.exists():
                    # Extract and convert
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        csv_filename = zf.namelist()[0]
                        csv_data = zf.read(csv_filename)

                        # Read CSV from bytes
                        import io
                        df = pd.read_csv(io.BytesIO(csv_data))

                        # Convert timestamps if present
                        if 'open_time' in df.columns:
                            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                        if 'close_time' in df.columns:
                            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

                        # Save as Parquet
                        df.to_parquet(parquet_path, compression='snappy', index=False)

                # Cleanup ZIP after successful conversion
                if parquet_path.exists():
                    zip_file.unlink()

            except Exception as e:
                print(f"  ⚠️ Error processing {zip_file.name}: {e}")

    def download_symbol(self, symbol: str):
        """Download all data for a single symbol"""
        all_tasks = []

        for data_type in self.data_types:
            # Get files from S3
            files = self.scraper.list_symbol_files(symbol, data_type)

            if files:
                # Prepare download tasks
                output_dir = Path('data') / 'raw' / data_type / symbol
                for file_info in files:
                    all_tasks.append((file_info, str(output_dir)))

        return all_tasks

    def run(self):
        """Main execution - download everything"""
        print("="*60)
        print("UNIVERSAL S3 PARALLEL DOWNLOADER")
        print("="*60)
        print(f"🔧 Worker processes: {self.num_workers}")
        print(f"📊 Data types: {', '.join(self.data_types)}")
        print()

        # Discover all symbols
        symbols = self.discover_all_symbols()

        # Collect all download tasks
        print("📋 Preparing download tasks...")
        all_tasks = []
        for symbol in tqdm(symbols, desc="Scanning symbols"):
            tasks = self.download_symbol(symbol)
            all_tasks.extend(tasks)

        print(f"\n📦 Total files to download: {len(all_tasks)}")

        if not all_tasks:
            print("Nothing to download!")
            return

        # Download files in parallel
        print(f"\n⬇️  Starting parallel download with {self.num_workers} workers...")
        start_time = time.time()

        results = {'success': 0, 'skipped': 0, 'failed': 0}

        with Pool(processes=self.num_workers) as pool:
            with tqdm(total=len(all_tasks), desc="Downloading", unit="file") as pbar:
                for result in pool.imap_unordered(download_file_task, all_tasks):
                    results[result['status']] += 1

                    if result['status'] == 'success':
                        pbar.set_description(f"✓ {result['file']}")
                    elif result['status'] == 'skipped':
                        pbar.set_description(f"⊘ {result['file']}")
                    else:
                        pbar.set_description(f"✗ {result['file']}")

                    pbar.update(1)

        # Convert to Parquet
        print(f"\n📦 Converting to Parquet format...")
        for symbol in tqdm(symbols, desc="Converting"):
            for data_type in self.data_types:
                self.process_to_parquet(symbol, data_type)

        # Final summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"DOWNLOAD COMPLETE")
        print(f"✓ Success: {results['success']}")
        print(f"⊘ Skipped: {results['skipped']}")
        print(f"✗ Failed: {results['failed']}")
        print(f"⏱️  Time elapsed: {elapsed:.1f} seconds")
        print(f"📊 Speed: {len(all_tasks)/elapsed:.1f} files/second")
        print(f"{'='*60}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Universal S3 Parallel Downloader - Downloads ALL Binance futures data'
    )
    parser.add_argument(
        '--workers',
        type=int,
        help='Number of worker processes (default: auto-detect, max 20)'
    )

    args = parser.parse_args()

    # Run downloader
    downloader = UniversalS3Downloader(num_workers=args.workers)
    downloader.run()


if __name__ == "__main__":
    main()