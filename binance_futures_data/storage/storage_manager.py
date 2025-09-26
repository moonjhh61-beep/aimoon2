"""
Storage Manager for handling CSV and Parquet files
Manages extraction, conversion, and organization of data
"""

import os
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import shutil
import json


class StorageManager:
    """
    Manages data storage in CSV and Parquet formats
    - Extracts ZIP files to CSV
    - Converts CSV to Parquet
    - Handles data organization and metadata
    """
    
    # Column mappings for standardization
    COLUMN_MAPPINGS = {
        'open_time': 'timestamp',
        'Open time': 'timestamp',
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'Close time': 'close_time',
        'Quote asset volume': 'quote_volume',
        'Number of trades': 'trades',
        'Taker buy base asset volume': 'taker_buy_base',
        'Taker buy quote asset volume': 'taker_buy_quote',
        'Ignore': 'ignore'
    }
    
    def __init__(self, base_dir: str = "data"):
        """Initialize storage manager"""
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.extracted_dir = self.base_dir / "extracted"
        self.processed_dir = self.base_dir / "processed"
        self.metadata_dir = self.base_dir / "metadata"
        
        # Create directories
        for dir_path in [self.raw_dir, self.extracted_dir, 
                         self.processed_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def process_downloaded_files(self, 
                                symbol: str = None,
                                data_type: str = None,
                                cleanup_zip: bool = False) -> Dict:
        """
        Process all downloaded ZIP files
        
        Args:
            symbol: Process only this symbol (None = all)
            data_type: Process only this data type (None = all)
            cleanup_zip: Remove ZIP files after extraction
            
        Returns:
            Processing statistics
        """
        stats = {
            'extracted': 0,
            'converted': 0,
            'errors': []
        }
        
        # Find ZIP files
        pattern = "**/*.zip"
        if symbol and data_type:
            pattern = f"{data_type}/{symbol}/*.zip"
        elif symbol:
            pattern = f"**/{symbol}/*.zip"
        elif data_type:
            pattern = f"{data_type}/**/*.zip"
        
        zip_files = list(self.raw_dir.glob(pattern))
        print(f"Found {len(zip_files)} ZIP files to process")
        
        for zip_path in zip_files:
            try:
                # Extract ZIP to CSV
                csv_path = self.extract_zip(zip_path)
                if csv_path:
                    stats['extracted'] += 1
                    
                    # Convert CSV to Parquet
                    parquet_path = self.convert_to_parquet(csv_path)
                    if parquet_path:
                        stats['converted'] += 1
                    
                    # Cleanup if requested
                    if cleanup_zip:
                        zip_path.unlink()
                        print(f"Removed ZIP: {zip_path}")
                        
            except Exception as e:
                error_msg = f"Error processing {zip_path}: {e}"
                print(error_msg)
                stats['errors'].append(error_msg)
        
        return stats
    
    def extract_zip(self, zip_path: Path) -> Optional[Path]:
        """
        Extract ZIP file to CSV
        
        Returns:
            Path to extracted CSV file
        """
        # Determine output path
        relative_path = zip_path.relative_to(self.raw_dir)
        csv_dir = self.extracted_dir / relative_path.parent
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Get the CSV filename (usually single file in ZIP)
                csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
                
                if not csv_files:
                    print(f"No CSV found in {zip_path}")
                    return None
                
                # Extract first CSV
                csv_filename = csv_files[0]
                zf.extract(csv_filename, csv_dir)
                
                # Rename to match ZIP name
                extracted_path = csv_dir / csv_filename
                target_path = csv_dir / f"{zip_path.stem}.csv"
                
                if extracted_path != target_path:
                    if target_path.exists():
                        target_path.unlink()
                    extracted_path.rename(target_path)
                
                return target_path
                
        except Exception as e:
            print(f"Failed to extract {zip_path}: {e}")
            return None
    
    def convert_to_parquet(self, csv_path: Path) -> Optional[Path]:
        """
        Convert CSV to Parquet format
        
        Returns:
            Path to Parquet file
        """
        # Determine output path
        relative_path = csv_path.relative_to(self.extracted_dir)
        parquet_dir = self.processed_dir / relative_path.parent
        parquet_dir.mkdir(parents=True, exist_ok=True)
        
        parquet_path = parquet_dir / f"{csv_path.stem}.parquet"
        
        # Check if already exists and is newer
        if parquet_path.exists():
            if parquet_path.stat().st_mtime >= csv_path.stat().st_mtime:
                return parquet_path
        
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Standardize columns
            df = self.standardize_dataframe(df, csv_path)
            
            # Save as Parquet
            df.to_parquet(
                parquet_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            
            # Update metadata
            self.update_file_metadata(parquet_path, df)
            
            return parquet_path
            
        except Exception as e:
            print(f"Failed to convert {csv_path}: {e}")
            return None
    
    def standardize_dataframe(self, df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
        """
        Standardize dataframe columns and types
        
        Args:
            df: Input dataframe
            file_path: Source file path for context
            
        Returns:
            Standardized dataframe
        """
        # Rename columns
        df.rename(columns=self.COLUMN_MAPPINGS, inplace=True)
        
        # Determine data type from path
        data_type = None
        for dtype in ['klines', 'mark_price', 'index_price', 
                     'premium_index', 'funding_rate']:
            if dtype in str(file_path):
                data_type = dtype
                break
        
        # Handle timestamp
        if 'timestamp' in df.columns:
            # Check if timestamp is in milliseconds
            if df['timestamp'].dtype in [np.int64, np.float64]:
                if df['timestamp'].iloc[0] > 1e10:  # Milliseconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:  # Seconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Handle funding rate specific columns
        if data_type == 'funding_rate':
            if 'fundingTime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
            if 'fundingRate' in df.columns:
                df['funding_rate'] = pd.to_numeric(df['fundingRate'])
            if 'symbol' not in df.columns:
                # Extract symbol from filename
                symbol = file_path.stem.split('-')[0]
                df['symbol'] = symbol
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                          'quote_volume', 'trades', 'taker_buy_base', 
                          'taker_buy_quote']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add symbol if not present
        if 'symbol' not in df.columns:
            symbol = file_path.parts[-2]  # Get from directory name
            df['symbol'] = symbol
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df.sort_values('timestamp', inplace=True)
        
        # Remove duplicates
        if 'timestamp' in df.columns:
            df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        
        return df
    
    def update_file_metadata(self, file_path: Path, df: pd.DataFrame):
        """Update metadata for processed file"""
        metadata_file = self.metadata_dir / "file_metadata.json"
        
        # Load existing metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Calculate file statistics
        file_key = str(file_path.relative_to(self.base_dir))
        
        stats = {
            'rows': len(df),
            'columns': list(df.columns),
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'processed_at': datetime.now().isoformat()
        }
        
        if 'timestamp' in df.columns:
            stats['date_range'] = {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            }
        
        metadata[file_key] = stats
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_data_summary(self) -> Dict:
        """
        Get summary of all stored data
        
        Returns:
            Summary statistics
        """
        summary = {
            'raw_files': 0,
            'extracted_files': 0,
            'processed_files': 0,
            'total_size_gb': 0,
            'symbols': set(),
            'data_types': set(),
            'date_range': {}
        }
        
        # Count files
        summary['raw_files'] = len(list(self.raw_dir.glob("**/*.zip")))
        summary['extracted_files'] = len(list(self.extracted_dir.glob("**/*.csv")))
        summary['processed_files'] = len(list(self.processed_dir.glob("**/*.parquet")))
        
        # Calculate total size
        for file_path in self.base_dir.glob("**/*"):
            if file_path.is_file():
                summary['total_size_gb'] += file_path.stat().st_size / (1024**3)
        
        # Get unique symbols and data types
        for parquet_file in self.processed_dir.glob("**/*.parquet"):
            parts = parquet_file.parts
            
            # Extract data type and symbol from path
            for dtype in ['klines', 'mark_price', 'index_price', 
                         'premium_index', 'funding_rate']:
                if dtype in parts:
                    summary['data_types'].add(dtype)
                    idx = parts.index(dtype)
                    if idx + 1 < len(parts):
                        summary['symbols'].add(parts[idx + 1])
        
        # Convert sets to lists for JSON serialization
        summary['symbols'] = sorted(list(summary['symbols']))
        summary['data_types'] = sorted(list(summary['data_types']))
        
        return summary
    
    def verify_data_integrity(self, file_path: Path) -> Tuple[bool, List[str]]:
        """
        Verify integrity of a data file
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            else:
                return False, ["Unsupported file format"]
            
            # Check for required columns
            if 'timestamp' not in df.columns:
                issues.append("Missing timestamp column")
            
            # Check for empty dataframe
            if len(df) == 0:
                issues.append("Empty dataframe")
            
            # Check for duplicates
            if 'timestamp' in df.columns:
                duplicates = df.duplicated(subset=['timestamp']).sum()
                if duplicates > 0:
                    issues.append(f"Found {duplicates} duplicate timestamps")
            
            # Check for time gaps (for hourly data)
            if 'timestamp' in df.columns and len(df) > 1:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                time_diff = df['timestamp'].diff()
                
                # Check for gaps > 1.5 hours (allowing for some tolerance)
                gaps = time_diff[time_diff > pd.Timedelta(hours=1.5)]
                if len(gaps) > 0:
                    issues.append(f"Found {len(gaps)} time gaps")
            
            # Check OHLC relationships (if applicable)
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                invalid_ohlc = df[
                    (df['high'] < df['low']) |
                    (df['high'] < df['open']) |
                    (df['high'] < df['close']) |
                    (df['low'] > df['open']) |
                    (df['low'] > df['close'])
                ]
                if len(invalid_ohlc) > 0:
                    issues.append(f"Found {len(invalid_ohlc)} invalid OHLC relationships")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Error reading file: {e}"]
    
    def cleanup_old_files(self, days_old: int = 30):
        """
        Clean up old temporary files
        
        Args:
            days_old: Remove files older than this many days
        """
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (days_old * 86400)
        
        removed_count = 0
        
        # Clean extracted CSV files if Parquet exists
        for csv_file in self.extracted_dir.glob("**/*.csv"):
            # Check if corresponding Parquet exists
            relative_path = csv_file.relative_to(self.extracted_dir)
            parquet_file = self.processed_dir / relative_path.with_suffix('.parquet')
            
            if parquet_file.exists():
                # Remove CSV if Parquet is newer
                if parquet_file.stat().st_mtime > csv_file.stat().st_mtime:
                    csv_file.unlink()
                    removed_count += 1
        
        print(f"Cleaned up {removed_count} redundant files")