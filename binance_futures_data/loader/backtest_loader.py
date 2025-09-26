"""
Backtest Data Loader
Efficient loading of processed data for backtesting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


class BacktestDataLoader:
    """
    High-performance data loader for backtesting
    - Loads from Parquet files
    - Memory-efficient chunking
    - Caching for repeated loads
    - Multi-symbol support
    """
    
    def __init__(self, base_dir: str = "data", cache_size_mb: int = 1000):
        """
        Initialize backtest loader
        
        Args:
            base_dir: Base data directory
            cache_size_mb: Maximum cache size in MB
        """
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / "processed"
        self.metadata_dir = self.base_dir / "metadata"
        
        # Cache for loaded data
        self.cache = {}
        self.cache_size_mb = cache_size_mb
        self.current_cache_size_mb = 0
    
    def load_data(self,
                  symbols: Union[str, List[str]],
                  start_date: str,
                  end_date: str,
                  data_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for backtesting
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data_types: List of data types to load (None = all)
            
        Returns:
            Dictionary mapping data type to DataFrame
        """
        # Normalize inputs
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if data_types is None:
            data_types = ['klines', 'mark_price', 'index_price', 
                         'premium_index', 'funding_rate']
        
        result = {}
        
        # Load each data type
        for data_type in data_types:
            print(f"Loading {data_type} data...")
            
            if data_type == 'funding_rate':
                # Funding rate needs special handling (different frequency)
                df = self._load_funding_rate(symbols, start_date, end_date)
            else:
                df = self._load_hourly_data(data_type, symbols, start_date, end_date)
            
            if not df.empty:
                result[data_type] = df
                print(f"  Loaded {len(df):,} rows for {data_type}")
        
        return result
    
    def _load_hourly_data(self,
                         data_type: str,
                         symbols: List[str],
                         start_date: str,
                         end_date: str) -> pd.DataFrame:
        """Load hourly data (klines, mark_price, index_price, premium_index)"""
        
        # Check cache first
        cache_key = f"{data_type}_{','.join(symbols)}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        all_data = []
        
        # Load data for each symbol
        for symbol in symbols:
            symbol_data = self._load_symbol_data(
                data_type, symbol, start_date, end_date
            )
            if not symbol_data.empty:
                all_data.append(symbol_data)
        
        if all_data:
            # Combine all symbol data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Sort by timestamp and symbol
            combined_df.sort_values(['timestamp', 'symbol'], inplace=True)
            
            # Cache if small enough
            self._cache_dataframe(cache_key, combined_df)
            
            return combined_df
        
        return pd.DataFrame()
    
    def _load_funding_rate(self,
                          symbols: List[str],
                          start_date: str,
                          end_date: str) -> pd.DataFrame:
        """Load funding rate data (8-hour frequency)"""
        
        cache_key = f"funding_rate_{','.join(symbols)}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        all_data = []
        
        for symbol in symbols:
            symbol_data = self._load_symbol_data(
                'funding_rate', symbol, start_date, end_date
            )
            if not symbol_data.empty:
                all_data.append(symbol_data)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df.sort_values(['timestamp', 'symbol'], inplace=True)
            
            # Cache if small enough
            self._cache_dataframe(cache_key, combined_df)
            
            return combined_df
        
        return pd.DataFrame()
    
    def _load_symbol_data(self,
                         data_type: str,
                         symbol: str,
                         start_date: str,
                         end_date: str) -> pd.DataFrame:
        """Load data for a single symbol"""
        
        data_path = self.processed_dir / data_type / symbol
        
        if not data_path.exists():
            return pd.DataFrame()
        
        # Parse dates with UTC timezone
        start_dt = pd.Timestamp(start_date, tz='UTC')
        end_dt = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        # Find relevant files based on naming convention
        relevant_files = self._find_relevant_files(
            data_path, start_dt, end_dt
        )
        
        if not relevant_files:
            return pd.DataFrame()
        
        # Load and combine files
        dfs = []
        for file_path in relevant_files:
            try:
                df = pd.read_parquet(file_path)
                
                # Ensure timestamp column with UTC timezone
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    
                    # Filter by date range
                    mask = (df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)
                    df = df[mask]
                
                # Add symbol if not present
                if 'symbol' not in df.columns:
                    df['symbol'] = symbol
                
                if not df.empty:
                    dfs.append(df)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            
            # Remove duplicates
            if 'timestamp' in combined.columns:
                combined.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
                combined.sort_values('timestamp', inplace=True)
            
            return combined
        
        return pd.DataFrame()
    
    def _find_relevant_files(self,
                            data_path: Path,
                            start_dt: pd.Timestamp,
                            end_dt: pd.Timestamp) -> List[Path]:
        """Find files that might contain data in the date range"""
        
        relevant_files = []
        
        for file_path in sorted(data_path.glob("*.parquet")):
            # Parse year and month from filename
            # Expected format: SYMBOL-1h-YYYY-MM.parquet
            parts = file_path.stem.split('-')
            
            try:
                if len(parts) >= 4:
                    # Format: SYMBOL-1h-YYYY-MM
                    year = int(parts[-2])
                    month = int(parts[-1])
                    
                    # Check if this file might contain relevant data
                    file_start = pd.Timestamp(year=year, month=month, day=1, tz='UTC')
                    file_end = file_start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
                    
                    # Check for overlap with requested range
                    if not (file_end < start_dt or file_start > end_dt):
                        relevant_files.append(file_path)
                elif len(parts) == 3:
                    # Handle daily files (SYMBOL-YYYY-MM-DD format)
                    try:
                        date_str = '-'.join(parts)
                        file_date = pd.Timestamp(date_str, tz='UTC')
                        
                        if start_dt <= file_date <= end_dt:
                            relevant_files.append(file_path)
                    except:
                        pass
                else:
                    # Unknown format, include it to be safe
                    relevant_files.append(file_path)
                    
            except (ValueError, IndexError):
                # Can't parse date, include file to be safe
                relevant_files.append(file_path)
        
        return relevant_files
    
    def load_universe(self,
                     start_date: str,
                     end_date: str,
                     min_volume: float = 1000000,
                     data_types: List[str] = None) -> pd.DataFrame:
        """
        Load entire universe of symbols
        
        Args:
            start_date: Start date
            end_date: End date
            min_volume: Minimum average daily volume filter
            data_types: Data types to load
            
        Returns:
            Combined DataFrame with all symbols
        """
        # Get all available symbols
        symbols = self.get_available_symbols()
        
        print(f"Loading universe of {len(symbols)} symbols...")
        
        # Load data
        data = self.load_data(symbols, start_date, end_date, data_types)
        
        # Apply volume filter if we have klines data
        if 'klines' in data and min_volume > 0:
            klines_df = data['klines']
            
            if 'quote_volume' in klines_df.columns:
                # Calculate average daily volume per symbol
                daily_volume = klines_df.groupby('symbol')['quote_volume'].mean()
                
                # Filter symbols
                valid_symbols = daily_volume[daily_volume >= min_volume].index.tolist()
                
                print(f"Filtered to {len(valid_symbols)} symbols with volume >= {min_volume:,.0f}")
                
                # Filter all dataframes
                for data_type, df in data.items():
                    if 'symbol' in df.columns:
                        data[data_type] = df[df['symbol'].isin(valid_symbols)]
        
        return data
    
    def get_available_symbols(self, data_type: str = 'klines') -> List[str]:
        """
        Get list of available symbols
        
        Args:
            data_type: Data type to check for availability
            
        Returns:
            List of available symbols
        """
        data_path = self.processed_dir / data_type
        
        if not data_path.exists():
            return []
        
        symbols = [d.name for d in data_path.iterdir() if d.is_dir()]
        
        return sorted(symbols)
    
    def get_date_range(self, symbol: str, data_type: str = 'klines') -> Optional[Dict]:
        """
        Get available date range for a symbol
        
        Returns:
            Dictionary with start and end dates
        """
        data_path = self.processed_dir / data_type / symbol
        
        if not data_path.exists():
            return None
        
        # Get all files
        files = sorted(data_path.glob("*.parquet"))
        
        if not files:
            return None
        
        # Read first and last file to get date range
        try:
            first_df = pd.read_parquet(files[0], columns=['timestamp'])
            last_df = pd.read_parquet(files[-1], columns=['timestamp'])
            
            first_df['timestamp'] = pd.to_datetime(first_df['timestamp'])
            last_df['timestamp'] = pd.to_datetime(last_df['timestamp'])
            
            return {
                'start': first_df['timestamp'].min().strftime('%Y-%m-%d'),
                'end': last_df['timestamp'].max().strftime('%Y-%m-%d'),
                'files': len(files)
            }
        except:
            return None
    
    def _cache_dataframe(self, key: str, df: pd.DataFrame):
        """Cache a dataframe if it fits in cache"""
        
        # Calculate dataframe size in MB
        df_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Check if it fits in cache
        if df_size_mb > self.cache_size_mb:
            return  # Too large for cache
        
        # Make room if needed
        while self.current_cache_size_mb + df_size_mb > self.cache_size_mb:
            if not self.cache:
                break
            
            # Remove oldest item (simple FIFO)
            oldest_key = next(iter(self.cache))
            oldest_df = self.cache.pop(oldest_key)
            self.current_cache_size_mb -= oldest_df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Add to cache
        self.cache[key] = df
        self.current_cache_size_mb += df_size_mb
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
        self.current_cache_size_mb = 0
    
    def load_aligned_data(self,
                         symbols: Union[str, List[str]],
                         start_date: str,
                         end_date: str) -> pd.DataFrame:
        """
        Load all data types and align them into single DataFrame
        
        Returns:
            Aligned DataFrame with all data types as columns
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Load all data types
        data = self.load_data(symbols, start_date, end_date)
        
        if not data:
            return pd.DataFrame()
        
        # Start with klines as base
        if 'klines' not in data:
            print("Warning: No klines data available")
            return pd.DataFrame()
        
        result = data['klines'].copy()
        
        # Merge other hourly data
        for data_type in ['mark_price', 'index_price', 'premium_index']:
            if data_type in data:
                df = data[data_type].copy()
                
                # Rename columns to avoid conflicts
                rename_cols = {}
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        rename_cols[col] = f"{data_type}_{col}"
                
                df.rename(columns=rename_cols, inplace=True)
                
                # Merge on timestamp and symbol
                result = pd.merge(
                    result, df,
                    on=['timestamp', 'symbol'],
                    how='left',
                    suffixes=('', f'_{data_type}')
                )
        
        # Handle funding rate separately (different frequency)
        if 'funding_rate' in data:
            funding_df = data['funding_rate'].copy()
            
            # Forward fill funding rate to hourly data
            # This represents the current funding rate at each hour
            result = pd.merge_asof(
                result.sort_values('timestamp'),
                funding_df[['timestamp', 'symbol', 'funding_rate']].sort_values('timestamp'),
                on='timestamp',
                by='symbol',
                direction='backward'
            )
        
        return result
    
    def get_metadata(self) -> Dict:
        """Get metadata about available data"""
        
        metadata = {
            'symbols': {},
            'summary': {
                'total_symbols': 0,
                'total_files': 0,
                'total_size_gb': 0
            }
        }
        
        # Scan all data types
        for data_type in ['klines', 'mark_price', 'index_price', 
                         'premium_index', 'funding_rate']:
            
            data_path = self.processed_dir / data_type
            if not data_path.exists():
                continue
            
            for symbol_dir in data_path.iterdir():
                if not symbol_dir.is_dir():
                    continue
                
                symbol = symbol_dir.name
                
                if symbol not in metadata['symbols']:
                    metadata['symbols'][symbol] = {}
                
                # Count files and size
                files = list(symbol_dir.glob("*.parquet"))
                total_size = sum(f.stat().st_size for f in files)
                
                metadata['symbols'][symbol][data_type] = {
                    'files': len(files),
                    'size_mb': total_size / (1024 * 1024)
                }
                
                metadata['summary']['total_files'] += len(files)
                metadata['summary']['total_size_gb'] += total_size / (1024**3)
        
        metadata['summary']['total_symbols'] = len(metadata['symbols'])
        
        return metadata