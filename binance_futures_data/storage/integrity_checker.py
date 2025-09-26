"""
Data Integrity Checker
Validates and verifies data completeness and quality
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json


class IntegrityChecker:
    """
    Comprehensive data integrity checking
    - Validates data completeness
    - Detects anomalies
    - Identifies missing periods
    - Suggests fixes
    """
    
    def __init__(self, base_dir: str = "data"):
        """Initialize integrity checker"""
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / "processed"
        self.metadata_dir = self.base_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def check_symbol_completeness(self, 
                                 symbol: str,
                                 start_date: str = "2019-09-13",
                                 end_date: str = None) -> Dict:
        """
        Check data completeness for a symbol
        
        Args:
            symbol: Trading symbol
            start_date: Expected start date
            end_date: Expected end date (None = today)
            
        Returns:
            Completeness report
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        report = {
            'symbol': symbol,
            'period': f"{start_date} to {end_date}",
            'data_types': {},
            'overall_completeness': 0,
            'issues': []
        }
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Check each data type
        for data_type in ['klines', 'mark_price', 'index_price', 
                         'premium_index', 'funding_rate']:
            
            type_report = self._check_data_type_completeness(
                symbol, data_type, start_dt, end_dt
            )
            report['data_types'][data_type] = type_report
        
        # Calculate overall completeness
        completeness_scores = [
            dt['completeness_percent'] 
            for dt in report['data_types'].values()
        ]
        if completeness_scores:
            report['overall_completeness'] = np.mean(completeness_scores)
        
        # Identify major issues
        for data_type, type_report in report['data_types'].items():
            if type_report['missing_periods']:
                report['issues'].append(
                    f"{data_type}: {len(type_report['missing_periods'])} missing periods"
                )
        
        return report
    
    def _check_data_type_completeness(self,
                                     symbol: str,
                                     data_type: str,
                                     start_date: datetime,
                                     end_date: datetime) -> Dict:
        """Check completeness for specific data type"""
        
        # Find all files for this symbol and data type
        data_path = self.processed_dir / data_type / symbol
        
        if not data_path.exists():
            return {
                'exists': False,
                'completeness_percent': 0,
                'expected_points': 0,
                'actual_points': 0,
                'missing_periods': [(start_date.strftime("%Y-%m-%d"), 
                                   end_date.strftime("%Y-%m-%d"))]
            }
        
        # Load all parquet files
        all_data = []
        for parquet_file in sorted(data_path.glob("*.parquet")):
            try:
                df = pd.read_parquet(parquet_file, columns=['timestamp'])
                all_data.append(df)
            except:
                continue
        
        if not all_data:
            return {
                'exists': True,
                'completeness_percent': 0,
                'expected_points': 0,
                'actual_points': 0,
                'missing_periods': [(start_date.strftime("%Y-%m-%d"),
                                   end_date.strftime("%Y-%m-%d"))]
            }
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df.drop_duplicates(subset=['timestamp'], inplace=True)
        combined_df.sort_values('timestamp', inplace=True)
        
        # Calculate expected vs actual points
        if data_type == 'funding_rate':
            # Funding rate is typically every 8 hours
            expected_points = int((end_date - start_date).total_seconds() / (8 * 3600))
        else:
            # Hourly data
            expected_points = int((end_date - start_date).total_seconds() / 3600)
        
        actual_points = len(combined_df)
        
        # Find missing periods
        missing_periods = self._find_missing_periods(
            combined_df, start_date, end_date, data_type
        )
        
        completeness_percent = (actual_points / expected_points * 100) if expected_points > 0 else 0
        
        return {
            'exists': True,
            'completeness_percent': min(completeness_percent, 100),
            'expected_points': expected_points,
            'actual_points': actual_points,
            'missing_periods': missing_periods,
            'actual_date_range': {
                'start': combined_df['timestamp'].min().strftime("%Y-%m-%d %H:%M"),
                'end': combined_df['timestamp'].max().strftime("%Y-%m-%d %H:%M")
            } if len(combined_df) > 0 else None
        }
    
    def _find_missing_periods(self,
                             df: pd.DataFrame,
                             start_date: datetime,
                             end_date: datetime,
                             data_type: str) -> List[Tuple[str, str]]:
        """Find missing time periods in data"""
        
        if len(df) == 0:
            return [(start_date.strftime("%Y-%m-%d"), 
                    end_date.strftime("%Y-%m-%d"))]
        
        missing_periods = []
        
        # Expected frequency
        if data_type == 'funding_rate':
            freq = pd.Timedelta(hours=8)
        else:
            freq = pd.Timedelta(hours=1)
        
        # Check for gaps
        df_timestamps = set(df['timestamp'])
        current = start_date
        gap_start = None
        
        while current <= end_date:
            if current not in df_timestamps:
                if gap_start is None:
                    gap_start = current
            else:
                if gap_start is not None:
                    # End of gap
                    gap_end = current - freq
                    if (gap_end - gap_start).total_seconds() > freq.total_seconds():
                        missing_periods.append((
                            gap_start.strftime("%Y-%m-%d %H:%M"),
                            gap_end.strftime("%Y-%m-%d %H:%M")
                        ))
                    gap_start = None
            
            current += freq
        
        # Check if gap extends to end
        if gap_start is not None:
            missing_periods.append((
                gap_start.strftime("%Y-%m-%d %H:%M"),
                end_date.strftime("%Y-%m-%d %H:%M")
            ))
        
        return missing_periods
    
    def validate_data_quality(self, file_path: Path) -> Dict:
        """
        Validate data quality of a file
        
        Returns:
            Quality report with issues and statistics
        """
        report = {
            'file': str(file_path),
            'valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Load data
            df = pd.read_parquet(file_path)
            
            # Basic statistics
            report['statistics'] = {
                'rows': len(df),
                'columns': list(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
            }
            
            # Check for required columns
            if 'timestamp' not in df.columns:
                report['issues'].append("Missing timestamp column")
                report['valid'] = False
            
            # Check for empty data
            if len(df) == 0:
                report['issues'].append("Empty dataframe")
                report['valid'] = False
                return report
            
            # Timestamp checks
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Check for duplicates
                duplicates = df.duplicated(subset=['timestamp']).sum()
                if duplicates > 0:
                    report['issues'].append(f"Found {duplicates} duplicate timestamps")
                    report['valid'] = False
                
                # Check chronological order
                if not df['timestamp'].is_monotonic_increasing:
                    report['warnings'].append("Timestamps not in chronological order")
                
                # Check for future timestamps
                future_timestamps = df[df['timestamp'] > datetime.now()]
                if len(future_timestamps) > 0:
                    report['issues'].append(f"Found {len(future_timestamps)} future timestamps")
                    report['valid'] = False
            
            # OHLC validation
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # Check OHLC relationships
                invalid_ohlc = df[
                    (df['high'] < df['low']) |
                    (df['high'] < df['open']) |
                    (df['high'] < df['close']) |
                    (df['low'] > df['open']) |
                    (df['low'] > df['close'])
                ]
                
                if len(invalid_ohlc) > 0:
                    report['issues'].append(f"Found {len(invalid_ohlc)} invalid OHLC relationships")
                    report['valid'] = False
                
                # Check for zero/negative prices
                zero_prices = df[
                    (df['open'] <= 0) | (df['high'] <= 0) | 
                    (df['low'] <= 0) | (df['close'] <= 0)
                ]
                
                if len(zero_prices) > 0:
                    report['issues'].append(f"Found {len(zero_prices)} zero/negative prices")
                    report['valid'] = False
                
                # Check for extreme price movements (>50% in 1 hour)
                if len(df) > 1:
                    price_changes = df['close'].pct_change().abs()
                    extreme_moves = price_changes[price_changes > 0.5]
                    
                    if len(extreme_moves) > 0:
                        report['warnings'].append(
                            f"Found {len(extreme_moves)} extreme price moves (>50%)"
                        )
            
            # Volume validation
            if 'volume' in df.columns:
                # Check for negative volume
                negative_volume = df[df['volume'] < 0]
                if len(negative_volume) > 0:
                    report['issues'].append(f"Found {len(negative_volume)} negative volumes")
                    report['valid'] = False
                
                # Check for suspicious volume patterns
                if len(df) > 24:  # Need at least 24 hours
                    volume_mean = df['volume'].rolling(24).mean()
                    volume_spikes = df[df['volume'] > volume_mean * 100]
                    
                    if len(volume_spikes) > 0:
                        report['warnings'].append(
                            f"Found {len(volume_spikes)} suspicious volume spikes"
                        )
            
            # Funding rate validation
            if 'funding_rate' in df.columns or 'fundingRate' in df.columns:
                funding_col = 'funding_rate' if 'funding_rate' in df.columns else 'fundingRate'
                
                # Check for extreme funding rates (>1% per 8 hours)
                extreme_funding = df[df[funding_col].abs() > 0.01]
                if len(extreme_funding) > 0:
                    report['warnings'].append(
                        f"Found {len(extreme_funding)} extreme funding rates (>1%)"
                    )
            
        except Exception as e:
            report['issues'].append(f"Error reading file: {e}")
            report['valid'] = False
        
        return report
    
    def generate_integrity_report(self, symbols: List[str] = None) -> Dict:
        """
        Generate comprehensive integrity report for all data
        
        Args:
            symbols: List of symbols to check (None = all)
            
        Returns:
            Full integrity report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'symbols': {},
            'summary': {
                'total_symbols': 0,
                'complete_symbols': 0,
                'partial_symbols': 0,
                'missing_symbols': 0,
                'total_issues': 0
            }
        }
        
        # Get all symbols if not specified
        if symbols is None:
            symbols = set()
            for data_type in ['klines', 'mark_price', 'index_price',
                            'premium_index', 'funding_rate']:
                type_path = self.processed_dir / data_type
                if type_path.exists():
                    symbols.update([d.name for d in type_path.iterdir() if d.is_dir()])
            symbols = sorted(list(symbols))
        
        # Check each symbol
        for symbol in symbols:
            symbol_report = self.check_symbol_completeness(symbol)
            report['symbols'][symbol] = symbol_report
            
            # Update summary
            report['summary']['total_symbols'] += 1
            
            if symbol_report['overall_completeness'] >= 95:
                report['summary']['complete_symbols'] += 1
            elif symbol_report['overall_completeness'] > 0:
                report['summary']['partial_symbols'] += 1
            else:
                report['summary']['missing_symbols'] += 1
            
            report['summary']['total_issues'] += len(symbol_report['issues'])
        
        # Save report
        report_file = self.metadata_dir / f"integrity_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Integrity report saved to {report_file}")
        
        return report
    
    def suggest_fixes(self, symbol: str) -> List[Dict]:
        """
        Suggest fixes for data issues
        
        Returns:
            List of suggested actions
        """
        suggestions = []
        
        # Check completeness
        completeness = self.check_symbol_completeness(symbol)
        
        for data_type, type_report in completeness['data_types'].items():
            if type_report['missing_periods']:
                for start, end in type_report['missing_periods']:
                    suggestions.append({
                        'action': 'download',
                        'data_type': data_type,
                        'symbol': symbol,
                        'period': f"{start} to {end}",
                        'priority': 'high' if type_report['completeness_percent'] < 50 else 'medium'
                    })
        
        # Check for quality issues
        data_path = self.processed_dir / 'klines' / symbol
        if data_path.exists():
            for parquet_file in data_path.glob("*.parquet"):
                quality = self.validate_data_quality(parquet_file)
                
                if not quality['valid']:
                    suggestions.append({
                        'action': 're-download',
                        'file': str(parquet_file),
                        'reason': quality['issues'],
                        'priority': 'high'
                    })
        
        return suggestions