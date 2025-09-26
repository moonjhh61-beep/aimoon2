"""
Binance Data Vision Downloader
Downloads historical data from Binance public data repository
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import calendar

from .download_tracker import DownloadTracker
from .smart_downloader import SmartDownloader


class DataVisionDownloader:
    """
    Downloads data from Binance Data Vision
    Handles all 5 data types with proper URL construction
    """
    
    BASE_URL = "https://data.binance.vision"
    
    # Data type mappings
    DATA_TYPES = {
        'klines': 'klines',
        'mark_price': 'markPriceKlines', 
        'index_price': 'indexPriceKlines',
        'premium_index': 'premiumIndexKlines',
        'funding_rate': 'fundingRate'
    }
    
    def __init__(self, tracker: DownloadTracker = None):
        """Initialize Data Vision downloader"""
        self.tracker = tracker or DownloadTracker()
        self.downloader = SmartDownloader(tracker=self.tracker)
        
    def download_symbol_history(self, 
                               symbol: str,
                               data_types: List[str] = None,
                               start_date: str = "2019-09-13",
                               end_date: str = None,
                               use_monthly: bool = True) -> Dict[str, List[str]]:
        """
        Download complete history for a symbol
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            data_types: List of data types to download (None = all)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (None = today)
            use_monthly: Use monthly files (True) or daily files (False)
            
        Returns:
            Dictionary mapping data type to list of downloaded files
        """
        if data_types is None:
            data_types = list(self.DATA_TYPES.keys())
        
        if end_date is None:
            # Data Vision has 1-day delay
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        results = {}
        
        for data_type in data_types:
            print(f"\nDownloading {data_type} for {symbol}")
            
            if use_monthly:
                downloaded = self._download_monthly_data(
                    symbol, data_type, start, end
                )
            else:
                downloaded = self._download_daily_data(
                    symbol, data_type, start, end
                )
            
            results[data_type] = downloaded
        
        return results
    
    def _download_monthly_data(self, 
                              symbol: str,
                              data_type: str,
                              start_date: datetime,
                              end_date: datetime) -> List[str]:
        """
        Download monthly aggregated data files
        More efficient for historical data
        """
        downloaded_files = []
        current_date = start_date.replace(day=1)
        
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            
            # Build file info
            file_info = self._build_monthly_file_info(
                symbol, data_type, year, month
            )
            
            if file_info:
                # Check if download needed
                should_download, _ = self.tracker.should_download(file_info['file_id'])
                
                if should_download:
                    # Download file
                    success = self.downloader.download_with_resume(
                        url=file_info['url'],
                        target_path=Path(file_info['target_path']),
                        file_id=file_info['file_id'],
                        metadata=file_info['metadata']
                    )
                    
                    if success:
                        downloaded_files.append(file_info['target_path'])
                else:
                    downloaded_files.append(file_info['target_path'])
            
            # Move to next month
            if month == 12:
                current_date = current_date.replace(year=year + 1, month=1)
            else:
                current_date = current_date.replace(month=month + 1)
        
        return downloaded_files
    
    def _download_daily_data(self,
                            symbol: str,
                            data_type: str,
                            start_date: datetime,
                            end_date: datetime) -> List[str]:
        """
        Download daily data files
        Better for recent data or specific dates
        """
        downloaded_files = []
        current_date = start_date
        
        while current_date <= end_date:
            # Build file info
            file_info = self._build_daily_file_info(
                symbol, data_type, current_date
            )
            
            if file_info:
                # Check if download needed
                should_download, _ = self.tracker.should_download(file_info['file_id'])
                
                if should_download:
                    # Download file
                    success = self.downloader.download_with_resume(
                        url=file_info['url'],
                        target_path=Path(file_info['target_path']),
                        file_id=file_info['file_id'],
                        metadata=file_info['metadata']
                    )
                    
                    if success:
                        downloaded_files.append(file_info['target_path'])
                else:
                    downloaded_files.append(file_info['target_path'])
            
            # Move to next day
            current_date += timedelta(days=1)
        
        return downloaded_files
    
    def _build_monthly_file_info(self, 
                                symbol: str,
                                data_type: str,
                                year: int,
                                month: int) -> Optional[Dict]:
        """
        Build file information for monthly data
        
        Returns:
            Dictionary with url, target_path, file_id, and metadata
        """
        data_type_path = self.DATA_TYPES.get(data_type)
        if not data_type_path:
            return None
        
        # Different path structure for different data types
        if data_type == 'funding_rate':
            # Funding rate doesn't have interval
            filename = f"{symbol}-{data_type_path}-{year}-{month:02d}.zip"
            url_path = f"/data/futures/um/monthly/{data_type_path}/{symbol}/{filename}"
        else:
            # Klines data types have 1h interval
            filename = f"{symbol}-1h-{year}-{month:02d}.zip"
            url_path = f"/data/futures/um/monthly/{data_type_path}/{symbol}/1h/{filename}"
        
        url = f"{self.BASE_URL}{url_path}"
        
        # Local target path
        target_path = f"data/raw/{data_type}/{symbol}/{filename}"
        
        # File ID for tracking
        file_id = f"{symbol}_{data_type}_{year}_{month:02d}"
        
        # Metadata
        metadata = {
            'symbol': symbol,
            'data_type': data_type,
            'year_month': f"{year}-{month:02d}",
            'remote_url': url
        }
        
        return {
            'url': url,
            'target_path': target_path,
            'file_id': file_id,
            'metadata': metadata
        }
    
    def _build_daily_file_info(self,
                              symbol: str,
                              data_type: str,
                              date: datetime) -> Optional[Dict]:
        """
        Build file information for daily data
        
        Returns:
            Dictionary with url, target_path, file_id, and metadata
        """
        data_type_path = self.DATA_TYPES.get(data_type)
        if not data_type_path:
            return None
        
        date_str = date.strftime("%Y-%m-%d")
        
        # Different path structure for different data types
        if data_type == 'funding_rate':
            filename = f"{symbol}-{data_type_path}-{date_str}.zip"
            url_path = f"/data/futures/um/daily/{data_type_path}/{symbol}/{filename}"
        else:
            filename = f"{symbol}-1h-{date_str}.zip"
            url_path = f"/data/futures/um/daily/{data_type_path}/{symbol}/1h/{filename}"
        
        url = f"{self.BASE_URL}{url_path}"
        
        # Local target path
        target_path = f"data/raw/{data_type}/{symbol}/{filename}"
        
        # File ID for tracking
        file_id = f"{symbol}_{data_type}_{date_str}"
        
        # Metadata
        metadata = {
            'symbol': symbol,
            'data_type': data_type,
            'year_month': date.strftime("%Y-%m"),
            'date': date_str,
            'remote_url': url
        }
        
        return {
            'url': url,
            'target_path': target_path,
            'file_id': file_id,
            'metadata': metadata
        }
    
    def download_recent_days(self, 
                           symbol: str,
                           days: int = 7,
                           data_types: List[str] = None) -> Dict[str, List[str]]:
        """
        Download recent days of data (useful for updates)
        
        Args:
            symbol: Trading symbol
            days: Number of recent days to download
            data_types: Data types to download
            
        Returns:
            Dictionary of downloaded files by data type
        """
        end_date = datetime.now() - timedelta(days=1)  # Data Vision has 1-day delay
        start_date = end_date - timedelta(days=days - 1)
        
        return self.download_symbol_history(
            symbol=symbol,
            data_types=data_types,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            use_monthly=False  # Use daily for recent data
        )
    
    def estimate_download_size(self,
                             symbols: List[str],
                             start_date: str = "2019-09-13",
                             end_date: str = None) -> Dict:
        """
        Estimate total download size
        
        Returns:
            Dictionary with size estimates
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Calculate number of months
        months = (end.year - start.year) * 12 + end.month - start.month + 1
        
        # Rough estimates per file (compressed)
        estimates = {
            'klines': 15,  # MB per month
            'mark_price': 10,
            'index_price': 10,
            'premium_index': 8,
            'funding_rate': 1
        }
        
        total_mb = 0
        for data_type, mb_per_month in estimates.items():
            total_mb += mb_per_month * months * len(symbols)
        
        return {
            'total_files': months * len(self.DATA_TYPES) * len(symbols),
            'estimated_size_mb': total_mb,
            'estimated_size_gb': total_mb / 1024,
            'symbols': len(symbols),
            'months': months
        }
    
    def verify_data_availability(self, 
                                symbol: str,
                                date: datetime) -> Dict[str, bool]:
        """
        Check if data is available for a specific symbol and date
        
        Returns:
            Dictionary mapping data type to availability status
        """
        availability = {}
        
        for data_type in self.DATA_TYPES.keys():
            file_info = self._build_daily_file_info(symbol, data_type, date)
            if file_info:
                # Try HEAD request to check if file exists
                import requests
                try:
                    response = requests.head(file_info['url'], timeout=5)
                    availability[data_type] = response.status_code == 200
                except:
                    availability[data_type] = False
            else:
                availability[data_type] = False
        
        return availability