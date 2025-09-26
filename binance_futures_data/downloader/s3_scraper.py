"""
S3-based directory scraper for Binance Data Vision
Lists all available files directly from S3 bucket
"""

import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re


class S3DirectoryScraper:
    """
    Scrapes Binance Data Vision S3 bucket for available files
    Eliminates 404 errors by only downloading files that exist
    """
    
    S3_BUCKET_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
    
    def __init__(self):
        """Initialize S3 scraper"""
        self.session = requests.Session()
    
    def list_directory(self, prefix: str, delimiter: str = "/") -> List[Dict]:
        """
        List all files in an S3 directory
        
        Args:
            prefix: S3 prefix path (e.g., 'data/futures/um/monthly/klines/BTCUSDT/1h/')
            delimiter: Delimiter for directory listing
            
        Returns:
            List of file information dictionaries
        """
        files = []
        marker = None
        is_truncated = True
        
        while is_truncated:
            # Build request parameters
            params = {
                'prefix': prefix,
                'delimiter': delimiter,
                'max-keys': 1000
            }
            if marker:
                params['marker'] = marker
            
            try:
                # Make request to S3
                response = self.session.get(self.S3_BUCKET_URL, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse XML response - handle namespace
                # Remove namespace for easier parsing
                xml_text = response.text.replace('xmlns="http://s3.amazonaws.com/doc/2006-03-01/"', '')
                root = ET.fromstring(xml_text)
                
                # Extract file information
                for content in root.findall('.//Contents'):
                    key = content.find('Key')
                    last_modified = content.find('LastModified')
                    etag = content.find('ETag')
                    size = content.find('Size')
                    
                    if key is not None:
                        file_info = {
                            'key': key.text,
                            'filename': Path(key.text).name,
                            'last_modified': last_modified.text if last_modified is not None else None,
                            'etag': etag.text.strip('"') if etag is not None else None,
                            'size': int(size.text) if size is not None else 0
                        }
                        files.append(file_info)
                        marker = key.text
                
                # Check if truncated
                is_truncated_elem = root.find('IsTruncated')
                is_truncated = is_truncated_elem is not None and is_truncated_elem.text.lower() == 'true'
                
            except Exception as e:
                print(f"Error listing directory {prefix}: {e}")
                break
        
        return files
    
    def list_symbol_files(self, 
                         symbol: str,
                         data_type: str = 'klines',
                         interval: str = '1h') -> List[Dict]:
        """
        List all files for a specific symbol and data type
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            data_type: Type of data ('klines', 'fundingRate', etc.)
            interval: Time interval for klines data
            
        Returns:
            List of available files with metadata
        """
        # Build S3 prefix based on data type
        if data_type == 'klines':
            prefix = f"data/futures/um/monthly/klines/{symbol}/{interval}/"
        elif data_type == 'fundingRate':
            prefix = f"data/futures/um/monthly/fundingRate/{symbol}/"
        elif data_type == 'markPriceKlines':
            prefix = f"data/futures/um/monthly/markPriceKlines/{symbol}/{interval}/"
        elif data_type == 'indexPriceKlines':
            prefix = f"data/futures/um/monthly/indexPriceKlines/{symbol}/{interval}/"
        elif data_type == 'premiumIndexKlines':
            prefix = f"data/futures/um/monthly/premiumIndexKlines/{symbol}/{interval}/"
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Get files from S3
        files = self.list_directory(prefix)
        
        # Filter out checksum files and parse dates
        data_files = []
        for file in files:
            if not file['filename'].endswith('.CHECKSUM'):
                # Extract date from filename (e.g., BTCUSDT-1h-2020-01.zip)
                match = re.search(r'(\d{4})-(\d{2})\.zip$', file['filename'])
                if match:
                    year, month = match.groups()
                    file['year'] = int(year)
                    file['month'] = int(month)
                    file['date'] = f"{year}-{month}"
                    data_files.append(file)
        
        # Sort by date
        data_files.sort(key=lambda x: (x.get('year', 0), x.get('month', 0)))
        
        return data_files
    
    def get_symbol_date_range(self, symbol: str, data_type: str = 'klines') -> Tuple[str, str]:
        """
        Get the available date range for a symbol
        
        Args:
            symbol: Trading symbol
            data_type: Type of data
            
        Returns:
            Tuple of (start_date, end_date) in YYYY-MM format
        """
        files = self.list_symbol_files(symbol, data_type)
        
        if not files:
            return None, None
        
        start_date = files[0].get('date')
        end_date = files[-1].get('date')
        
        return start_date, end_date
    
    def discover_all_symbols(self, data_type: str = 'klines') -> List[str]:
        """
        Discover all available symbols by listing directories
        
        Args:
            data_type: Type of data to check
            
        Returns:
            List of all available symbols
        """
        # List the parent directory
        if data_type == 'klines':
            prefix = "data/futures/um/monthly/klines/"
        elif data_type == 'fundingRate':
            prefix = "data/futures/um/monthly/fundingRate/"
        else:
            prefix = f"data/futures/um/monthly/{data_type}/"
        
        # Get directory listing (using delimiter to get subdirectories)
        response = self.session.get(
            self.S3_BUCKET_URL,
            params={
                'prefix': prefix,
                'delimiter': '/',
                'max-keys': 1000
            },
            timeout=30
        )
        
        symbols = []
        if response.status_code == 200:
            # Remove namespace for easier parsing
            xml_text = response.text.replace('xmlns="http://s3.amazonaws.com/doc/2006-03-01/"', '')
            root = ET.fromstring(xml_text)
            
            # CommonPrefixes contains subdirectories
            for prefix_elem in root.findall('.//CommonPrefixes'):
                prefix_text = prefix_elem.find('Prefix')
                if prefix_text is not None:
                    # Extract symbol from path
                    path_parts = prefix_text.text.rstrip('/').split('/')
                    if len(path_parts) > 0:
                        symbol = path_parts[-1]
                        if symbol and symbol.endswith('USDT'):
                            symbols.append(symbol)
        
        return sorted(symbols)
    
    def get_file_url(self, file_key: str) -> str:
        """
        Get download URL for a file
        
        Args:
            file_key: S3 key for the file
            
        Returns:
            Download URL
        """
        return f"https://data.binance.vision/{file_key}"
    
    def validate_checksum(self, symbol: str, 
                         data_type: str,
                         year: int,
                         month: int) -> Optional[str]:
        """
        Get checksum for a specific file
        
        Args:
            symbol: Trading symbol
            data_type: Type of data
            year: Year
            month: Month
            
        Returns:
            Checksum string if available
        """
        # Build checksum file path
        if data_type == 'klines':
            checksum_key = f"data/futures/um/monthly/klines/{symbol}/1h/{symbol}-1h-{year:04d}-{month:02d}.zip.CHECKSUM"
        elif data_type == 'fundingRate':
            checksum_key = f"data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{year:04d}-{month:02d}.zip.CHECKSUM"
        else:
            return None
        
        try:
            response = self.session.get(self.get_file_url(checksum_key), timeout=10)
            if response.status_code == 200:
                # Parse checksum file (format: "SHA256  filename")
                lines = response.text.strip().split('\n')
                if lines:
                    parts = lines[0].split()
                    if len(parts) >= 2:
                        return parts[0]  # Return the checksum
        except:
            pass
        
        return None