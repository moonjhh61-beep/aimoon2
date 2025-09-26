"""
Symbol Manager for Binance Futures
Manages list of all futures symbols including active and delisted
"""

import requests
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class SymbolManager:
    """
    Manages futures symbol information
    - Fetches active symbols from API
    - Tracks delisted symbols
    - Provides symbol metadata
    """
    
    FUTURES_API = "https://fapi.binance.com/fapi/v1"
    
    def __init__(self, cache_dir: str = "/tmp/crypto_data_metadata"):
        """Initialize symbol manager with cache directory"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.symbols_file = self.cache_dir / "symbols_info.json"
        self._symbols_cache = None
        
    def fetch_active_symbols(self) -> List[Dict]:
        """
        Fetch currently active futures symbols from Binance API
        
        Returns:
            List of symbol information dictionaries
        """
        url = f"{self.FUTURES_API}/exchangeInfo"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            symbols = []
            for symbol_info in data.get('symbols', []):
                if symbol_info['status'] == 'TRADING':
                    # Convert Unix timestamp to date string
                    onboard_timestamp = symbol_info.get('onboardDate')
                    listing_date = None
                    if onboard_timestamp:
                        # Binance uses milliseconds, convert to seconds
                        listing_date = datetime.fromtimestamp(onboard_timestamp / 1000).strftime('%Y-%m-%d')
                    
                    symbols.append({
                        'symbol': symbol_info['symbol'],
                        'base_asset': symbol_info['baseAsset'],
                        'quote_asset': symbol_info['quoteAsset'],
                        'status': symbol_info['status'],
                        'contract_type': symbol_info.get('contractType', 'PERPETUAL'),
                        'listing_date': listing_date,
                        'onboard_timestamp': onboard_timestamp,
                        'filters': symbol_info.get('filters', [])
                    })
            
            print(f"Found {len(symbols)} active futures symbols")
            return symbols
            
        except Exception as e:
            print(f"Error fetching active symbols: {e}")
            return []
    
    def get_all_symbols(self, include_delisted: bool = True) -> List[str]:
        """
        Get list of all symbols
        
        Args:
            include_delisted: Include delisted symbols
            
        Returns:
            List of symbol names
        """
        # Load from cache if exists
        if self._symbols_cache is None:
            self._load_symbols_cache()
        
        symbols = []
        
        # Get active symbols
        active_symbols = self.fetch_active_symbols()
        symbols.extend([s['symbol'] for s in active_symbols])
        
        # Add known delisted symbols if requested
        if include_delisted and self._symbols_cache:
            delisted = self._symbols_cache.get('delisted_symbols', [])
            symbols.extend(delisted)
        
        # Remove duplicates and sort
        symbols = sorted(list(set(symbols)))
        
        # Update cache
        self._update_symbols_cache(symbols)
        
        return symbols
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed information for a specific symbol
        
        Returns:
            Symbol information dictionary or None if not found
        """
        if self._symbols_cache is None:
            self._load_symbols_cache()
        
        if self._symbols_cache and 'symbols' in self._symbols_cache:
            return self._symbols_cache['symbols'].get(symbol)
        
        return None
    
    def get_listing_date(self, symbol: str) -> Optional[str]:
        """
        Get listing date for a symbol
        
        Returns:
            Listing date in YYYY-MM-DD format or default
        """
        # First try to get from cache
        info = self.get_symbol_info(symbol)
        if info and 'listing_date' in info and info['listing_date']:
            return info['listing_date']
        
        # If not in cache, fetch from API and update cache
        print(f"Fetching listing date for {symbol} from API...")
        active_symbols = self.fetch_active_symbols()
        for sym_info in active_symbols:
            if sym_info['symbol'] == symbol:
                listing_date = sym_info.get('listing_date')
                if listing_date:
                    # Update cache with new info
                    if self._symbols_cache is None:
                        self._symbols_cache = {}
                    if 'symbols' not in self._symbols_cache:
                        self._symbols_cache['symbols'] = {}
                    self._symbols_cache['symbols'][symbol] = {
                        'listing_date': listing_date,
                        'onboard_timestamp': sym_info.get('onboard_timestamp'),
                        'status': sym_info.get('status'),
                        'contract_type': sym_info.get('contract_type')
                    }
                    # Save updated cache
                    try:
                        with open(self.symbols_file, 'w') as f:
                            json.dump(self._symbols_cache, f, indent=2)
                    except:
                        pass
                    return listing_date
        
        # For delisted symbols or if not found, default to Binance Futures launch date
        print(f"Warning: Could not find listing date for {symbol}, using default")
        return "2019-09-13"
    
    def _load_symbols_cache(self):
        """Load symbols cache from file"""
        if self.symbols_file.exists():
            try:
                with open(self.symbols_file, 'r') as f:
                    self._symbols_cache = json.load(f)
            except Exception as e:
                print(f"Error loading symbols cache: {e}")
                self._symbols_cache = {}
        else:
            self._symbols_cache = {}
    
    def _update_symbols_cache(self, symbols: List[str]):
        """Update symbols cache file"""
        if self._symbols_cache is None:
            self._symbols_cache = {}
        
        # Update timestamp
        self._symbols_cache['last_updated'] = datetime.now().isoformat()
        
        # Update symbols list
        self._symbols_cache['all_symbols'] = symbols
        
        # Fetch and store detailed symbol info with listing dates
        active_symbols = self.fetch_active_symbols()
        if active_symbols:
            symbols_dict = {}
            for sym_info in active_symbols:
                symbols_dict[sym_info['symbol']] = {
                    'listing_date': sym_info.get('listing_date'),
                    'onboard_timestamp': sym_info.get('onboard_timestamp'),
                    'status': sym_info.get('status'),
                    'contract_type': sym_info.get('contract_type')
                }
            self._symbols_cache['symbols'] = symbols_dict
        
        # Save to file
        try:
            with open(self.symbols_file, 'w') as f:
                json.dump(self._symbols_cache, f, indent=2)
        except Exception as e:
            print(f"Error saving symbols cache: {e}")
    
    def get_priority_symbols(self) -> List[str]:
        """
        Get high-priority symbols for download
        Based on volume and popularity
        """
        return [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'SHIBUSDT', 'DOTUSDT',
            'MATICUSDT', 'LTCUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT',
            'ETCUSDT', 'XLMUSDT', 'FILUSDT', 'TRXUSDT', 'NEARUSDT',
            'ALGOUSDT', 'FTMUSDT', 'SANDUSDT', 'MANAUSDT', 'GALAUSDT',
            'APEUSDT', 'GMTUSDT', 'ROSEUSDT', 'PEOPLEUSDT', 'OPUSDT',
            'ARBUSDT', 'INJUSDT', 'SUIUSDT', 'SEIUSDT', 'TIAUSDT'
        ]
    
    def filter_symbols_by_volume(self, min_volume_usdt: float = 1000000) -> List[str]:
        """
        Filter symbols by minimum daily volume
        
        Args:
            min_volume_usdt: Minimum 24h volume in USDT
            
        Returns:
            List of symbols meeting volume criteria
        """
        url = f"{self.FUTURES_API}/ticker/24hr"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            tickers = response.json()
            
            filtered = []
            for ticker in tickers:
                volume = float(ticker.get('quoteVolume', 0))
                if volume >= min_volume_usdt:
                    filtered.append(ticker['symbol'])
            
            return sorted(filtered)
            
        except Exception as e:
            print(f"Error filtering symbols by volume: {e}")
            return []
    
    def get_delisted_symbols(self) -> List[str]:
        """
        Get known delisted symbols
        These are hardcoded based on historical knowledge
        """
        return [
            'DEFIUSDT', 'BTCSTUSDT', 'CVCUSDT', 'MDTUSDT', 'MITHUSDT',
            'TCTUSDT', 'TORNUSDT', 'KMDUSDT', 'SCUSDT', 'DGBUSDT',
            'NMRUSDT', 'BAKEUSDT', 'BURGERUSDT', 'WINGUSDT', 'SXPUSDT',
            'COCOSUSDT', 'FTTUSDT', 'LUNAUSDT', 'ANCUSDT', 'MIRUSDT',
            'USTUSDT', 'SRMUSDT', 'RAYUSDT', 'C98USDT', 'MASKUSDT'
        ]
    
    def categorize_symbols(self) -> Dict[str, List[str]]:
        """
        Categorize symbols by type/sector
        
        Returns:
            Dictionary with categories as keys and symbol lists as values
        """
        categories = {
            'major': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
            'defi': ['LINKUSDT', 'UNIUSDT', 'AAVEUSDT', 'SNXUSDT', 'COMPUSDT',
                    'MKRUSDT', 'SUSHIUSDT', 'YFIUSDT', 'CRVUSDT', '1INCHUSDT'],
            'layer1': ['SOLUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'ATOMUSDT',
                      'NEARUSDT', 'ALGOUSDT', 'FTMUSDT', 'ICPUSDT', 'EGLDUSDT'],
            'layer2': ['MATICUSDT', 'ARBUSDT', 'OPUSDT', 'IMXUSDT'],
            'meme': ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT'],
            'gaming': ['SANDUSDT', 'MANAUSDT', 'AXSUSDT', 'ENJUSDT', 'GALAUSDT',
                      'ALICEUSDT', 'CHRUSDT', 'TLMUSDT'],
            'ai': ['FETUSDT', 'AGIXUSDT', 'OCEANUSDT', 'RNDRUSDT', 'ROSEUSDT'],
            'storage': ['FILUSDT', 'ARUSDT', 'STORJUSDT'],
            'exchange': ['BNBUSDT', 'OKBUSDT', 'HTUSDT', 'FTTUSDT'],
            'stable_pairs': ['BTCDOMUSDT', 'USDCUSDT']
        }
        
        return categories
    
    def estimate_data_points(self, symbol: str, 
                           start_date: str = "2019-09-13",
                           end_date: str = None) -> int:
        """
        Estimate number of data points for a symbol
        
        Returns:
            Estimated number of hourly data points
        """
        listing_date = self.get_listing_date(symbol)
        start = max(
            datetime.strptime(start_date, "%Y-%m-%d"),
            datetime.strptime(listing_date, "%Y-%m-%d")
        )
        
        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end = datetime.now()
        
        # Calculate hours between dates
        delta = end - start
        hours = int(delta.total_seconds() / 3600)
        
        return hours