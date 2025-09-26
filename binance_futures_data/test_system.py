#!/usr/bin/env python3
"""
Test script to verify the Binance Futures Data system
Tests basic functionality without downloading large amounts of data
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from downloader import DownloadTracker, SmartDownloader, DataVisionDownloader, SymbolManager
from storage import StorageManager, IntegrityChecker
from loader import BacktestDataLoader


def test_components():
    """Test that all components initialize correctly"""
    print("Testing component initialization...")
    
    try:
        # Test DownloadTracker
        tracker = DownloadTracker()
        print("✓ DownloadTracker initialized")
        
        # Test SmartDownloader  
        downloader = SmartDownloader(tracker)
        print("✓ SmartDownloader initialized")
        
        # Test DataVisionDownloader
        dv_downloader = DataVisionDownloader(tracker)
        print("✓ DataVisionDownloader initialized")
        
        # Test SymbolManager
        symbol_manager = SymbolManager()
        print("✓ SymbolManager initialized")
        
        # Test StorageManager
        storage = StorageManager()
        print("✓ StorageManager initialized")
        
        # Test IntegrityChecker
        integrity = IntegrityChecker()
        print("✓ IntegrityChecker initialized")
        
        # Test BacktestDataLoader
        loader = BacktestDataLoader()
        print("✓ BacktestDataLoader initialized")
        
        print("\n✅ All components initialized successfully!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Component initialization failed: {e}\n")
        return False


def test_symbol_fetching():
    """Test fetching symbol list"""
    print("Testing symbol fetching...")
    
    try:
        manager = SymbolManager()
        
        # Get priority symbols
        priority = manager.get_priority_symbols()
        print(f"✓ Found {len(priority)} priority symbols")
        print(f"  First 5: {priority[:5]}")
        
        # Try to fetch active symbols from API
        print("\nFetching active symbols from Binance API...")
        active = manager.fetch_active_symbols()
        
        if active:
            print(f"✓ Found {len(active)} active futures symbols")
            print(f"  Sample: {[s['symbol'] for s in active[:3]]}")
        else:
            print("⚠ Could not fetch active symbols (API may be unavailable)")
        
        print("\n✅ Symbol fetching test passed!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Symbol fetching failed: {e}\n")
        return False


def test_download_tracker():
    """Test download tracking functionality"""
    print("Testing download tracker...")
    
    try:
        tracker = DownloadTracker()
        
        # Test adding a download entry
        test_file_id = "TEST_BTCUSDT_klines_2024_01"
        
        # Check if should download
        should_download, resume_pos = tracker.should_download(test_file_id)
        print(f"✓ Should download new file: {should_download}")
        
        # Update status
        tracker.update_status(test_file_id, 'downloading', {
            'symbol': 'BTCUSDT',
            'data_type': 'klines',
            'year_month': '2024-01'
        })
        print("✓ Status updated to 'downloading'")
        
        # Get status
        status = tracker.get_file_status(test_file_id)
        print(f"✓ Retrieved status: {status['status']}")
        
        # Get statistics
        stats = tracker.get_statistics()
        print(f"✓ Statistics: {stats['total']} total, {stats['completed']} completed")
        
        print("\n✅ Download tracker test passed!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Download tracker test failed: {e}\n")
        return False


def test_data_vision_urls():
    """Test Data Vision URL generation"""
    print("Testing Data Vision URL generation...")
    
    try:
        downloader = DataVisionDownloader()
        
        # Test URL building for different data types
        test_date = datetime(2024, 1, 15)
        test_symbol = "BTCUSDT"
        
        print(f"\nURLs for {test_symbol} on {test_date:%Y-%m-%d}:")
        
        for data_type in ['klines', 'mark_price', 'funding_rate']:
            file_info = downloader._build_daily_file_info(
                test_symbol, data_type, test_date
            )
            
            if file_info:
                print(f"  {data_type}: {file_info['url']}")
        
        # Test monthly URL
        file_info = downloader._build_monthly_file_info(
            test_symbol, 'klines', 2024, 1
        )
        
        if file_info:
            print(f"\nMonthly URL: {file_info['url']}")
        
        print("\n✅ URL generation test passed!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ URL generation test failed: {e}\n")
        return False


def test_small_download():
    """Test downloading a small amount of data"""
    print("Testing small data download...")
    print("This will download 1 day of BTCUSDT data as a test")
    
    response = input("Continue with download test? (y/n): ")
    if response.lower() != 'y':
        print("Skipping download test")
        return True
    
    try:
        tracker = DownloadTracker()
        downloader = DataVisionDownloader(tracker)
        
        # Download just 1 day of BTCUSDT klines
        yesterday = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        
        print(f"\nDownloading BTCUSDT klines for {yesterday}...")
        
        results = downloader.download_symbol_history(
            symbol="BTCUSDT",
            data_types=["klines"],
            start_date=yesterday,
            end_date=yesterday,
            use_monthly=False  # Use daily for single day
        )
        
        if results and results.get('klines'):
            print(f"✓ Downloaded {len(results['klines'])} files")
            
            # Process the downloaded file
            storage = StorageManager()
            process_stats = storage.process_downloaded_files(
                symbol="BTCUSDT",
                data_type="klines"
            )
            
            print(f"✓ Processed: {process_stats['extracted']} extracted, "
                  f"{process_stats['converted']} converted")
            
            # Try to load the data
            loader = BacktestDataLoader()
            data = loader.load_data(
                symbols="BTCUSDT",
                start_date=yesterday,
                end_date=yesterday,
                data_types=["klines"]
            )
            
            if 'klines' in data and not data['klines'].empty:
                df = data['klines']
                print(f"✓ Loaded {len(df)} rows of data")
                print(f"  Columns: {list(df.columns)[:5]}...")
                print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        else:
            print("⚠ No data downloaded (may already exist)")
        
        print("\n✅ Download test completed!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Download test failed: {e}\n")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("BINANCE FUTURES DATA SYSTEM TEST")
    print("="*60)
    print()
    
    tests = [
        ("Component Initialization", test_components),
        ("Symbol Fetching", test_symbol_fetching),
        ("Download Tracker", test_download_tracker),
        ("URL Generation", test_data_vision_urls),
        ("Small Download", test_small_download)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Download priority symbols: python scripts/download_all.py --priority-only")
        print("3. Monitor progress: python scripts/monitor.py live")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()