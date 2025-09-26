# Binance Futures Data Management System

A robust, production-ready system for downloading and managing Binance futures historical data with perfect resume capability and data integrity.

## Features

- ✅ **Complete Historical Data**: Download from 2019-09-13 (Binance Futures launch)
- ✅ **5 Data Types**: Klines, Mark Price, Index Price, Premium Index, Funding Rate
- ✅ **500+ Symbols**: All futures symbols including delisted
- ✅ **Perfect Resume**: Resume downloads from exact byte position
- ✅ **Parallel Downloads**: Multiple symbols simultaneously
- ✅ **Data Integrity**: Automatic validation and checksums
- ✅ **Dual Storage**: CSV (human-readable) + Parquet (high-performance)
- ✅ **Smart Updates**: Only download new data
- ✅ **Rich Monitoring**: Beautiful progress tracking

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Download Priority Symbols

```bash
# Download top 35 priority symbols
python scripts/download_all.py --priority-only
```

### Download All Data

```bash
# Download all symbols (500+) - can be safely interrupted and resumed
python scripts/download_all.py

# Resume from interruption
python scripts/download_all.py  # Automatically resumes

# Start fresh (ignore previous progress)
python scripts/download_all.py --no-resume
```

### Monitor Progress

```bash
# Live monitoring dashboard
python scripts/monitor.py live

# Check data integrity
python scripts/monitor.py integrity

# Show summary
python scripts/monitor.py summary
```

### Daily Updates

```bash
# Update with yesterday's data
python scripts/daily_update.py

# Update last 7 days
python scripts/daily_update.py --days-back 7

# Fill gaps in data
python scripts/daily_update.py --fill-gaps
```

## Data Structure

```
data/
├── raw/                    # Original ZIP files from Binance
│   ├── klines/
│   ├── mark_price/
│   ├── index_price/
│   ├── premium_index/
│   └── funding_rate/
├── extracted/              # Unzipped CSV files
├── processed/              # Optimized Parquet files
│   └── [same structure]
└── metadata/
    ├── download_status.db  # SQLite tracking database
    └── symbols_info.json   # Symbol metadata
```

## Usage in Backtesting

```python
from loader import BacktestDataLoader

# Initialize loader
loader = BacktestDataLoader()

# Load specific symbols
data = loader.load_data(
    symbols=['BTCUSDT', 'ETHUSDT'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    data_types=['klines', 'funding_rate']
)

# Access data
klines_df = data['klines']
funding_df = data['funding_rate']

# Load entire universe
universe_data = loader.load_universe(
    start_date='2024-01-01',
    end_date='2024-12-31',
    min_volume=1000000  # Filter by volume
)

# Get aligned data (all types merged)
aligned_df = loader.load_aligned_data(
    symbols=['BTCUSDT'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)
```

## Data Types

### 1. Klines (1-hour OHLCV)
- `timestamp`: Time
- `open`, `high`, `low`, `close`: Prices
- `volume`: Base asset volume
- `quote_volume`: USDT volume
- `taker_buy_base`, `taker_buy_quote`: Buy-side volume

### 2. Mark Price (1-hour)
- Used for PnL calculations and liquidations
- Same structure as klines

### 3. Index Price (1-hour)
- Underlying spot price from major exchanges
- Same structure as klines

### 4. Premium Index (1-hour)
- Futures-spot premium percentage
- `close` column contains the premium value

### 5. Funding Rate (8-hour)
- `timestamp`: Funding time
- `funding_rate`: Rate value
- Applied at 00:00, 08:00, 16:00 UTC

## Performance

- **Download Speed**: ~10-20 symbols per minute
- **Storage**: ~150MB per symbol (5 years, all data types)
- **Total Size**: ~75GB for 500 symbols (Parquet compressed)
- **Query Speed**: <100ms for loading 1 year of data

## Resume & Recovery

The system automatically handles:
- **Network interruptions**: Resume from exact byte
- **System crashes**: SQLite tracks all progress
- **Partial downloads**: Complete incomplete files
- **Data corruption**: Checksum validation

## Configuration

Edit `config/settings.yaml` for:
- Parallel download threads
- Cache size
- Storage formats
- Data quality thresholds

Edit `config/symbols.yaml` for:
- Priority symbols
- Symbol categories
- Volume filters

## Architecture

1. **DownloadTracker**: SQLite-based progress tracking
2. **SmartDownloader**: Resume-capable downloader with checksums
3. **DataVisionDownloader**: Binance Data Vision interface
4. **StorageManager**: CSV/Parquet conversion and organization
5. **IntegrityChecker**: Data validation and quality checks
6. **BacktestDataLoader**: High-performance data loading

## Troubleshooting

### Download Failures
```bash
# Reset failed downloads
python scripts/download_all.py --no-resume

# Check specific symbol
python scripts/monitor.py integrity --symbols BTCUSDT
```

### Data Gaps
```bash
# Find and fill gaps
python scripts/daily_update.py --fill-gaps --max-gap-days 30
```

### Storage Issues
```bash
# Process already downloaded files
python scripts/download_all.py --process-only

# Clean up redundant files
python scripts/cleanup.py
```

## License

MIT

## Support

For issues or questions, please open an issue on GitHub.