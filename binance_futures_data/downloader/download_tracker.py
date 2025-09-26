"""
Download Tracker with SQLite for persistent state management
Tracks download progress and enables perfect resume capability
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import hashlib


class DownloadTracker:
    """
    Tracks download state in SQLite database
    Enables resume from exact point of interruption
    """
    
    def __init__(self, db_path: str = "/tmp/crypto_data_metadata/download_status.db"):
        """Initialize download tracker with SQLite database"""
        import os
        # Use absolute path to ensure consistency across processes
        if not os.path.isabs(db_path):
            db_path = os.path.abspath(db_path)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Create database tables if not exists"""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Main download status table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS download_status (
                    file_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    year_month TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    local_path TEXT,
                    remote_url TEXT,
                    expected_size INTEGER,
                    actual_size INTEGER,
                    expected_checksum TEXT,
                    actual_checksum TEXT,
                    row_count INTEGER,
                    download_started TIMESTAMP,
                    download_completed TIMESTAMP,
                    last_byte_position INTEGER DEFAULT 0,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Symbol completion tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS symbol_status (
                    symbol TEXT PRIMARY KEY,
                    listing_date TEXT,
                    total_files INTEGER,
                    completed_files INTEGER,
                    failed_files INTEGER,
                    status TEXT DEFAULT 'pending',
                    last_update TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Download statistics
            conn.execute('''
                CREATE TABLE IF NOT EXISTS download_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_files INTEGER,
                    completed_files INTEGER,
                    failed_files INTEGER,
                    total_bytes_downloaded INTEGER,
                    session_id TEXT
                )
            ''')
            
            # Create indexes for better performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON download_status(symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON download_status(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_data_type ON download_status(data_type)')
            
            conn.commit()
    
    def should_download(self, file_id: str, remote_size: int = None, 
                       remote_checksum: str = None) -> Tuple[bool, int]:
        """
        Check if file should be downloaded
        Returns: (should_download, resume_position)
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                'SELECT status, actual_size, expected_size, actual_checksum, '
                'expected_checksum, last_byte_position, local_path '
                'FROM download_status WHERE file_id = ?',
                (file_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                # New file, need to download
                return True, 0
            
            status, actual_size, expected_size, actual_checksum, \
            expected_checksum, last_byte_position, local_path = row
            
            # Check if already completed
            if status == 'completed':
                # Verify checksum if provided
                if remote_checksum and actual_checksum != remote_checksum:
                    print(f"Checksum mismatch for {file_id}, re-downloading")
                    return True, 0
                
                # Verify size if provided
                if remote_size and actual_size < remote_size:
                    print(f"Size mismatch for {file_id} ({actual_size} < {remote_size})")
                    return True, 0
                
                # File is complete and valid
                return False, 0
            
            # Check if download was interrupted
            if status == 'downloading':
                # Check if partial file exists
                if local_path and Path(local_path).exists():
                    file_size = Path(local_path).stat().st_size
                    return True, file_size  # Resume from current size
                return True, 0
            
            # Failed or pending status
            return True, 0
    
    def update_status(self, file_id: str, status: str, metadata: Dict = None):
        """Update download status for a file"""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Check if record exists
            cursor = conn.execute(
                'SELECT file_id FROM download_status WHERE file_id = ?',
                (file_id,)
            )
            exists = cursor.fetchone() is not None
            
            if not exists and metadata:
                # Insert new record
                conn.execute('''
                    INSERT INTO download_status (
                        file_id, symbol, data_type, year_month, status,
                        remote_url, expected_size, expected_checksum
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    file_id,
                    metadata.get('symbol', ''),
                    metadata.get('data_type', ''),
                    metadata.get('year_month', ''),
                    status,
                    metadata.get('remote_url', ''),
                    metadata.get('expected_size', 0),
                    metadata.get('expected_checksum', '')
                ))
            
            # Update status and metadata
            update_fields = ['status = ?', 'updated_at = CURRENT_TIMESTAMP']
            update_values = [status]
            
            if metadata:
                if 'local_path' in metadata:
                    update_fields.append('local_path = ?')
                    update_values.append(metadata['local_path'])
                
                if 'actual_size' in metadata:
                    update_fields.append('actual_size = ?')
                    update_values.append(metadata['actual_size'])
                
                if 'actual_checksum' in metadata:
                    update_fields.append('actual_checksum = ?')
                    update_values.append(metadata['actual_checksum'])
                
                if 'last_byte_position' in metadata:
                    update_fields.append('last_byte_position = ?')
                    update_values.append(metadata['last_byte_position'])
                
                if 'error_message' in metadata:
                    update_fields.append('error_message = ?')
                    update_values.append(metadata['error_message'])
                
                if 'retry_count' in metadata:
                    update_fields.append('retry_count = ?')
                    update_values.append(metadata['retry_count'])
                
                if status == 'downloading' and 'download_started' not in metadata:
                    update_fields.append('download_started = CURRENT_TIMESTAMP')
                
                if status == 'completed':
                    update_fields.append('download_completed = CURRENT_TIMESTAMP')
            
            update_values.append(file_id)
            
            query = f"UPDATE download_status SET {', '.join(update_fields)} WHERE file_id = ?"
            conn.execute(query, update_values)
            conn.commit()
    
    def get_file_status(self, file_id: str) -> Optional[Dict]:
        """Get current status of a file"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT * FROM download_status WHERE file_id = ?',
                (file_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def get_pending_downloads(self, symbol: str = None) -> List[Dict]:
        """Get list of pending downloads"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            if symbol:
                cursor = conn.execute(
                    'SELECT * FROM download_status WHERE status IN (?, ?, ?) AND symbol = ? '
                    'ORDER BY year_month DESC',
                    ('pending', 'downloading', 'failed', symbol)
                )
            else:
                cursor = conn.execute(
                    'SELECT * FROM download_status WHERE status IN (?, ?, ?) '
                    'ORDER BY symbol, year_month DESC',
                    ('pending', 'downloading', 'failed')
                )
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_retry_count(self, file_id: str) -> int:
        """Get retry count for a file"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                'SELECT retry_count FROM download_status WHERE file_id = ?',
                (file_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else 0
    
    def increment_retry_count(self, file_id: str):
        """Increment retry count for a file"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                'UPDATE download_status SET retry_count = retry_count + 1 WHERE file_id = ?',
                (file_id,)
            )
            conn.commit()
    
    def mark_symbol_complete(self, symbol: str):
        """Mark a symbol as completely downloaded"""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Count files for this symbol
            cursor = conn.execute(
                'SELECT COUNT(*) as total, '
                'SUM(CASE WHEN status = "completed" THEN 1 ELSE 0 END) as completed, '
                'SUM(CASE WHEN status = "failed" THEN 1 ELSE 0 END) as failed '
                'FROM download_status WHERE symbol = ?',
                (symbol,)
            )
            total, completed, failed = cursor.fetchone()
            
            # Update or insert symbol status
            conn.execute('''
                INSERT OR REPLACE INTO symbol_status 
                (symbol, total_files, completed_files, failed_files, status, last_update)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (symbol, total, completed, failed, 
                  'completed' if completed == total else 'partial'))
            conn.commit()
    
    def is_symbol_complete(self, symbol: str) -> bool:
        """Check if all files for a symbol are downloaded"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                'SELECT status FROM symbol_status WHERE symbol = ?',
                (symbol,)
            )
            row = cursor.fetchone()
            return row and row[0] == 'completed'
    
    def get_statistics(self) -> Dict:
        """Get overall download statistics"""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Overall file statistics
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = "completed" THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = "failed" THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status = "downloading" THEN 1 ELSE 0 END) as downloading,
                    SUM(CASE WHEN status = "pending" THEN 1 ELSE 0 END) as pending,
                    SUM(actual_size) as total_bytes
                FROM download_status
            ''')
            stats = dict(zip([d[0] for d in cursor.description], cursor.fetchone()))
            
            # Symbol statistics
            cursor = conn.execute(
                'SELECT COUNT(*) as total_symbols, '
                'SUM(CASE WHEN status = "completed" THEN 1 ELSE 0 END) as completed_symbols '
                'FROM symbol_status'
            )
            symbol_stats = cursor.fetchone()
            stats['total_symbols'] = symbol_stats[0] or 0
            stats['completed_symbols'] = symbol_stats[1] or 0
            
            # Calculate percentage
            if stats['total'] > 0:
                stats['progress_percent'] = (stats['completed'] / stats['total']) * 100
            else:
                stats['progress_percent'] = 0
            
            # Convert bytes to GB
            stats['total_gb'] = (stats['total_bytes'] or 0) / (1024**3)
            
            return stats
    
    def get_failed_files(self) -> List[Dict]:
        """Get list of failed downloads"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT file_id, symbol, data_type, year_month, error_message, retry_count '
                'FROM download_status WHERE status = "failed" '
                'ORDER BY symbol, year_month'
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def reset_failed_downloads(self):
        """Reset all failed downloads to pending"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                'UPDATE download_status SET status = "pending", retry_count = 0 '
                'WHERE status = "failed"'
            )
            conn.commit()
    
    def cleanup_incomplete_downloads(self):
        """Clean up interrupted downloads"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                'UPDATE download_status SET status = "pending" '
                'WHERE status = "downloading"'
            )
            conn.commit()