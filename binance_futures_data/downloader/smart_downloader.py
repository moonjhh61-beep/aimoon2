"""
Smart Downloader with resume capability
Handles interrupted downloads and can resume from exact byte position
"""

import os
import requests
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import time
from tqdm import tqdm

from .download_tracker import DownloadTracker


class SmartDownloader:
    """
    Intelligent downloader with resume capability
    - Downloads to temp file first
    - Supports byte-range resume
    - Automatic retry with exponential backoff
    - Checksum verification
    """
    
    def __init__(self, tracker: DownloadTracker = None, max_retries: int = 3):
        """
        Initialize smart downloader
        
        Args:
            tracker: Download tracker instance
            max_retries: Maximum number of retry attempts
        """
        self.tracker = tracker or DownloadTracker()
        self.max_retries = max_retries
        self.temp_dir = Path("data/temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_with_resume(self, url: str, target_path: Path, 
                            file_id: str, metadata: dict = None) -> bool:
        """
        Download file with resume capability
        
        Args:
            url: Remote file URL
            target_path: Final destination path
            file_id: Unique identifier for tracking
            metadata: Additional metadata for tracking
            
        Returns:
            True if download successful, False otherwise
        """
        # Create target directory if not exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get file info from remote
        try:
            remote_size, remote_checksum = self._get_remote_info(url)
        except Exception as e:
            print(f"Failed to get remote info for {url}: {e}")
            return False
        
        # Check if download needed
        should_download, resume_position = self.tracker.should_download(
            file_id, remote_size, remote_checksum
        )
        
        if not should_download:
            print(f"✓ {file_id} already up to date")
            return True
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'remote_url': url,
            'expected_size': remote_size,
            'expected_checksum': remote_checksum
        })
        
        # Update status to downloading
        self.tracker.update_status(file_id, 'downloading', metadata)
        
        # Temporary file path
        temp_file = self.temp_dir / f"{file_id}.downloading"
        
        # Check existing partial download
        if temp_file.exists() and resume_position > 0:
            existing_size = temp_file.stat().st_size
            if existing_size < resume_position:
                resume_position = existing_size
            print(f"Resuming {file_id} from byte {resume_position}/{remote_size}")
        else:
            resume_position = 0
        
        # Download with retries
        for attempt in range(self.max_retries):
            try:
                success = self._download_file(
                    url, temp_file, resume_position, 
                    remote_size, file_id
                )
                
                if success:
                    # Verify checksum if available
                    if remote_checksum:
                        actual_checksum = self._calculate_checksum(temp_file)
                        if actual_checksum != remote_checksum:
                            raise ValueError(f"Checksum mismatch: {actual_checksum} != {remote_checksum}")
                    
                    # Move to final destination
                    if target_path.exists():
                        # Backup existing file
                        backup_path = target_path.with_suffix('.backup')
                        target_path.rename(backup_path)
                    
                    temp_file.rename(target_path)
                    
                    # Update tracker
                    self.tracker.update_status(file_id, 'completed', {
                        'local_path': str(target_path),
                        'actual_size': remote_size,
                        'actual_checksum': remote_checksum
                    })
                    
                    print(f"✓ Completed: {file_id}")
                    return True
                    
            except Exception as e:
                print(f"Attempt {attempt + 1}/{self.max_retries} failed for {file_id}: {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    
                    # Update resume position for next attempt
                    if temp_file.exists():
                        resume_position = temp_file.stat().st_size
                else:
                    # Final failure
                    self.tracker.update_status(file_id, 'failed', {
                        'error_message': str(e),
                        'retry_count': self.tracker.get_retry_count(file_id) + 1
                    })
                    return False
        
        return False
    
    def _get_remote_info(self, url: str) -> Tuple[int, Optional[str]]:
        """
        Get remote file size and checksum
        
        Returns:
            Tuple of (file_size, checksum)
        """
        # Try HEAD request first
        response = self.session.head(url, timeout=10)
        file_size = int(response.headers.get('content-length', 0))
        
        # Try to get checksum file
        checksum = None
        checksum_url = f"{url}.CHECKSUM"
        
        try:
            checksum_response = self.session.get(checksum_url, timeout=5)
            if checksum_response.status_code == 200:
                # Parse checksum (format: "hash filename")
                checksum_text = checksum_response.text.strip()
                if checksum_text:
                    checksum = checksum_text.split()[0]
        except:
            pass  # Checksum is optional
        
        return file_size, checksum
    
    def _download_file(self, url: str, temp_file: Path, 
                      resume_position: int, total_size: int, 
                      file_id: str) -> bool:
        """
        Actually download the file with progress bar
        
        Returns:
            True if successful, False otherwise
        """
        headers = {}
        mode = 'wb'
        
        if resume_position > 0:
            headers['Range'] = f'bytes={resume_position}-'
            mode = 'ab'
        
        response = self.session.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get content length
        content_length = int(response.headers.get('content-length', 0))
        
        # Progress bar
        progress_bar = tqdm(
            total=total_size,
            initial=resume_position,
            unit='B',
            unit_scale=True,
            desc=f"Downloading {file_id}"
        )
        
        # Download in chunks
        chunk_size = 8192
        downloaded = resume_position
        
        with open(temp_file, mode) as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress_bar.update(len(chunk))
                    
                    # Update tracker periodically (every 10MB)
                    if downloaded % (10 * 1024 * 1024) == 0:
                        self.tracker.update_status(file_id, 'downloading', {
                            'last_byte_position': downloaded
                        })
        
        progress_bar.close()
        
        # Verify downloaded size
        actual_size = temp_file.stat().st_size
        if actual_size != total_size:
            raise ValueError(f"Size mismatch: downloaded {actual_size} != expected {total_size}")
        
        return True
    
    def _calculate_checksum(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """
        Calculate file checksum
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (sha256, md5, etc)
            
        Returns:
            Hex digest of file hash
        """
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def batch_download(self, download_list: list, parallel: int = 3) -> dict:
        """
        Download multiple files with parallelization
        
        Args:
            download_list: List of dicts with url, target_path, file_id
            parallel: Number of parallel downloads
            
        Returns:
            Dictionary with download results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {
            'success': [],
            'failed': []
        }
        
        def download_wrapper(item):
            """Wrapper for thread pool"""
            success = self.download_with_resume(
                url=item['url'],
                target_path=Path(item['target_path']),
                file_id=item['file_id'],
                metadata=item.get('metadata', {})
            )
            return item['file_id'], success
        
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = []
            
            for item in download_list:
                future = executor.submit(download_wrapper, item)
                futures.append(future)
            
            for future in as_completed(futures):
                file_id, success = future.result()
                
                if success:
                    results['success'].append(file_id)
                else:
                    results['failed'].append(file_id)
        
        return results
    
    def cleanup_temp_files(self, older_than_hours: int = 24):
        """
        Clean up old temporary files
        
        Args:
            older_than_hours: Remove files older than this many hours
        """
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (older_than_hours * 3600)
        
        for temp_file in self.temp_dir.glob("*.downloading"):
            if temp_file.stat().st_mtime < cutoff_time:
                print(f"Removing old temp file: {temp_file}")
                temp_file.unlink()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session"""
        self.session.close()