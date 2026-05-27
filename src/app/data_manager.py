# src/app/data_manager.py
import os
import numpy as np
import threading
import tifffile as tiff
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

class DataManager:
    def __init__(self):
        self.storage_path = None
        self._lock = threading.Lock()
        self._writers = {}
        self._counts = {}
        
        self._disk_pool = None
        self._cam_pool = None

    def configure(self, storage_path: str):
        self.storage_path = storage_path
        
        self._disk_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="DiskIO")
        self._cam_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="CamIO")
        
        logger.info(f"DataManager globally configured at: {storage_path}")

    def save(self, stream_name: str, data: np.ndarray, chunk_size: int = None, is_float: bool = False):
        if self._disk_pool is None:
            raise RuntimeError("DataManager must be configured before calling save().")
        
        self._disk_pool.submit(self._execute_save, stream_name, data, chunk_size, is_float)

    def submit_task(self, func, *args, **kwargs):
        if self._cam_pool is None:
            raise RuntimeError("DataManager must be configured before submitting tasks.")
        
        self._cam_pool.submit(func, *args, **kwargs)

    def _execute_save(self, stream_name: str, data: np.ndarray, chunk_size: int = None, is_float: bool = False):
        if chunk_size is not None:
            folder_path = os.path.join(self.storage_path, stream_name)
            
            with self._lock:
                count = self._counts.get(stream_name, 0)
                writer = self._writers.get(stream_name)

                if count % chunk_size == 0:
                    if writer:
                        writer.close()
                    os.makedirs(folder_path, exist_ok=True)
                    chunk_idx = count // chunk_size
                    file_path = os.path.join(folder_path, f"{stream_name}_chunk_{chunk_idx:03d}.tif")
                    writer = tiff.TiffWriter(file_path, bigtiff=False)
                    self._writers[stream_name] = writer

                self._counts[stream_name] = count + 1

                writer.write(data, contiguous=True)
            
        else:
            os.makedirs(self.storage_path, exist_ok=True)
            file_path = os.path.join(self.storage_path, f"{stream_name}.tif")
            tiff.imwrite(file_path, data)

    def close(self):
        logger.info("Flushing remaining image writes and shutting down DataManager thread pools...")
        
        if self._cam_pool:
            self._cam_pool.shutdown(wait=True)
            
        if self._disk_pool:
            self._disk_pool.shutdown(wait=True)
        
        with self._lock:
            for writer in self._writers.values():
                if writer:
                    writer.close()
            self._writers.clear()
        logger.success("DataManager cleanly closed.")

data_manager = DataManager()