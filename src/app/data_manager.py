import os
import numpy as np
import tifffile as tiff
from app.config import Config

class DataManager:
    def __init__(self, storage_path, bf_shape=None, fl_shape=None):
        self.storage_path = storage_path

        # Create folders
        self.bf_path = os.path.join(self.storage_path, "brightfield")
        self.fl_path = os.path.join(self.storage_path, "fluorescence")
        os.makedirs(self.bf_path, exist_ok=True)
        os.makedirs(self.fl_path, exist_ok=True)
        
        # Frame counters
        self.bf_count = 0
        self.fl_count = 0
        
        # Active file writing streams
        self.bf_writer = None
        self.fl_writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.bf_writer:
            self.bf_writer.close()
        if self.fl_writer:
            self.fl_writer.close()

    def write_brightfield_frame(self, frame, timestamp=None):
        if self.bf_count % Config.ExperimentConfig.BRIGHTFIELD_CHUNK_SIZE == 0:
            if self.bf_writer:
                self.bf_writer.close()
            chunk_idx = self.bf_count // Config.ExperimentConfig.BRIGHTFIELD_CHUNK_SIZE
            
            file_path = os.path.join(self.bf_path, f"bf_chunk_{chunk_idx:03d}.tif")
            self.bf_writer = tiff.TiffWriter(file_path, bigtiff=True)
            
        self.bf_writer.write(frame.astype(np.uint16))
        self.bf_count += 1

    def write_fluorescence_frame(self, frame, timestamp=None):
        if self.fl_count % Config.ExperimentConfig.FLUORESCENCE_CHUNK_SIZE == 0:
            if self.fl_writer:
                self.fl_writer.close()
            chunk_idx = self.fl_count // Config.ExperimentConfig.FLUORESCENCE_CHUNK_SIZE
            
            file_path = os.path.join(self.fl_path, f"fl_chunk_{chunk_idx:03d}.tif")
            self.fl_writer = tiff.TiffWriter(file_path, bigtiff=True)
            
        self.fl_writer.write(frame.astype(np.uint16))
        self.fl_count += 1