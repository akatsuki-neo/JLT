"""
Fast Parquet Dataset - batch reading optimization for training.

Parquet random access is slow, so we:
1. Read files in batches (not row-by-row)
2. Cache decoded samples in memory
3. Use memory-efficient approach

For best performance, convert to WebDataset or pre-decode to tensors.
"""
import os
import io
import bisect
from glob import glob
from typing import Callable, Optional, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import pyarrow.parquet as pq
import numpy as np
from torchvision import transforms

class ParquetImageNetDataset(Dataset):
    """Parquet Dataset with batch reading optimization.
    
    Reads data in chunks to amortize the parquet overhead.
    Good balance between memory and speed.
    
    Args:
        data_dir: Path to directory containing parquet files
        split: 'train', 'validation', or 'test'
        transform: Optional transform
        chunk_size: Number of rows to read at once (larger = faster but more memory)
        image_key: Column name for image
        label_key: Column name for label
    """
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = 'train', 
        transform: Optional[Callable] = None,
        chunk_size: int = 1000,
        image_key: str = 'image',
        label_key: str = 'label',
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.chunk_size = chunk_size
        self.image_key = image_key
        self.label_key = label_key
        
        # Find all parquet files
        pattern = os.path.join(data_dir, f"{split}*.parquet")
        self.parquet_files = sorted(glob(pattern))
        
        if len(self.parquet_files) == 0:
            raise ValueError(f"No parquet files found for split '{split}' in {data_dir}")
        
        # Build index: list of (file_idx, row_in_file, global_idx)
        print(f"Indexing {len(self.parquet_files)} parquet files...")
        self._index = []  # (file_idx, row_in_file)
        self._file_offsets = [0]  # Cumulative row counts
        
        for file_idx, fpath in enumerate(self.parquet_files):
            pf = pq.ParquetFile(fpath)
            num_rows = pf.metadata.num_rows
            
            for row_idx in range(num_rows):
                self._index.append((file_idx, row_idx))
            
            self._file_offsets.append(self._file_offsets[-1] + num_rows)
        
        self._len = len(self._index)
        print(f"Indexed {self._len} images")
        
        # Chunk cache
        self._cache_file_idx = -1
        self._cache_chunk_start = -1
        self._cache_data = []  # List of (image_bytes, label) tuples
    
    def __len__(self):
        return self._len
    
    def _find_chunk(self, idx: int) -> Tuple[int, int, int]:
        """Find which file and chunk contains idx.
        Returns: (file_idx, chunk_start, local_idx_in_chunk)
        """
        file_idx, row_in_file = self._index[idx]
        chunk_start = (row_in_file // self.chunk_size) * self.chunk_size
        local_idx = row_in_file - chunk_start
        return file_idx, chunk_start, local_idx
    
    def _load_chunk(self, file_idx: int, chunk_start: int):
        """Load a chunk of data from file."""
        fpath = self.parquet_files[file_idx]
        pf = pq.ParquetFile(fpath)
        
        # Calculate chunk bounds
        total_rows = pf.metadata.num_rows
        chunk_end = min(chunk_start + self.chunk_size, total_rows)
        num_rows = chunk_end - chunk_start
        
        # Read chunk
        table = pq.read_table(
            fpath, 
            columns=[self.image_key, self.label_key],
            use_threads=True,
        )
        
        # Slice to chunk
        chunk_table = table.slice(chunk_start, num_rows)
        
        # Store as list for fast indexing
        images = chunk_table[self.image_key].to_pylist()
        labels = chunk_table[self.label_key].to_pylist()
        
        self._cache_data = list(zip(images, labels))
        self._cache_file_idx = file_idx
        self._cache_chunk_start = chunk_start
    
    def __getitem__(self, idx):
        # Find which chunk this idx belongs to
        file_idx, chunk_start, local_idx = self._find_chunk(idx)
        
        # Load chunk if not cached
        if (self._cache_file_idx != file_idx or 
            self._cache_chunk_start != chunk_start):
            self._load_chunk(file_idx, chunk_start)
        
        # Get data from cache
        img_data, label = self._cache_data[local_idx]
        
        # Decode image
        if isinstance(img_data, dict):
            img_bytes = img_data['bytes']
        else:
            img_bytes = img_data
        
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

class HuggingFaceImageNetDataset(Dataset):
    """Dataset for loading ImageNet from local Parquet files using HuggingFace datasets.
    
    Uses memory mapping for fast random access.
    """
    
    def __init__(
        self, 
        data_dir: str,
        split="train", 
        transform: Optional[Callable] = None,
        cache_dir=None,
    ):
        """
        Args:
            data_dir: Path to directory containing parquet files
            split: Dataset split ("train" or "validation")
            transform: Transform to apply to images (use JLT's transform)
            cache_dir: Directory to cache the dataset
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install `datasets` package: pip install datasets")
        
        import os
        from glob import glob
        
        self.transform = transform
        
        # Find all parquet files for this split
        pattern = os.path.join(data_dir, f"{split}*.parquet")
        parquet_files = sorted(glob(pattern))
        
        if len(parquet_files) == 0:
            raise ValueError(f"No parquet files found for split '{split}' in {data_dir}")
        
        print(f"Loading {len(parquet_files)} parquet files with HuggingFace datasets...")
        
        # Load dataset from local parquet files
        self.dataset = load_dataset(
            'parquet',
            data_files={split: parquet_files},
            split=split,
            cache_dir=cache_dir,
        )
        
        print(f"Loaded {len(self.dataset)} images (memory mapped)")
    
    def __len__(self):
        # if self.streaming:
            # Streaming mode doesn't support len
            # raise NotImplementedError("Length not available in streaming mode")
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Get image and label from HuggingFace format
        image = sample["image"]
        label = sample["label"]
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Apply transforms
        image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


# Alias for compatibility
FastParquetImageNetDataset = ParquetImageNetDataset
