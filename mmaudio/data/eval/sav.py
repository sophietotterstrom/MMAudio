import json
import logging
import os
from pathlib import Path
from typing import Union, Optional

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder

from mmaudio.utils.dist_utils import local_rank

from mmaudio.data.eval.video_dataset import VideoDataset

log = logging.getLogger()


_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0



class SAVDataset(VideoDataset):

    def __init__(
        self,
        video_root: Union[str, Path],
        csv_path: Union[str, Path],
        *,
        duration_sec: float = 8.0,
    ):
        super().__init__(video_root, duration_sec=duration_sec)
        
        self.csv_path = Path(csv_path)

        # Build a mapping of video_id to full path (supporting recursive structure)
        self.video_paths = {}
        self._build_video_paths_recursive(self.video_root)

        # Read SA-V CSV format with comma separator
        df = pd.read_csv(csv_path).to_dict(orient='records')

        videos_not_found = []
        
        for row in df: 
            video_id = str(row['id'])
            
            if video_id not in self.video_paths:
                videos_not_found.append(video_id)
                continue

            # Get caption from CSV
            caption = row['caption']
            
            # Skip entries with empty captions
            if pd.isna(caption) or caption.strip() == '':
                continue
                
            self.captions[video_id] = caption

        if local_rank == 0:
            log.info(f'{len(self.video_paths)} total videos found recursively in {video_root}')
            log.info(f'{len(self.captions)} useable videos found')
            if videos_not_found:
                log.info(f'{len(videos_not_found)} videos in CSV but not found in video directory structure')

        self.videos = sorted(list(self.captions.keys()))

    def _build_video_paths_recursive(self, root_path: Path):
        """Recursively build a mapping of video_id to full path."""
        for item in root_path.rglob('*.mp4'):
            # Extract video_id from filename (remove .mp4 extension)
            video_id = item.stem
            self.video_paths[video_id] = item

    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        """Override parent sample method to use correct (recursive) paths."""
        video_id = self.videos[idx]
        caption = self.captions[video_id]

        # Use the actual path from our mapping instead of constructing it
        video_path = self.video_paths[video_id]
        
        reader = StreamingMediaDecoder(video_path)
        reader.add_basic_video_stream(
            frames_per_chunk=int(_CLIP_FPS * self.duration_sec),
            frame_rate=int(_CLIP_FPS),
            format='rgb24',
        )
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=int(_SYNC_FPS),
            format='rgb24',
        )

        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        clip_chunk = data_chunk[0]
        sync_chunk = data_chunk[1] if len(data_chunk) > 1 else None
        if clip_chunk is None:
            raise RuntimeError(f'CLIP video returned None {video_id}')
        if clip_chunk.shape[0] < self.clip_expected_length:
            raise RuntimeError(
                f'CLIP video too short {video_id}, expected {self.clip_expected_length}, got {clip_chunk.shape[0]}'
            )

        if sync_chunk is None:
            raise RuntimeError(f'Sync video returned None {video_id}')
        if sync_chunk.shape[0] < self.sync_expected_length:
            raise RuntimeError(
                f'Sync video too short {video_id}, expected {self.sync_expected_length}, got {sync_chunk.shape[0]}'
            )

        # truncate the video
        clip_chunk = clip_chunk[:self.clip_expected_length]
        if clip_chunk.shape[0] != self.clip_expected_length:
            raise RuntimeError(f'CLIP video wrong length {video_id}, '
                               f'expected {self.clip_expected_length}, '
                               f'got {clip_chunk.shape[0]}')
        clip_chunk = self.clip_transform(clip_chunk)

        sync_chunk = sync_chunk[:self.sync_expected_length]
        if sync_chunk.shape[0] != self.sync_expected_length:
            raise RuntimeError(f'Sync video wrong length {video_id}, '
                               f'expected {self.sync_expected_length}, '
                               f'got {sync_chunk.shape[0]}')
        sync_chunk = self.sync_transform(sync_chunk)

        data = {
            'name': video_id,
            'caption': caption,
            'clip_video': clip_chunk,
            'sync_video': sync_chunk,
        }

        return data
