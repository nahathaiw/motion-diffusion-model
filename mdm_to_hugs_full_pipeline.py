#!/usr/bin/env python3
"""
Complete MDM → HUGS Pipeline: From Motion Video to 3D Avatar Animation

This script handles the full workflow:
1. Load MDM-generated motion (optionally convert from results.npy)
2. Apply coordinate system transformation (90° rotation around X-axis)
3. Pass to HUGS for rendering
4. Generate final MP4 output

Coordinate System Handling:
- MDM uses Y-up convention (mocap standard)
- HUGS uses Z-up convention (graphics standard)
- Solution: 90° rotation around X-axis to map Y→Z, Z→Y, X→X

Usage:
    python mdm_to_hugs_full_pipeline.py \
        --input_motion /path/to/hugs_smpl_rotX90.npz \
        --output_video /path/to/output/my_animation.mp4 \
        --model_checkpoint /path/to/pretrained/human_final.pth \
        --fps 20
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add import paths
sys.path.insert(0, str(Path(__file__).parent))


def validate_motion_file(motion_path):
    """Validate that motion file has correct HUGS format."""
    print(f"📋 Validating motion file: {motion_path}")
    
    if not os.path.exists(motion_path):
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    
    data = np.load(motion_path)
    
    # Check required keys
    required_keys = {'global_orient', 'body_pose', 'transl', 'betas'}
    provided_keys = set(data.keys())
    
    if not required_keys.issubset(provided_keys):
        missing = required_keys - provided_keys
        raise ValueError(f"Missing keys in motion file: {missing}")
    
    # Check shapes
    shapes = {k: v.shape for k, v in data.items()}
    print(f"  ✓ Keys: {list(shapes.keys())}")
    print(f"  ✓ Shapes: {shapes}")
    
    n_frames = shapes['global_orient'][0]
    print(f"  ✓ Number of frames: {n_frames}")
    
    return n_frames, shapes


def create_hugs_config(motion_path, output_dir, fps=20, dataset_seq='custom'):
    """
    Create a minimal HUGS config for rendering custom motion.
    
    Returns path to created config file.
    """
    config_content = f"""# Auto-generated HUGS config for custom motion rendering
# Motion: {motion_path}

mode: 'human'
seed: 0

dataset:
  name: 'custom'
  seq: '{dataset_seq}'
  data_path: ''  # Not needed for motion-only rendering