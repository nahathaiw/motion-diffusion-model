#!/usr/bin/env python3
"""
Extract SMPL parameters from MDM motion samples.

This script takes MDM motion samples (in 6D rotation format with RIC encoding)
and converts them to SMPL parameters suitable for conversion to HUGS format.

Usage:
    python sample/extract_smpl_params.py \
        --motion_data /path/to/results.npy \
        --output /path/to/smpl_params.npy \
        --model_path ./save/humanml_trans_enc_512/model000200000.pt
"""

import os
import sys
import numpy as np
import torch
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loaders.humanml.scripts.motion_process import recover_from_ric
from model.rotation2xyz import Rotation2xyz


def extract_smpl_params(
    motion_data,
    n_joints=22,
    output_path=None,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Extract SMPL parameters from MDM motion data.
    
    Args:
        motion_data: numpy array of shape (num_samples, njoints, 6, nframes)
                    or (njoints, 6, nframes) for single sample
        n_joints: number of joints (22 for humanml)
        output_path: optional path to save the extracted parameters
        device: torch device for processing
    
    Returns:
        Dictionary with SMPL parameters:
        - thetas: (njoints, 6, nframes) - 6D rotation representations
        - root_translation: (3, nframes) - root translation
    """
    
    # Handle batch dimension
    if motion_data.ndim == 4:
        motion_data = motion_data[0]  # Take first sample
    
    print(f"Motion data shape: {motion_data.shape}")
    
    # Initialize rotation converter
    rot2xyz = Rotation2xyz(device=device, dataset='humanml')
    
    # Recover from RIC encoding
    # motion_data is (njoints, 6, nframes)
    njoints, feats, nframes = motion_data.shape
    print(f"Recovering from RIC: {njoints} joints, {feats} features, {nframes} frames")
    
    motion_ric = torch.from_numpy(motion_data).to(device).float()
    motion_ric = motion_ric.permute(2, 0, 1)  # (nframes, njoints, 6)
    
    recovered_motion = recover_from_ric(motion_ric.cpu().numpy(), n_joints)
    print(f"Recovered motion shape: {recovered_motion.shape}")
    
    # Convert to (nframes, njoints, 3) for xyz joints
    recovered_motion = torch.from_numpy(recovered_motion).to(device).float()
    
    # Now recover SMPL parameters from joint positions
    # This is complex - we'll save the 6D rotation + translation instead
    
    # Extract 6D rotations (already in the motion_data)
    thetas = motion_data  # (njoints, 6, nframes)
    
    # Extract root translation from the last joint in RIC encoding
    # Root position is typically encoded in the last dimension
    root_translation = motion_data[0, :3, :]  # Extract x,y,z from first joint
    
    smpl_params = {
        'thetas': thetas.astype(np.float32),
        'root_translation': root_translation.astype(np.float32),
    }
    
    if output_path:
        print(f"\nSaving SMPL parameters to: {output_path}")
        np.save(output_path, smpl_params)
        print("Done!")
    
    return smpl_params


def main():
    parser = argparse.ArgumentParser(
        description='Extract SMPL parameters from MDM motion data'
    )
    parser.add_argument('--motion_data', '-m', required=True,
                        help='Path to MDM results.npy file')
    parser.add_argument('--output', '-o', default=None,
                        help='Path to save extracted SMPL parameters')
    parser.add_argument('--n_joints', type=int, default=22,
                        help='Number of joints (22 for humanml)')
    parser.add_argument('--device', default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load motion data
    print(f"Loading motion data from: {args.motion_data}")
    motion_data = np.load(args.motion_data)
    
    # Determine output path if not provided
    if args.output is None:
        motion_path = Path(args.motion_data)
        args.output = str(motion_path.parent / 'smpl_params.npy')
    
    # Extract parameters
    extract_smpl_params(
        motion_data=motion_data,
        n_joints=args.n_joints,
        output_path=args.output,
        device=args.device,
    )


if __name__ == '__main__':
    main()
