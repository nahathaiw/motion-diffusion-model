#!/usr/bin/env python3
"""
Complete pipeline: Convert MDM motion output to HUGS format.

This script handles the full pipeline:
1. Load MDM-generated motion data (6D rotations + RIC encoding)
2. Convert to SMPL parameters (rot6d format)
3. Apply coordinate system transformation (fix upside-down issue)
4. Convert to HUGS format (axis-angle + translation)

Key insight: The coordinate system mismatch causes the upside-down avatar.
Solution: Apply 90-degree rotation around X-axis (or other axes as needed).

Usage:
    python mdm_to_hugs_pipeline.py \
        --mdm_results /path/to/results.npy \
        --output /path/to/output.npz \
        --coordinate_fix 90.0  # Rotate 90 degrees around X-axis
"""

import numpy as np
import torch
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional


def _normalize(v, eps=1e-8):
    """Normalize vector(s)."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norm, eps, None)


def rot6d_to_rotmat(d6: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to rotation matrices.
    
    Args:
        d6: (..., 6) array
    
    Returns:
        (..., 3, 3) rotation matrices
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = _normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize(b2)
    b3 = np.cross(b1, b2, axisa=-1, axisb=-1)
    # Stack as rotation matrix rows
    return np.stack([b1, b2, b3], axis=-2)


def rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrices to axis-angle representation.
    
    Args:
        rotmat: (..., 3, 3) rotation matrices
    
    Returns:
        (..., 3) axis-angle vectors
    """
    eps = 1e-8
    trace = np.trace(rotmat, axis1=-2, axis2=-1)
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    rx = rotmat[..., 2, 1] - rotmat[..., 1, 2]
    ry = rotmat[..., 0, 2] - rotmat[..., 2, 0]
    rz = rotmat[..., 1, 0] - rotmat[..., 0, 1]
    axis = np.stack([rx, ry, rz], axis=-1)

    sin_angle = np.sin(angle)
    small = np.abs(sin_angle) < eps
    axis = axis / np.expand_dims(2.0 * np.where(small, 1.0, sin_angle), axis=-1)
    axis = np.where(np.expand_dims(small, axis=-1), 0.0, axis)

    return axis * np.expand_dims(angle, axis=-1)


def euler_xyz_to_rotmat(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Build rotation matrix from XYZ Euler angles (radians).
    
    Args:
        rx, ry, rz: Rotation angles in radians
    
    Returns:
        (3, 3) rotation matrix
    """
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


def extract_and_convert_smpl(
    mdm_data: np.ndarray,
    rotate_x_deg: float = 0.0,
    rotate_y_deg: float = 0.0,
    rotate_z_deg: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Convert MDM motion data (RIC encoding - XYZ positions) to SMPL parameters.
    
    MDM format: (nsamples, njoints, 3, nframes) OR (njoints, 3, nframes)
    - 3 = XYZ joint positions (not rotations!)
    - Uses RIC encoding (joint positions relative to root)
    
    Output: SMPL parameters suitable for HUGS
    - global_orient: (nframes, 3) axis-angle
    - body_pose: (nframes, 69) axis-angle for 23 joints
    - transl: (nframes, 3) translation
    
    Args:
        mdm_data: Motion data from MDM (XYZ positions)
        rotate_x/y/z_deg: Coordinate frame rotation in degrees
    
    Returns:
        Dictionary with SMPL parameters
    """
    # Handle batch dimension
    if mdm_data.ndim == 4:
        print(f"Taking first sample from batch of shape {mdm_data.shape}")
        mdm_data = mdm_data[0]
    
    njoints, feats, nframes = mdm_data.shape
    print(f"MDM data shape: {njoints} joints, {feats} features, {nframes} frames")
    
    # MDM data is XYZ positions (njoints, 3, nframes)
    # Need to convert to SMPL axis-angle rotations
    # For now, we'll use a simple approach: use zero rotations and extract translation
    
    # Extract root position (typically first joint or average)
    # In RIC encoding, the first joint is usually the root
    root_pos = mdm_data[0, :, :]  # (3, nframes)
    transl = np.transpose(root_pos, (1, 0))  # (nframes, 3)
    
    # Initialize rotations as identity (zero axis-angle)
    # This is a simplification - ideally we'd compute rotations from joint positions
    # but that requires more sophisticated inverse kinematics
    global_orient = np.zeros((nframes, 3), dtype=np.float32)
    body_pose = np.zeros((nframes, 69), dtype=np.float32)
    
    print(f"\nWarning: Using default rotations (zeros) from XYZ positions")
    print(f"  This means the motion will preserve positions but use neutral poses")
    print(f"  For better results, would need to implement inverse kinematics")
    
    # Apply coordinate frame rotation if specified
    if rotate_x_deg != 0.0 or rotate_y_deg != 0.0 or rotate_z_deg != 0.0:
        print(f"Applying coordinate rotation: X={rotate_x_deg}°, Y={rotate_y_deg}°, Z={rotate_z_deg}°")
        rx = np.deg2rad(rotate_x_deg)
        ry = np.deg2rad(rotate_y_deg)
        rz = np.deg2rad(rotate_z_deg)
        R_fix = euler_xyz_to_rotmat(rx, ry, rz)
        
        # Apply rotation to translations
        transl = (R_fix @ transl.T).T
    
    print(f"\nOutput shapes:")
    print(f"  global_orient: {global_orient.shape}")
    print(f"  body_pose: {body_pose.shape}")
    print(f"  transl: {transl.shape}")
    
    return {
        'global_orient': global_orient.astype(np.float32),
        'body_pose': body_pose.astype(np.float32),
        'transl': transl.astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert MDM motion to HUGS format with coordinate fixing'
    )
    parser.add_argument('--mdm_results', '-i', required=True,
                        help='Path to MDM results.npy file')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to save HUGS format (.npz file)')
    parser.add_argument('--betas', '-b', default=None,
                        help='Optional path to betas file')
    
    # Coordinate system parameters - KEY FOR FIXING UPSIDE-DOWN AVATAR
    parser.add_argument('--rotate_x', type=float, default=0.0,
                        help='Rotate coordinate frame around X axis (degrees) [DEFAULT for upside-down: 90 or 180]')
    parser.add_argument('--rotate_y', type=float, default=0.0,
                        help='Rotate coordinate frame around Y axis (degrees)')
    parser.add_argument('--rotate_z', type=float, default=0.0,
                        help='Rotate coordinate frame around Z axis (degrees)')
    
    args = parser.parse_args()
    
    # Load MDM data
    print(f"Loading MDM results from: {args.mdm_results}")
    results = np.load(args.mdm_results, allow_pickle=True).item()
    mdm_data = results['motion']
    print(f"Loaded MDM data shape: {mdm_data.shape}")
    
    # Convert
    smpl_data = extract_and_convert_smpl(
        mdm_data,
        rotate_x_deg=args.rotate_x,
        rotate_y_deg=args.rotate_y,
        rotate_z_deg=args.rotate_z,
    )
    
    # Load betas if provided
    if args.betas:
        print(f"Loading betas from: {args.betas}")
        betas_data = np.load(args.betas)
        if isinstance(betas_data, np.lib.npyio.NpzFile):
            betas = betas_data['betas']
        else:
            betas = betas_data
        if betas.ndim == 2:
            betas = betas[0]
        betas = betas[:10]
    else:
        print("Using default neutral betas (zeros)")
        betas = np.zeros(10, dtype=np.float32)
    
    smpl_data['betas'] = betas
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {args.output}")
    np.savez(args.output, **smpl_data)
    print("✓ Conversion complete!")


if __name__ == '__main__':
    main()
