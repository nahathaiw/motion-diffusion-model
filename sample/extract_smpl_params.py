#!/usr/bin/env python3
"""
Extract SMPL parameters from MDM motion results.npy and save as HUGS-compatible npz.

MDM outputs results.npy as a pickled dict with shape (num_samples, 22, 3, nframes)
containing XYZ joint positions.  This script uses SMPLify-3D (joints2smpl) to fit
SMPL pose parameters to those joint positions, then saves a hugs_smpl_original.npz
with keys:  global_orient (T,3), body_pose (T,69), transl (T,3), betas (10,)

Must be run from the motion-diffusion-model repo root so that relative model paths
(./body_models/...) resolve correctly.

Usage:
    cd /path/to/motion-diffusion-model
    python sample/extract_smpl_params.py \
        --motion_data save/.../results.npy \
        --output     save/.../hugs_smpl_original.npz
"""

import os
import sys
import numpy as np
import torch
import argparse
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualize.joints2smpl.src import config
from visualize.simplify_loc2rot import joints2smpl
import utils.rotation_conversions as geometry
import smplx
import h5py


def extract_smpl_params(
    results_npy_path,
    output_npz_path,
    device_id=0,
    cuda=True,
):
    """
    Load MDM results.npy, fit SMPL via SMPLify-3D, save hugs_smpl_original.npz.

    Args:
        results_npy_path : str | Path  – MDM results.npy file
        output_npz_path  : str | Path  – destination .npz file
        device_id        : int         – CUDA device index
        cuda             : bool        – use CUDA if True
    """
    results_npy_path = Path(results_npy_path)
    output_npz_path  = Path(output_npz_path)
    device = torch.device("cuda:" + str(device_id) if cuda else "cpu")

    print(f"Loading MDM results from: {results_npy_path}")
    results = np.load(str(results_npy_path), allow_pickle=True).item()

    motion = results['motion']           # (num_samples, 22, 3, nframes)
    length = int(results['lengths'][0])  # actual number of valid frames
    print(f"Motion shape: {motion.shape}  |  valid frames: {length}")

    # Take first sample, transpose to (nframes, 22, 3)
    joints = motion[0].transpose(2, 0, 1)[:length]   # (T, 22, 3)
    nframes = joints.shape[0]
    print(f"Using first sample: {joints.shape}")

    # ---- Set up SMPLify-3D directly (mirrors joints2smpl internals) ----
    print(f"\nFitting SMPL parameters ({nframes} frames) via SMPLify-3D...")
    print(config.SMPL_MODEL_DIR)

    from visualize.joints2smpl.src.smplify import SMPLify3D

    smplmodel = smplx.create(
        config.SMPL_MODEL_DIR,
        model_type="smpl", gender="neutral", ext="pkl",
        batch_size=nframes,
    ).to(device)

    file = h5py.File(config.SMPL_MEAN_FILE, 'r')
    init_pose  = torch.from_numpy(file['pose'][:]).unsqueeze(0).repeat(nframes, 1).float().to(device)
    init_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).repeat(nframes, 1).float().to(device)
    cam_zero   = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(device)

    smplify = SMPLify3D(
        smplxmodel=smplmodel,
        batch_size=nframes,
        joints_category="AMASS",
        num_iters=150,
        device=device,
    )

    keypoints_3d = torch.tensor(joints, dtype=torch.float32).to(device)
    confidence   = torch.ones(22).to(device)

    _, _, new_opt_pose, new_opt_betas, _, _ = smplify(
        init_pose.detach(),
        init_shape.detach(),
        cam_zero.detach(),
        keypoints_3d,
        conf_3d=confidence,
    )
    # new_opt_pose: (T, 72) axis-angle  |  new_opt_betas: (T, 10)

    poses = new_opt_pose.detach().cpu().numpy()   # (T, 72)
    betas = new_opt_betas[0].detach().cpu().numpy()[:10]  # (10,) – use first frame betas

    global_orient = poses[:, :3].astype(np.float32)   # (T, 3)
    body_pose     = poses[:, 3:].astype(np.float32)   # (T, 69)
    transl        = joints[:, 0, :].astype(np.float32) # (T, 3) root hip position

    print(f"\nSMPL parameters extracted:")
    print(f"  global_orient : {global_orient.shape}")
    print(f"  body_pose     : {body_pose.shape}")
    print(f"  transl        : {transl.shape}")
    print(f"  betas         : {betas.shape}")

    # ---- Save ----
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_npz_path),
             global_orient=global_orient,
             body_pose=body_pose,
             transl=transl,
             betas=betas)
    print(f"\n✓ Saved hugs_smpl_original.npz to: {output_npz_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract SMPL parameters from MDM results.npy → hugs_smpl_original.npz'
    )
    parser.add_argument('--motion_data', '-m', required=True,
                        help='Path to MDM results.npy file')
    parser.add_argument('--output', '-o', default=None,
                        help='Destination .npz file (default: same dir as motion_data, '
                             'named hugs_smpl_original.npz)')
    parser.add_argument('--device_id', type=int, default=0,
                        help='CUDA device index (default: 0)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if CUDA is available')

    args = parser.parse_args()

    motion_path = Path(args.motion_data)
    if args.output is None:
        output_path = motion_path.parent / 'hugs_smpl_original.npz'
    else:
        output_path = Path(args.output)

    cuda = (not args.cpu) and torch.cuda.is_available()
    if not cuda:
        print("Running on CPU (SMPLify will be slow)")

    extract_smpl_params(
        results_npy_path=motion_path,
        output_npz_path=output_path,
        device_id=args.device_id,
        cuda=cuda,
    )


if __name__ == '__main__':
    main()
