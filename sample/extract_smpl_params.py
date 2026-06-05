#!/usr/bin/env python3
"""
Extract SMPL parameters from MDM motion results.npy and save as HUGS-compatible npz.

Fits SMPL pose parameters to the joint XYZ positions via SMPLify-3D.
Runs on GPU if available (fast, ~10-30s); falls back to CPU (slow, ~5min).

Must be run from the motion-diffusion-model repo root so that relative model paths
(./body_models/...) resolve correctly.

Usage:
    cd /path/to/motion-diffusion-model
    python sample/extract_smpl_params.py \
        --motion_data save/.../results.npy \
        --output     save/.../hugs_smpl_original.npz
"""

import sys
import numpy as np
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualize.joints2smpl.src import config
from visualize.joints2smpl.src.smplify import SMPLify3D
import smplx
import h5py


def extract_smpl_params(
    results_npy_path,
    output_npz_path,
    device_id=0,
    cuda=True,
    num_iters=150,
):
    results_npy_path = Path(results_npy_path)
    output_npz_path  = Path(output_npz_path)

    print(f"Loading MDM results from: {results_npy_path}")
    results = np.load(str(results_npy_path), allow_pickle=True).item()

    motion = results['motion']           # (num_samples, 22, 3, nframes)
    length = int(results['lengths'][0])
    print(f"Motion shape: {motion.shape}  |  valid frames: {length}")

    joints = motion[0].transpose(2, 0, 1)[:length]   # (T, 22, 3)
    nframes = joints.shape[0]

    device = torch.device(f"cuda:{device_id}" if cuda and torch.cuda.is_available() else "cpu")
    print(f"Fitting SMPL ({nframes} frames, {num_iters} iters) on {device}")

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
        num_iters=num_iters,
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

    poses = new_opt_pose.detach().cpu().numpy()
    betas = new_opt_betas[0].detach().cpu().numpy()[:10]

    global_orient = poses[:, :3].astype(np.float32)
    body_pose     = poses[:, 3:].astype(np.float32)
    transl        = joints[:, 0, :].astype(np.float32)

    print(f"\nSMPL parameters extracted:")
    print(f"  global_orient : {global_orient.shape}")
    print(f"  body_pose     : {body_pose.shape}")
    print(f"  transl        : {transl.shape}")
    print(f"  betas         : {betas.shape}")

    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_npz_path),
             global_orient=global_orient,
             body_pose=body_pose,
             transl=transl,
             betas=betas)
    print(f"\n✓ Saved to: {output_npz_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion_data', '-m', required=True)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--num_iters', type=int, default=150,
                        help='SMPLify-3D iterations (default: 150; use 50 for faster/slightly rougher fit)')

    args = parser.parse_args()

    motion_path = Path(args.motion_data)
    output_path = Path(args.output) if args.output else motion_path.parent / 'hugs_smpl_original.npz'
    cuda = (not args.cpu) and torch.cuda.is_available()

    extract_smpl_params(
        results_npy_path=motion_path,
        output_npz_path=output_path,
        device_id=args.device_id,
        cuda=cuda,
        num_iters=args.num_iters,
    )


if __name__ == '__main__':
    main()
