#!/usr/bin/env python3
"""Simple SMPL control demo.

Creates a short sequence of SMPL meshes by perturbing body_pose parameters
and saves per-frame OBJ files plus a `_smpl_params.npy` containing the
SMPL parameters used. This is intentionally simple so you can experiment
with different body_pose indices; results may look broken but show control.
"""
import os
import argparse
import numpy as np
import torch

from model.smpl import SMPL


def write_obj(filename, vertices, faces):
    # vertices: (V,3), faces: (F,3)  (faces are 0-based indices)
    with open(filename, 'w') as f:
        for v in vertices:
            f.write('v {:.6f} {:.6f} {:.6f}\n'.format(float(v[0]), float(v[1]), float(v[2])))
        for face in faces:
            # OBJ format is 1-based
            f.write('f {} {} {}\n'.format(int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='./debug_smpl_output', help='Where to save OBJ frames and params')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id (uses CPU if unavailable)')
    parser.add_argument('--frames', type=int, default=16, help='Number of frames to create')
    parser.add_argument('--joint_idx', type=int, default=16, help='Body-pose joint index to perturb (0-based joint id)')
    parser.add_argument('--amplitude', type=float, default=1.0, help='Amplitude of axis-angle perturbation')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device}' if use_cuda else 'cpu')

    os.makedirs(args.out_dir, exist_ok=True)

    # Instantiate SMPL wrapper from the repo
    smpl = SMPL().to(device).eval()

    # Prepare neutral params
    batch = 1
    num_betas = 10
    # body_pose size depends on SMPL implementation. We'll query a forward pass shape
    # SMPL expects global_orient (1,3) and body_pose (1, 3*(N-1)) where N is 24 joints
    global_orient = torch.zeros((batch, 3), dtype=torch.float32, device=device)
    betas = torch.zeros((batch, num_betas), dtype=torch.float32, device=device)

    # Create neutral body_pose vector with zeros; length 69 (23*3) is typical
    body_pose = torch.zeros((batch, 69), dtype=torch.float32, device=device)

    # We'll perturb the specified joint index across frames with a simple sinusoidal
    joint_idx = args.joint_idx
    if joint_idx < 0 or joint_idx >= body_pose.shape[1] // 3:
        print(f'Warning: joint_idx {joint_idx} is out of range for body_pose with {body_pose.shape[1]//3} joints. Clamping.')
        joint_idx = max(0, min(joint_idx, body_pose.shape[1] // 3 - 1))

    all_params = []
    faces = None
    for fi in range(args.frames):
        t = fi / float(args.frames - 1) if args.frames > 1 else 0.0
        angle = np.sin(t * np.pi * 2) * args.amplitude

        bp = body_pose.clone()
        # set axis-angle (rx, ry, rz) for the chosen joint
        # we'll rotate around X axis for demonstration
        bp[0, joint_idx * 3 + 0] = angle

        # smplx may expect rotation matrices for global_orient depending on
        # version/config. Convert axis-angle (3) to rotation matrix (1x9) to
        # be safe.
        def axis_angle_to_rotmat(a):
            # a: (B,3) axis-angle
            B = a.shape[0]
            theta = torch.norm(a, dim=1, keepdim=True)  # (B,1)
            eps = 1e-8
            # safe unit axis
            k = torch.zeros_like(a)
            mask = (theta.squeeze(1) > eps)
            if mask.any():
                k[mask] = a[mask] / theta[mask]

            # build skew-symmetric K
            K = torch.zeros((B, 3, 3), device=a.device, dtype=a.dtype)
            kx = k[:, 0]; ky = k[:, 1]; kz = k[:, 2]
            K[:, 0, 1] = -kz; K[:, 0, 2] = ky
            K[:, 1, 0] = kz; K[:, 1, 2] = -kx
            K[:, 2, 0] = -ky; K[:, 2, 1] = kx

            I = torch.eye(3, device=a.device, dtype=a.dtype).unsqueeze(0).repeat(B, 1, 1)
            sin_t = torch.sin(theta).unsqueeze(-1)  # (B,1,1)
            cos_t = torch.cos(theta).unsqueeze(-1)
            theta2 = (1 - cos_t)
            R = I + sin_t * K + theta2 * torch.bmm(K, K)
            return R.view(B, 9)

        aa = global_orient
        aa_rot = axis_angle_to_rotmat(aa)

        # convert body_pose (B, 3*num_joints) -> rotation matrices (B, num_joints*9)
        def axis_angle_many_to_rotmat(flat_a):
            # flat_a: (B, 3*n) or (B, n, 3)
            if flat_a.dim() == 2:
                B, C = flat_a.shape
                n = C // 3
                a_reshaped = flat_a.view(B, n, 3)
            else:
                a_reshaped = flat_a
                B, n, _ = a_reshaped.shape
            a_flat = a_reshaped.contiguous().view(B * n, 3)
            R_flat = axis_angle_to_rotmat(a_flat)  # (B*n, 9)
            R = R_flat.view(B, n * 9)
            return R

        bp_rot = axis_angle_many_to_rotmat(bp)

        with torch.no_grad():
            out = smpl(global_orient=aa_rot, body_pose=bp_rot, betas=betas)
        verts = out['vertices'].cpu().numpy().squeeze()
        # faces: try to retrieve from the SMPL layer object
        try:
            # smpl may have attribute faces_tensor
            faces_tensor = getattr(smpl, 'faces_tensor', None)
            if faces_tensor is None:
                faces_tensor = getattr(smpl, 'faces', None)
            if faces_tensor is not None:
                faces = faces_tensor.cpu().numpy().reshape(-1, 3)
        except Exception:
            faces = None

        obj_path = os.path.join(args.out_dir, f'frame{fi:03d}.obj')
        if faces is None:
            # If we don't have faces, just save vertices as a point cloud (still an OBJ)
            with open(obj_path, 'w') as f:
                for v in verts:
                    f.write('v {:.6f} {:.6f} {:.6f}\n'.format(float(v[0]), float(v[1]), float(v[2])))
        else:
            write_obj(obj_path, verts, faces)

        # store params for inspection
        all_params.append({'global_orient': global_orient.cpu().numpy(),
                           'body_pose': bp.cpu().numpy(),
                           'betas': betas.cpu().numpy()})

    # Save params as npy
    np.save(os.path.join(args.out_dir, 'debug_smpl_params.npy'), all_params)
    print(f'Wrote {len(all_params)} frames to {args.out_dir} (OBJ files + debug_smpl_params.npy)')


if __name__ == '__main__':
    main()
