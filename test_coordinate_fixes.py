#!/usr/bin/env python3
"""
Diagnostic tool to test different coordinate transformations.

This script helps identify which coordinate system transformation fixes
the upside-down avatar issue. It will convert MDM output with different
rotation parameters and save multiple versions for testing in HUGS.

Usage:
    python test_coordinate_fixes.py \
        --mdm_results /path/to/results.npy \
        --output_dir ./test_conversions \
        --test_all
"""

import numpy as np
import argparse
import sys
from pathlib import Path

# Add parent to path to import the pipeline
sys.path.insert(0, str(Path(__file__).parent))
from mdm_to_hugs_pipeline import extract_and_convert_smpl


def test_all_transformations(mdm_data: np.ndarray, output_dir: Path):
    """
    Test common coordinate transformations and save results.
    
    Common fixes:
    - 90° X rotation: Swaps Y-up to Z-up (common for motion capture)
    - 180° X rotation: Flips Y and Z
    - -90° X rotation: Opposite of 90°
    """
    
    test_configs = [
        {"name": "original", "rx": 0, "ry": 0, "rz": 0},
        {"name": "90deg_x", "rx": 90, "ry": 0, "rz": 0},
        {"name": "neg90deg_x", "rx": -90, "ry": 0, "rz": 0},
        {"name": "180deg_x", "rx": 180, "ry": 0, "rz": 0},
        {"name": "90deg_y", "rx": 0, "ry": 90, "rz": 0},
        {"name": "90deg_z", "rx": 0, "ry": 0, "rz": 90},
        {"name": "180deg_y", "rx": 0, "ry": 180, "rz": 0},
        {"name": "flip_xy_swap_z", "rx": 90, "ry": 0, "rz": 180},  # Y-up to Z-up with Z-flip
    ]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"  Rotation: X={config['rx']}°, Y={config['ry']}°, Z={config['rz']}°")
        print(f"{'='*60}")
        
        try:
            smpl_data = extract_and_convert_smpl(
                mdm_data,
                rotate_x_deg=config['rx'],
                rotate_y_deg=config['ry'],
                rotate_z_deg=config['rz'],
            )
            
            # Add default betas
            smpl_data['betas'] = np.zeros(10, dtype=np.float32)
            
            # Save
            output_file = output_dir / f"{config['name']}.npz"
            np.savez(output_file, **smpl_data)
            print(f"✓ Saved to: {output_file}")
            
            # Store first frame joint positions for inspection
            n_joints = 24
            results[config['name']] = {
                'global_orient': smpl_data['global_orient'][0],
                'body_pose': smpl_data['body_pose'][0],
                'transl': smpl_data['transl'][0],
            }
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary of generated files:")
    print(f"{'='*60}")
    for f in sorted(output_dir.glob("*.npz")):
        size = f.stat().st_size / 1024
        print(f"  {f.name:<30} ({size:.1f} KB)")
    
    print(f"\nTest all these in HUGS and check which one has the correct orientation:")
    print(f"  python main.py --motion_file {output_dir / 'VARIANT_NAME.npz'}")
    print(f"\nLikely correct: 90deg_x or flip_xy_swap_z")


def main():
    parser = argparse.ArgumentParser(
        description='Test different coordinate transformations'
    )
    parser.add_argument('--mdm_results', '-i', required=True,
                        help='Path to MDM results.npy file')
    parser.add_argument('--output_dir', '-o', default='./test_conversions',
                        help='Directory to save test files')
    
    args = parser.parse_args()
    
    # Load MDM data
    print(f"Loading MDM results from: {args.mdm_results}")
    results = np.load(args.mdm_results, allow_pickle=True).item()
    
    # Extract motion data
    mdm_data = results['motion']
    print(f"Loaded shape: {mdm_data.shape}")
    
    # Test transformations
    test_all_transformations(mdm_data, args.output_dir)


if __name__ == '__main__':
    main()
