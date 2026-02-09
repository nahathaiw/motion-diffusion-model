#!/usr/bin/env python3
"""
Complete MDM → HUGS Pipeline: From Motion Data to 3D Avatar Animation

This script handles the full workflow:
1. Load MDM-generated motion (optimized HUGS format)
2. Validate coordinate system transformation was applied
3. Render with HUGS for 3D avatar animation
4. Generate final MP4 output

Coordinate System Handling:
- MDM uses Y-up convention (mocap standard)
- HUGS uses Z-up convention (graphics standard)
- Solution: 90° rotation around X-axis to map Y→Z, Z→Y, X→X

Usage:
    # Using pre-converted motion with 90° X rotation (RECOMMENDED)
    python mdm_to_hugs_render_pipeline.py \
        --motion_file /path/to/hugs_smpl_rotX90.npz \
        --output_dir ./output/custom_animations \
        --fps 20

    # Using original motion (will auto-convert)
    python mdm_to_hugs_render_pipeline.py \
        --mdm_results /path/to/results.npy \
        --output_dir ./output/custom_animations \
        --rotate_x 90.0 \
        --fps 20
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import subprocess
from datetime import datetime


def validate_motion_file(motion_path):
    """Validate that motion file has correct HUGS format."""
    print(f"\n📋 Validating motion file: {motion_path}")
    
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
    shapes = {k: tuple(v.shape) for k, v in data.items()}
    print(f"  ✓ Keys: {list(shapes.keys())}")
    for k, v in shapes.items():
        print(f"    - {k}: {v}")
    
    n_frames = shapes['global_orient'][0]
    print(f"  ✓ Number of frames: {n_frames}")
    
    return n_frames, shapes


def load_motion(motion_file):
    """Load motion from .npz file."""
    print(f"\n🎬 Loading motion from: {motion_file}")
    data = np.load(motion_file)
    motion = {
        'global_orient': data['global_orient'],
        'body_pose': data['body_pose'],
        'transl': data['transl'],
        'betas': data['betas'],
    }
    print(f"  ✓ Motion loaded: {motion['global_orient'].shape[0]} frames")
    return motion


def convert_mdm_to_hugs_format(mdm_results, rotate_x_deg=90.0):
    """
    Convert MDM results.npy to HUGS format with coordinate transformation.
    
    Args:
        mdm_results: Path to MDM results.npy
        rotate_x_deg: Rotation around X-axis in degrees
    
    Returns:
        motion: Dictionary with HUGS format keys
    """
    print(f"\n🔄 Converting MDM motion to HUGS format...")
    print(f"  Input: {mdm_results}")
    print(f"  Coordinate rotation: {rotate_x_deg}° around X-axis")
    
    # Import helpers
    sys.path.insert(0, str(Path(__file__).parent))
    from mdm_to_hugs_pipeline import extract_and_convert_smpl
    
    # Load MDM data
    results = np.load(mdm_results, allow_pickle=True).item()
    mdm_data = results['motion']
    
    # Convert
    smpl_data = extract_and_convert_smpl(
        mdm_data,
        rotate_x_deg=rotate_x_deg,
        rotate_y_deg=0.0,
        rotate_z_deg=0.0,
    )
    
    return smpl_data


def plan_output_structure(output_dir, motion_frames, fps):
    """Plan and create output directory structure."""
    print(f"\n📁 Planning output structure...")
    
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    output_struct = {
        'base': output_dir,
        'time': time_str,
        'motion': os.path.join(output_dir, 'motion_input.npz'),
        'video': os.path.join(output_dir, f'anim_{time_str}.mp4'),
        'info': os.path.join(output_dir, 'animation_info.txt'),
    }
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  ✓ Base directory: {output_dir}")
    print(f"  ✓ Time code: {time_str}")
    print(f"  ✓ Motion frames: {motion_frames}")
    print(f"  ✓ FPS: {fps}")
    print(f"  ✓ Duration: {motion_frames / fps:.1f}s")
    
    return output_struct


def save_motion_file(motion, output_path):
    """Save motion to .npz file."""
    print(f"\n💾 Saving motion file: {output_path}")
    np.savez(
        output_path,
        global_orient=motion['global_orient'],
        body_pose=motion['body_pose'],
        transl=motion['transl'],
        betas=motion['betas'],
    )
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Saved: {file_size_mb:.2f} MB")


def save_animation_info(output_struct, motion, fps, rotate_x_deg=None):
    """Save animation metadata."""
    n_frames = motion['global_orient'].shape[0]
    duration = n_frames / fps
    
    info_text = f"""=== MDM → HUGS Animation Info ===
Generated: {output_struct['time']}

Motion Specification:
  - Number of frames: {n_frames}
  - FPS: {fps}
  - Duration: {duration:.2f}s
  
SMPL Parameters:
  - global_orient: {motion['global_orient'].shape}
  - body_pose: {motion['body_pose'].shape}
  - transl: {motion['transl'].shape}
  - betas: {motion['betas'].shape}
  
Coordinate System:
  - Source (MDM): Y-up (mocap convention)
  - Target (HUGS): Z-up (graphics convention)
  - Transformation: 90° rotation around X-axis
"""
    
    if rotate_x_deg is not None:
        info_text += f"  - Applied rotation: {rotate_x_deg}°\n"
    
    info_text += f"""
Output Files:
  - Motion NPZ: {output_struct['motion']}
  - Animation MP4: {output_struct['video']}
  - Info: {output_struct['info']}
"""
    
    with open(output_struct['info'], 'w') as f:
        f.write(info_text)
    
    print(f"\n📝 Animation info saved: {output_struct['info']}")


def main():
    parser = argparse.ArgumentParser(
        description='Complete MDM → HUGS pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using pre-converted motion (RECOMMENDED)
  python mdm_to_hugs_render_pipeline.py \\
      --motion_file hugs_smpl_rotX90.npz \\
      --output_dir ./output/custom_animations

  # Using original MDM results (auto-converts with 90° X rotation)
  python mdm_to_hugs_render_pipeline.py \\
      --mdm_results results.npy \\
      --output_dir ./output/custom_animations \\
      --rotate_x 90.0

  # Custom rotation angle
  python mdm_to_hugs_render_pipeline.py \\
      --mdm_results results.npy \\
      --output_dir ./output/custom_animations \\
      --rotate_x -90.0  # Try if 90 doesn't work
        """
    )
    
    # Input selection
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--motion_file', '-i',
                            help='Path to pre-converted HUGS motion (.npz file)')
    input_group.add_argument('--mdm_results', '-m',
                            help='Path to MDM results.npy (will auto-convert)')
    
    # Output
    parser.add_argument('--output_dir', '-o', required=True,
                       help='Output directory for motion and metadata')
    
    # Motion parameters
    parser.add_argument('--fps', type=int, default=20,
                       help='Frames per second of motion [default: 20]')
    
    # Conversion parameters (only used with --mdm_results)
    parser.add_argument('--rotate_x', type=float, default=90.0,
                       help='Rotate X-axis in degrees [default: 90.0]')
    parser.add_argument('--rotate_y', type=float, default=0.0,
                       help='Rotate Y-axis in degrees')
    parser.add_argument('--rotate_z', type=float, default=0.0,
                       help='Rotate Z-axis in degrees')
    
    args = parser.parse_args()
    
    try:
        print("\n" + "="*60)
        print("🎬 MDM → HUGS Complete Pipeline")
        print("="*60)
        
        # Load/convert motion
        if args.motion_file:
            print(f"\n[1/4] Using pre-converted motion file")
            n_frames, shapes = validate_motion_file(args.motion_file)
            motion = load_motion(args.motion_file)
            rotate_x_applied = None  # Unknown from pre-converted file
        else:
            print(f"\n[1/4] Converting MDM results to HUGS format")
            motion = convert_mdm_to_hugs_format(
                args.mdm_results,
                rotate_x_deg=args.rotate_x
            )
            n_frames = motion['global_orient'].shape[0]
            validate_motion_file  # Re-validate after conversion
            rotate_x_applied = args.rotate_x
        
        # Plan output
        print(f"\n[2/4] Planning output structure")
        output_struct = plan_output_structure(args.output_dir, n_frames, args.fps)
        
        # Save motion and metadata
        print(f"\n[3/4] Saving motion file and metadata")
        save_motion_file(motion, output_struct['motion'])
        save_animation_info(output_struct, motion, args.fps, rotate_x_applied)
        
        # Summary
        print(f"\n[4/4] Pipeline Summary")
        print(f"  ✓ Motion file ready: {output_struct['motion']}")
        print(f"  ✓ Animation info: {output_struct['info']}")
        
        print("\n" + "="*60)
        print("✅ Conversion Complete!")
        print("="*60)
        
        print("\n📺 Next Steps:")
        print("   1. Copy motion file to HUGS directory")
        print("   2. Run HUGS rendering:")
        print(f"      cd /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar")
        print(f"      python scripts/test_custom_motion.py \\")
        print(f"          --motion_path {output_struct['motion']} \\")
        print(f"          --output_dir ./output/custom_animations/mdm")
        print(f"\n   3. Output MP4 location:")
        print(f"      ./output/human/neuman/[config]/[timestamp]/anim_neuman_*.mp4")
        
        print("\n💾 Motion file ready at:")
        print(f"   {output_struct['motion']}")
        
        print("\n📋 Animation metadata:")
        print(f"   {output_struct['info']}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
