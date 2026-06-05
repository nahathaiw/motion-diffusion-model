# 🎬 Complete MDM → HUGS Pipeline Guide

**Convert Motion Diffusion Model (MDM) outputs to 3D avatar animations with ML-HUGS**

This guide provides step-by-step instructions, scripts, and coordinate system fixes for converting text-to-motion animations from MDM into rendered 3D avatars using HUGS.

## 📚 Table of Contents
1. [Quick Start](#quick-start)
2. [Coordinate System Details](#coordinate-system-details)
3. [Step-by-Step Pipeline](#step-by-step-pipeline)
4. [Using the Scripts](#using-the-scripts)
5. [Your Sample Data](#your-sample-data)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

---

## 🚀 Quick Start

### Option 1: Using Shell Script (Easiest)
```bash
cd /home/sigma/skibidi/motion-diffusion-model

bash mdm_to_hugs_complete_pipeline.sh \
    ./save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10_a_person_jumps/hugs_smpl_rotX90.npz \
    ./output_animation \
    20
```

### Option 2: Using Python Script (More Control)
```bash
cd /home/sigma/skibidi/motion-diffusion-model

python mdm_to_hugs_render_pipeline.py \
    --motion_file ./save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10_a_person_jumps/hugs_smpl_rotX90.npz \
    --output_dir ./output_animation \
    --fps 20
```

Both create a motion file ready for HUGS rendering.

---

## 🔄 Coordinate System Details

### The Problem
MDM and HUGS use **different coordinate conventions**:

```
Motion Diffusion Model (MDM)     HUGS (ML-HUGS-NTHUavatar)
─────────────────────────────     ─────────────────────────
Y-Up Convention (Mocap)           Z-Up Convention (Graphics)

    Y (Up)                            Z (Up)
    |                                 |
    |  Z (Forward)                    |  Y (Right)
    | /                               | /
    |/                                |/
────+────── X (Right)          ─────+────── X (Forward)

Normal for motion capture,      Common in 3D graphics,
SMPL, AMASS                     Gaussian rendering
```

### The Solution: 90° Rotation Around X-Axis

We apply a **90-degree rotation around the X-axis** to map between coordinate systems:

**Transformation Matrix:**
```
[x']   [1    0    0] [x]
[y'] = [0    0   -1] [y]    ← 90° rotation around X
[z']   [0    1    0] [z]

Results:
  MDM Y (Up)      → HUGS Z (Up)     ✓ Correct!
  MDM Z (Forward) → HUGS -Y
  MDM X (Right)   → HUGS X
```

### Why This Matters
- **Without transformation**: Avatar appears upside-down or sideways
- **With transformation**: Avatar stands upright, motions look natural

**Your pre-converted files in the sample directory:**
- ✅ `hugs_smpl_rotX90.npz` - 90° X rotation (RECOMMENDED) 
- ⚠️ `hugs_smpl_rotX-90.npz` - -90° X rotation (alternative)
- ❌ `hugs_smpl_original.npz` - No rotation (likely upside-down)

---

## 📋 Step-by-Step Pipeline

### Step 1: Input - MDM Generated Motion

**Source:** Your MDM sample directory
```
/home/sigma/skibidi/motion-diffusion-model/
  save/humanml_enc_512_50steps/
    samples_humanml_enc_512_50steps_000750000_seed10_a_person_jumps/
      ├── results.npy                    ← Original MDM output
      ├── sample00_rep00_smpl_params.npy ← SMPL parameters
      ├── hugs_smpl.npz                 ← Basic conversion (no rotation)
      ├── hugs_smpl_rotX90.npz          ← ✓ YOUR MAIN INPUT
      ├── hugs_smpl_rotX-90.npz         ← Alternative rotation
      └── samples_00_to_00.mp4          ← Reference video
```

**Motion Structure:**
- 120 frames @ 20 FPS = 6.0 seconds
- Each frame has:
  - `global_orient` (root rotation): (3,) axis-angle
  - `body_pose` (joint rotations): (69,) = 23 joints × 3 values
  - `transl` (root position): (3,) = X, Y, Z coordinates
  - `betas` (body shape): (10,) = SMPL shape parameters

### Step 2: Validate Motion Format

The conversion script automatically validates:
```bash
python mdm_to_hugs_render_pipeline.py --motion_file hugs_smpl_rotX90.npz
```

**Checks:**
- ✓ File exists and is readable
- ✓ Contains required keys: `global_orient`, `body_pose`, `transl`, `betas`
- ✓ Correct shapes:
  - `global_orient`: (N_frames, 3)
  - `body_pose`: (N_frames, 69)
  - `transl`: (N_frames, 3)
  - `betas`: (10,) or (N_frames, 10)

### Step 3: Prepare Motion for HUGS

**Output directory with metadata:**
```
output_animation/
├── motion_input.npz          ← Motion file (copy of input)
└── animation_info.txt        ← Metadata about the animation
```

**Metadata includes:**
- Input source
- Number of frames and duration
- Coordinate transformation applied
- Next steps for rendering

### Step 4: Render with HUGS

Once motion file is ready, render the animation:

```bash
cd /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar

python scripts/test_custom_motion.py \
    --motion_path /path/to/motion_input.npz \
    --output_dir ./output/custom_animations/mdm
```

**Expected outputs:**
```
./output/human/neuman/custom/mdm/[TIMESTAMP]/
├── anim_neuman_bike_final.mp4      ← Your animation!
├── anim_neuman_bike_canonical.mp4
└── [other intermediate files]
```

### Step 5: View Results

The final MP4 will be at:
```
/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/output/human/neuman/*/[timestamp]/anim_neuman_*.mp4
```

**Compare with:**
```
/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/output/human/neuman/bike/hugs_trimlp/20240303_1758_release_test-dataset.seq=citron/2026-01-24_17-38-31/anim_neuman_bike_final.mp4
```

---

## 🛠️ Using the Scripts

### Script 1: Python Render Pipeline
**File:** `mdm_to_hugs_render_pipeline.py`

**Purpose:** Validate motion, prepare metadata, and guide HUGS rendering

**Usage:**
```bash
# Using pre-converted motion (RECOMMENDED)
python mdm_to_hugs_render_pipeline.py \
    --motion_file hugs_smpl_rotX90.npz \
    --output_dir ./output \
    --fps 20

# Using original MDM results (auto-converts)
python mdm_to_hugs_render_pipeline.py \
    --mdm_results results.npy \
    --output_dir ./output \
    --rotate_x 90.0 \
    --fps 20

# Custom rotation angle (if motion is wrong orientation)
python mdm_to_hugs_render_pipeline.py \
    --mdm_results results.npy \
    --output_dir ./output \
    --rotate_x 180.0  # Try different angles: 90, 180, -90
```

**Options:**
```
--motion_file     Path to .npz motion file (pre-converted)
--mdm_results     Path to results.npy (will auto-convert)
--output_dir      Where to save motion and metadata
--fps             Frames per second [default: 20]
--rotate_x        X-axis rotation in degrees [default: 90.0]
--rotate_y        Y-axis rotation in degrees [default: 0.0]
--rotate_z        Z-axis rotation in degrees [default: 0.0]
```

### Script 2: Complete Pipeline Bash Script
**File:** `mdm_to_hugs_complete_pipeline.sh`

**Purpose:** Automate the entire pipeline in one command

**Usage:**
```bash
bash mdm_to_hugs_complete_pipeline.sh <motion_file> [output_dir] [fps]

# Example
bash mdm_to_hugs_complete_pipeline.sh hugs_smpl_rotX90.npz ./output 20
```

**What it does:**
1. ✓ Validates motion file format
2. ✓ Checks HUGS installation
3. ✓ Copies motion and creates metadata
4. ✓ Displays next steps
5. ✓ Optionally runs HUGS rendering

---

## 📁 Your Sample Data

### Input File Details

**MDM Sample Location:**
```
/home/sigma/skibidi/motion-diffusion-model/
save/humanml_enc_512_50steps/
samples_humanml_enc_512_50steps_000750000_seed10_a_person_jumps/
```

**Available Motion Files:**

1. **hugs_smpl_rotX90.npz** ⭐ RECOMMENDED
   - Size: 37 KB
   - Rotation: +90° around X-axis
   - Status: ✓ Correct orientation for HUGS
   - Use this file!

2. **hugs_smpl_rotX-90.npz**
   - Rotation: -90° around X-axis
   - Status: Try if rotX90 doesn't work

3. **hugs_smpl_original.npz**
   - Rotation: None (identity)
   - Status: Likely upside-down (for reference only)

4. **hugs_smpl.npz**
   - Extra field: `scale` parameter
   - Status: Advanced use only

### Quick Copy Command

Getting the file ready:
```bash
# From MDM directory
cp save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10_a_person_jumps/hugs_smpl_rotX90.npz ./motion_ready.npz

# Then render
cd /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar
python scripts/test_custom_motion.py --motion_path /path/to/motion_ready.npz
```

---

## 🐛 Troubleshooting

### Issue: Avatar is Upside-Down

**Cause:** Wrong coordinate transformation

**Solution:** Try alternative rotations
```bash
# Try 180° rotation
python mdm_to_hugs_render_pipeline.py \
    --mdm_results results.npy \
    --rotate_x 180.0

# Or try -90°
python mdm_to_hugs_render_pipeline.py \
    --mdm_results results.npy \
    --rotate_x -90.0

# Or try Y/Z axis rotations
python mdm_to_hugs_render_pipeline.py \
    --mdm_results results.npy \
    --rotate_y 90.0  # Try this if X rotations don't work
```

### Issue: Avatar is Sideways

**Cause:** Different forward direction

**Solution:**
```bash
# Try rotating around Y or Z axis
python mdm_to_hugs_render_pipeline.py \
    --mdm_results results.npy \
    --rotate_y 90.0   # Or try rotate_z 90.0
```

### Issue: Motion is Too Fast/Slow

**Cause:** FPS mismatch

**Solution:** Check MDM generation FPS
```bash
# Most MDM models use 20 FPS (default)
# If motion looks 2x too fast, double the duration:
python mdm_to_hugs_render_pipeline.py \
    --motion_file motion.npz \
    --fps 10  # Half the original FPS
```

### Issue: CUDA Out of Memory During Rendering

**Cause:** HUGS needs GPU memory for large resolutions

**Solution:**
- Reduce resolution in config
- Use smaller batch size
- Or run on a GPU with more memory

### Issue: Motion File Not Found

**Cause:** Path is relative or incorrect

**Solution:** Use absolute paths
```bash
# Get absolute path first
MOTION_FILE="/home/sigma/skibidi/motion-diffusion-model/save/.../hugs_smpl_rotX90.npz"

# Then use it
python mdm_to_hugs_render_pipeline.py --motion_file "$MOTION_FILE"
```

---

## 🔧 Advanced Usage

### Converting Multiple Motions at Once

**Create a batch script:**
```bash
#!/bin/bash
# batch_convert.sh

for motion_file in save/humanml_enc_512_50steps/*/hugs_smpl_rotX90.npz; do
    output_dir="./batch_output/$(basename $(dirname $motion_file))"
    echo "Converting: $motion_file → $output_dir"
    python mdm_to_hugs_render_pipeline.py \
        --motion_file "$motion_file" \
        --output_dir "$output_dir" \
        --fps 20
done
```

### Creating Config File for HUGS

**File:** `cfg_files/custom/mdm_motion.yaml`

```yaml
# Custom HUGS config for MDM motions
mode: 'human'
dataset:
  name: 'custom'
  seq: 'mdm_jumping'

# Motion input
motion:
  path: '/path/to/motion_input.npz'
  fps: 20

# Model
human:
  name: 'neuman'

# Rendering resolution
render:
  resolution: [1024, 1024]

# Training (set to 0 for motion-only, > 0 if you have video input)
train:
  iterations: 0

# Output
output_path: './output'
exp_name: 'mdm_jumping'
```

**Then render with:**
```bash
python main.py --cfg_file cfg_files/custom/mdm_motion.yaml --cfg_id 0
```

### Testing All Coordinate Transformations

```bash
# Generate 8 variants with different rotations
python test_coordinate_fixes.py \
    --mdm_results results.npy \
    --output_dir ./test_variants
```

**Generates:**
- `original.npz` (no rotation)
- `90deg_x.npz`, `180deg_x.npz`, `neg90deg_x.npz`
- `90deg_y.npz`
- `90deg_z.npz`
- Plus custom combinations

Then test each in HUGS to find the correct orientation.

### Custom Body Shape (Betas)

To use a specific body shape:

```bash
# Create motion with custom betas
python mdm_to_hugs_pipeline.py \
    --mdm_results results.npy \
    --output output.npz \
    --rotate_x 90.0 \
    --betas /path/to/custom_betas.npy
```

---

## 📊 Coordinate System Reference Table

| Component | MDM Convention | HUGS Convention | Transform |
|-----------|----------------|-----------------|-----------|
| **Up Direction** | Y-axis | Z-axis | Rotate 90° X |
| **Forward** | Z-axis | Y-axis | Rotate 90° X |
| **Right** | X-axis | X-axis | Unchanged |
| **Root Position** | (X, Y, Z) Y-up | (X, Y, Z) Z-up | Apply Rx90 |
| **Joint Angles** | Relative Y-up | Relative Z-up | Apply Rx90 |
| **Body Model** | SMPL (standard) | SMPL (standard) | Same |
| **SMPL Format** | axis-angle or rot6d | axis-angle | Convert if needed |

---

## 📐 Mathematical Details

### Rotation Matrix Formula

90° rotation around X-axis:
```
Rx(90°) = [ 1   0   0 ]
          [ 0   0  -1 ]
          [ 0   1   0 ]
```

### Applying to Positions
```
p_hugs = Rx(90°) @ p_mdm

Example: point at (x=1, y=2, z=3) in MDM becomes (1, -3, 2) in HUGS
```

### Applying to Rotations
```
R_hugs = Rx(90°) @ R_mdm @ Rx(90°)^T
```

This ensures rotations are consistent with the new coordinate frame.

---

## 🎯 Summary

**Your Pipeline:**
1. ✅ Input: `hugs_smpl_rotX90.npz` (already converted!)
2. ✅ Validate: `mdm_to_hugs_render_pipeline.py`
3. ✅ Export: Motion file ready for HUGS
4. ✅ Render: `python scripts/test_custom_motion.py`
5. ✅ Output: MP4 animation at `./output/human/neuman/.../anim_neuman_*.mp4`

**Next Command:**
```bash
bash mdm_to_hugs_complete_pipeline.sh hugs_smpl_rotX90.npz ./output 20
```

---

## 📞 Need Help?

**Check these files:**
- `animation_info.txt` - Metadata about your conversion
- `COORDINATE_SYSTEMS.md` - Deep coordinate system explanation
- `test_coordinate_fixes.py` - Debug orientation issues
- `mdm_to_hugs_pipeline.py` - Core conversion logic

Good luck! 🚀
