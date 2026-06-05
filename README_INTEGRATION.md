# 🎬 MDM to HUGS: Complete Integration Guide

## Overview

You have two powerful projects:
1. **Motion Diffusion Model (MDM)** - Generates realistic motions from text
2. **ML-HUGS** - Renders those motions as animated 3D avatars

This guide connects them into one pipeline.

## The Core Problem: Coordinate System Mismatch

Your upside-down avatar happened because:

```
MDM Output (Y-up)    →  Convert (missing step!)  →  HUGS Input (Z-up)
                                                     ❌ Upside-down!
```

**Solution**: Apply coordinate transformation during conversion

```
MDM Output (Y-up)    →  Convert + Rotate (90°X)  →  HUGS Input (Z-up)
                                                     ✓ Correct!
```

## What We Built For You

### 1. **Main Conversion Script** (`mdm_to_hugs_pipeline.py`)

Converts MDM output to HUGS format with coordinate fixing:

```bash
python mdm_to_hugs_pipeline.py \
    --mdm_results /path/to/results.npy \
    --output /path/to/output.npz \
    --rotate_x 90.0          # ← Key parameter to fix upside-down!
```

### 2. **Test Script** (`test_coordinate_fixes.py`)

Automatically tests 8 different coordinate transformations:

```bash
python test_coordinate_fixes.py \
    --mdm_results /path/to/results.npy \
    --output_dir ./test_variants
```

Creates:
- `original.npz` - No transformation
- `90deg_x.npz` - 90° rotation around X (most common!)
- `180deg_x.npz` - 180° rotation
- `neg90deg_x.npz` - -90° rotation
- Plus Y and Z variants

### 3. **Helper Script** (`mdm_to_hugs.sh`)

Convenient wrapper for common tasks:

```bash
./mdm_to_hugs.sh test-all /path/to/results.npy
./mdm_to_hugs.sh convert /path/to/results.npy
./mdm_to_hugs.sh convert-custom /path/to/results.npy 90 0 0
```

---

## Step-by-Step Usage

### Phase 1: Generate Motion with MDM

```bash
cd /home/sigma/skibidi/motion-diffusion-model
# Your existing MDM generation command
# Output: save/.../results.npy
```

### Phase 2: Convert to HUGS Format

**Option A: Quick (if you know the right rotation)**
```bash
cd /home/sigma/skibidi/motion-diffusion-model

python mdm_to_hugs_pipeline.py \
    --mdm_results save/humanml_enc_512_50steps/samples_.../results.npy \
    --output save/humanml_enc_512_50steps/samples_.../hugs_motion.npz \
    --rotate_x 90.0  # Most common fix!
```

**Option B: Thorough (test all variations - RECOMMENDED FIRST TIME)**
```bash
cd /home/sigma/skibidi/motion-diffusion-model

python test_coordinate_fixes.py \
    --mdm_results save/humanml_enc_512_50steps/samples_.../results.npy \
    --output_dir ./test_conversions
```

### Phase 3: Test in HUGS

```bash
cd /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar

# Test the most likely variant first
python main.py --motion_file /home/sigma/skibidi/motion-diffusion-model/test_conversions/90deg_x.npz

# Check: output/human_scene/neuman/[model]/[config]/[date]/anim_neuman_*.mp4
# Is avatar upright? Is motion natural? If yes, use this variant!

# If not, test others:
python main.py --motion_file /home/sigma/skibidi/motion-diffusion-model/test_conversions/180deg_x.npz
# etc.
```

### Phase 4: Production Render

Once you know the correct rotation:

```bash
cd /home/sigma/skibidi/motion-diffusion-model

python mdm_to_hugs_pipeline.py \
    --mdm_results save/humanml_enc_512_50steps/samples_.../results.npy \
    --output save/humanml_enc_512_50steps/samples_.../hugs_motion_final.npz \
    --rotate_x 90.0  # (or whatever worked in Phase 3)
```

```bash
cd /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar

python main.py --motion_file /home/sigma/skibidi/motion-diffusion-model/save/humanml_enc_512_50steps/samples_.../hugs_motion_final.npz

# Output appears at: output/human_scene/neuman/[model]/[config]/[date]/anim_neuman_*.mp4
```

---

## Coordinate System Parameters

| Parameter | Effect | When to Try |
|-----------|--------|------------|
| `--rotate_x 90` | Swap Y↔Z (Y-up → Z-up) | **Try this first!** |
| `--rotate_x 180` | Flip Y and Z | If 90 is wrong |
| `--rotate_x -90` | Opposite of 90 | Alternative Z-up |
| `--rotate_y 90` | Swap X↔Z | Less common |
| `--rotate_z 90` | Rotate in XY plane | Rotation mismatch |

---

## File Locations

### Inputs (from MDM)
```
/home/sigma/skibidi/motion-diffusion-model/
  save/humanml_enc_512_50steps/
    samples_humanml_enc_512_50steps_000750000_seed10_a_person_jumps/
      results.npy  ← Input for conversion
```

### Intermediate (test variants)
```
/home/sigma/skibidi/motion-diffusion-model/
  test_conversions/
    original.npz, 90deg_x.npz, 180deg_x.npz, etc.
```

### Final Output (from HUGS)
```
/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/
  output/
    human_scene/
      neuman/
        [model]/[config]/[timestamp]/
          anim_neuman_[model]_[config]_final.mp4  ← Final output!
```

---

## Troubleshooting

### Avatar is Upside-Down
```bash
# Try these in order
--rotate_x 90.0
--rotate_x 180.0
--rotate_x -90.0
--rotate_y 90.0
```

### Avatar is Sideways/Rotated
```bash
# Indicates rotation axis is wrong, try:
--rotate_y 90.0  or  --rotate_z 90.0
```

### Motion Looks Scaled Wrong (Too Tall/Short)
```bash
# Current: Not implemented in these scripts
# Workaround: Adjust in HUGS config
# Contact if you need this feature
```

### Script Not Found / Import Errors
```bash
# Make sure you're in the right directory:
cd /home/sigma/skibidi/motion-diffusion-model

# Try with full path:
python ./mdm_to_hugs_pipeline.py --mdm_results ...
```

---

## Data Flow

```
MDM generates motion
        ↓
  results.npy (6D rotations + RIC encoding)
        ↓
test_coordinate_fixes.py tests 8 variants
        ↓
  *.npz files (8 versions with different rotations)
        ↓
test each in HUGS (Python main.py)
        ↓
identify correct rotation
        ↓
mdm_to_hugs_pipeline.py with correct params
        ↓
  hugs_motion_final.npz (SMPL format with correct orientation)
        ↓
HUGS renders with correct orientation
        ↓
  anim_neuman_*.mp4 (correct upright avatar!)
```

---

## Quick Reference

### Most Likely First Command (if avatar is upside-down):
```bash
cd /home/sigma/skibidi/motion-diffusion-model

python mdm_to_hugs_pipeline.py \
    --mdm_results results.npy \
    --output hugs_motion.npz \
    --rotate_x 90.0
```

### Test All Variants:
```bash
python test_coordinate_fixes.py \
    --mdm_results results.npy \
    --output_dir test_conversions
```

### Use in HUGS:
```bash
cd /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar
python main.py --motion_file /path/to/hugs_motion.npz
```

---

## Expected Results

**Before Integration:**
- MDM: Text → Motion ✓
- HUGS: Manual motion → Render ✓
- Pipeline: Broken (upside-down) ✗

**After Integration:**
- MDM: "a person jumps" → results.npy ✓
- Convert: results.npy → hugs_motion.npz ✓
- HUGS: hugs_motion.npz → Correct 3D animation ✓

---

## Files Created

1. **mdm_to_hugs_pipeline.py** (8.1 KB) - Main conversion tool
2. **test_coordinate_fixes.py** (4.0 KB) - Test all variants
3. **mdm_to_hugs.sh** (3.9 KB) - Bash wrapper
4. **MDM_TO_HUGS_PIPELINE.txt** - Detailed documentation
5. **QUICK_START.txt** - Quick reference
6. **README_INTEGRATION.md** - This file

---

## Next Steps

1. ✓ Files created in `/home/sigma/skibidi/motion-diffusion-model/`
2. Generate motion with MDM (you already know how)
3. Locate `results.npy` from that generation
4. Run `test_coordinate_fixes.py` to create 8 variants
5. Test each variant in HUGS
6. Note which one looks correct
7. Use that rotation parameter for final conversions
8. Enjoy your upright avatar animations! 🎉

---

## Notes

- **Coordinate transformation** is applied to all rotations and translations
- **All variants** are saved during testing for flexibility
- **Most common** fix is `--rotate_x 90.0` (Y-up to Z-up)
- **Test first** to avoid wasted renders
- **Once identified**, you can reuse the same rotation for all future MDM → HUGS conversions
