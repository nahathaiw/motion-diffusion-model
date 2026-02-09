# MDM → HUGS Pipeline: Visual Summary

## Complete Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MDM → HUGS ANIMATION PIPELINE                    │
└─────────────────────────────────────────────────────────────────────┘

STEP 1: Input
═════════════════════════════════════════════════════════════════════
  MDM Motion Output (Y-up convention)
  ↓
  /home/sigma/skibidi/motion-diffusion-model/
  save/humanml_enc_512_50steps/
    samples_humanml_enc_512_50steps_000750000_seed10_a_person_jumps/
    ├── results.npy                 (Original MDM output)
    ├── hugs_smpl_rotX90.npz        ✓ YOUR MAIN INPUT
    ├── hugs_smpl_rotX-90.npz       (Alternative)
    └── samples_00_to_00.mp4        (Reference video)


STEP 2: Coordinate Transform
═════════════════════════════════════════════════════════════════════
  
  Your Input File: hugs_smpl_rotX90.npz
  ├── Already has 90° X-axis rotation applied ✓
  │   (Transforms from Y-up to Z-up)
  │
  ├── SMPL Parameters Structure:
  │   ├── global_orient: (120, 3)     [root rotation]
  │   ├── body_pose: (120, 69)        [23 joints × 3 values]
  │   ├── transl: (120, 3)            [root position]
  │   └── betas: (10,)                [body shape]
  │
  └── Ready for HUGS ✓


STEP 3: Validation & Preparation
═════════════════════════════════════════════════════════════════════
  
  Run validation:
  ┌─────────────────────────────────────────┐
  │ python mdm_to_hugs_render_pipeline.py   │
  │   --motion_file hugs_smpl_rotX90.npz   │
  │   --output_dir ./output                 │
  │   --fps 20                              │
  └─────────────────────────────────────────┘
  
  Output directory structure:
  ./output/
  ├── motion_input.npz        (Copy of validated motion)
  └── animation_info.txt      (Metadata)


STEP 4: HUGS Rendering
═════════════════════════════════════════════════════════════════════
  
  Switch to HUGS directory:
  ┌──────────────────────────────────────────────────┐
  │ cd /home/sigma/project_avatar2_hugs/ml-hugs     │
  │                                                   │
  │ python scripts/test_custom_motion.py \          │
  │   --motion_path ./output/motion_input.npz \     │
  │   --output_dir ./output/custom_animations/mdm   │
  └──────────────────────────────────────────────────┘
  
  ✓ HUGS processes motion with pretrained model
  ✓ Generates 3D avatar keyframes
  ✓ Renders animation sequence


STEP 5: Output
═════════════════════════════════════════════════════════════════════
  
  Final MP4 Location:
  
  /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/
    output/human/neuman/[MODEL]/[CONFIG]/[TIMESTAMP]/
    └── anim_neuman_[MODEL]_final.mp4  ✓ YOUR ANIMATION!
  
  Example:
  ./output/human/neuman/bike/hugs_trimlp/2026-02-09_14-30-48/
  └── anim_neuman_bike_final.mp4


═════════════════════════════════════════════════════════════════════
```

## Coordinate System Transformation

```
    MDM (Y-Up)              →           HUGS (Z-Up)
    ──────────                          ──────────
    
        Y (Up)                              Z (Up)
        │                                   │
        │ Z (Forward)                       │ Y (Right)
        │/                                  │/
    ────┼──── X (Right)              ────┼──── X (Forward)
    
    
    Transformation: 90° Rotation around X-axis
    
    [x']   [1    0    0] [x]
    [y'] = [0    0   -1] [y]
    [z']   [0    1    0] [z]
    
    
    Result:
    ───────
    ✓ Avatar stands upright
    ✓ Motions look natural
    ✓ Same body shape (SMPL)
    ✓ Smooth animation flow
```

## Files Created for You

### 1. Documentation
- **MDM_TO_HUGS_COMPLETE_PIPELINE.md** (517 lines)
  - Complete reference guide
  - Step-by-step instructions
  - Troubleshooting section
  - Advanced usage examples
  - Coordinate system deep dive

### 2. Python Script
- **mdm_to_hugs_render_pipeline.py** (313 lines)
  - Validates motion files
  - Handles coordinate transformations
  - Supports both .npz and .npy inputs
  - Creates animation metadata
  - Ready for HUGS rendering

### 3. Bash Script
- **mdm_to_hugs_complete_pipeline.sh** (263 lines)
  - One-command automation
  - Interactive pipeline
  - Color-coded output
  - Automatic validation
  - Optional HUGS integration

## Quick Start Commands

### Fastest Way (All-in-One)
```bash
cd /home/sigma/skibidi/motion-diffusion-model

bash mdm_to_hugs_complete_pipeline.sh \
    ./save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10_a_person_jumps/hugs_smpl_rotX90.npz \
    ./my_animation
```

### Step-by-Step
```bash
# Step 1: Validate and prepare
cd /home/sigma/skibidi/motion-diffusion-model
python mdm_to_hugs_render_pipeline.py \
    --motion_file ./save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10_a_person_jumps/hugs_smpl_rotX90.npz \
    --output_dir ./motion_ready

# Step 2: Render with HUGS
cd /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar
python scripts/test_custom_motion.py \
    --motion_path /home/sigma/skibidi/motion-diffusion-model/motion_ready/motion_input.npz

# Step 3: Find output
find ./output -name "anim_neuman_*" -type f
```

## Testing Your Setup

### Verify Motion File Quality
```bash
python << 'EOF'
import numpy as np

motion_file = "./save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10_a_person_jumps/hugs_smpl_rotX90.npz"
data = np.load(motion_file)

print("✓ Motion file structure:")
for key, value in data.items():
    print(f"  {key}: {value.shape}")

n_frames = data['global_orient'].shape[0]
print(f"\n✓ Animation: {n_frames} frames @ 20 FPS = {n_frames/20:.1f}s")
EOF
```

### Check Output Location
```bash
# See where HUGS renders go
ls -lh /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/output/human/neuman/
```

## Troubleshooting Quick Reference

| Problem | Cause | Solution |
|---------|-------|----------|
| Avatar upside-down | Wrong rotation | Try `--rotate_x 180.0` |
| Avatar sideways | Y-Z swap issue | Try `--rotate_y 90.0` |
| Motion jumpy | FPS mismatch | Adjust `--fps` parameter |
| File not found | Relative path | Use absolute `/home/...` paths |
| CUDA error | Memory issue | Reduce resolution in config |

## Coordinate System Files for Reference

- **COORDINATE_SYSTEMS.md** - Visual coordinate system guide
- **MDM_TO_HUGS_PIPELINE.txt** - Original implementation notes
- **test_coordinate_fixes.py** - Generate 8 rotation variants for testing

## Your Sample Data Status

```
✓ Input:    hugs_smpl_rotX90.npz (120 frames, 6 seconds)
✓ Format:   HUGS-compatible SMPL parameters
✓ Rotation: 90° X-axis (Y-up → Z-up conversion)
✓ Ready:    Can be immediately passed to HUGS rendering
```

## Next Steps

1. **Read** the complete guide:
   ```
   cat MDM_TO_HUGS_COMPLETE_PIPELINE.md
   ```

2. **Run the pipeline**:
   ```bash
   bash mdm_to_hugs_complete_pipeline.sh hugs_smpl_rotX90.npz ./output
   ```

3. **Render with HUGS**:
   ```bash
   cd /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar
   python scripts/test_custom_motion.py --motion_path /path/to/motion_input.npz
   ```

4. **Check output**:
   ```bash
   find ./output -name "*anim_neuman*" -type f
   ```

---

**All tools are ready to use. Start with the shell script for easiest execution!** 🚀
