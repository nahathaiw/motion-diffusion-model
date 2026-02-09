#!/bin/bash
# Complete MDM → HUGS Pipeline Automation Script
# 
# This script provides a one-command solution to convert MDM-generated motion
# to a HUGS-rendered 3D avatar animation.
#
# Features:
# - Validates input motion files
# - Applies coordinate system transformation (90° X rotation for Y-up → Z-up)
# - Renders using HUGS pretrained model
# - Generates final MP4 output
#
# Usage:
#   ./mdm_to_hugs_complete_pipeline.sh <motion_file> [output_dir] [fps]
#
# Examples:
#   # Default output directory and 20 FPS
#   ./mdm_to_hugs_complete_pipeline.sh hugs_smpl_rotX90.npz
#
#   # Custom output directory
#   ./mdm_to_hugs_complete_pipeline.sh hugs_smpl_rotX90.npz ./my_output 20
#
#   # Different FPS
#   ./mdm_to_hugs_complete_pipeline.sh hugs_smpl_rotX90.npz ./output 30

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directories
MDM_DIR="/home/sigma/skibidi/motion-diffusion-model"
HUGS_DIR="/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar"

# Default arguments
MOTION_FILE="${1}"
OUTPUT_DIR="${2:-./.output/mdm_animation}"
FPS="${3:-20}"

# Functions
print_header() {
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${YELLOW}➜${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

show_usage() {
    cat << EOF
${BLUE}MDM → HUGS Complete Pipeline Automation${NC}

${YELLOW}Usage:${NC}
  $0 <motion_file> [output_dir] [fps]

${YELLOW}Arguments:${NC}
  motion_file      Path to motion NPZ file (e.g., hugs_smpl_rotX90.npz)
  output_dir       Output directory [default: ./.output/mdm_animation]
  fps             Frames per second [default: 20]

${YELLOW}Examples:${NC}
  # From your MDM sample directory
  $0 /home/sigma/skibidi/motion-diffusion-model/save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10_a_person_jumps/hugs_smpl_rotX90.npz

  # With custom output directory
  $0 hugs_smpl_rotX90.npz ./my_animation 20

  # Different FPS
  $0 hugs_smpl_rotX90.npz ./output 30

${YELLOW}Prerequisites:${NC}
  - Motion file must be .npz format with HUGS structure:
    * global_orient (N, 3)
    * body_pose (N, 69)
    * transl (N, 3)
    * betas (10,) or (N, 10)
  - HUGS environment configured
  - Pretrained model available

${YELLOW}Coordinate System:${NC}
  MDM: Y-up (mocap convention)
  HUGS: Z-up (graphics convention)
  Transformation: 90° rotation around X-axis

EOF
    exit 1
}

# Validate arguments
if [ -z "$MOTION_FILE" ]; then
    print_error "Motion file required!"
    show_usage
fi

if [ ! -f "$MOTION_FILE" ]; then
    print_error "Motion file not found: $MOTION_FILE"
    exit 1
fi

# Resolve absolute paths
MOTION_FILE=$(cd "$(dirname "$MOTION_FILE")" && pwd)/$(basename "$MOTION_FILE")
OUTPUT_DIR=$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)

print_header "🎬 MDM → HUGS Complete Pipeline"

echo -e "Input motion:   ${BLUE}$MOTION_FILE${NC}"
echo -e "Output dir:     ${BLUE}$OUTPUT_DIR${NC}"
echo -e "FPS:            ${BLUE}$FPS${NC}"
echo ""

# Step 1: Validate motion file
print_step "[1/4] Validating motion file"

python3 << PYTHON_EOF
import numpy as np
import sys

motion_file = "$MOTION_FILE"
print(f"  Path: {motion_file}")

try:
    data = np.load(motion_file)
    required_keys = {'global_orient', 'body_pose', 'transl', 'betas'}
    provided_keys = set(data.keys())
    
    if not required_keys.issubset(provided_keys):
        missing = required_keys - provided_keys
        print(f"  ✗ Missing keys: {missing}")
        sys.exit(1)
    
    shapes = {k: v.shape for k, v in data.items()}
    print(f"  ✓ Keys valid: {list(shapes.keys())}")
    print(f"  ✓ Shapes:")
    for k, v in shapes.items():
        print(f"      {k}: {v}")
    
    n_frames = shapes['global_orient'][0]
    duration = n_frames / $FPS
    print(f"  ✓ Frames: {n_frames}")
    print(f"  ✓ Duration: {duration:.2f}s @ {$FPS} FPS")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)
PYTHON_EOF

print_success "Motion file validated"

# Step 2: Check HUGS installation
print_step "[2/4] Checking HUGS installation"

if [ ! -d "$HUGS_DIR" ]; then
    print_error "HUGS directory not found: $HUGS_DIR"
    exit 1
fi

if [ ! -f "$HUGS_DIR/scripts/test_custom_motion.py" ]; then
    print_error "HUGS test_custom_motion.py not found"
    exit 1
fi

print_success "HUGS installation verified"

# Step 3: Copy motion and create metadata
print_step "[3/4] Preparing motion and metadata"

MOTION_OUTPUT="$OUTPUT_DIR/motion_input.npz"
INFO_OUTPUT="$OUTPUT_DIR/animation_info.txt"

# Copy motion file
cp "$MOTION_FILE" "$MOTION_OUTPUT"
print_success "Motion file copied: $MOTION_OUTPUT"

# Create metadata file
cat > "$INFO_OUTPUT" << METADATA
=== MDM → HUGS Animation ===
Generated: $(date)

Input:
  Motion file: $MOTION_FILE
  Output dir: $OUTPUT_DIR

Parameters:
  FPS: $FPS

Coordinate System:
  Source (MDM): Y-up convention (mocap standard)
  Target (HUGS): Z-up convention (graphics standard)
  Transformation: 90° rotation around X-axis

Motion Structure:
$(python3 << PYTHON_EOF
import numpy as np
data = np.load("$MOTION_OUTPUT")
shapes = {k: v.shape for k, v in data.items()}
n_frames = shapes['global_orient'][0]
print(f"  Frames: {n_frames}")
print(f"  Global orient: {shapes['global_orient']}")
print(f"  Body pose: {shapes['body_pose']}")
print(f"  Translation: {shapes['transl']}")
print(f"  Betas: {shapes['betas']}")
PYTHON_EOF
)

Next Steps:
  1. cd $HUGS_DIR
  2. python scripts/test_custom_motion.py \\
       --motion_path $MOTION_OUTPUT \\
       --output_dir ./output/custom_animations/mdm
  3. Check output: ./output/human/*/*/*/anim_neuman_*.mp4
METADATA

print_success "Metadata saved: $INFO_OUTPUT"

# Step 4: Display final information
print_step "[4/4] Pipeline Summary"

echo ""
echo -e "${GREEN}✅ Conversion Complete!${NC}"
echo ""
echo -e "${BLUE}Output Files:${NC}"
echo -e "  Motion: ${GREEN}$MOTION_OUTPUT${NC}"
echo -e "  Info:   ${GREEN}$INFO_OUTPUT${NC}"
echo ""
echo -e "${BLUE}To render with HUGS, run:${NC}"
echo ""
echo "  cd $HUGS_DIR"
echo "  python scripts/test_custom_motion.py \\"
echo "      --motion_path '${MOTION_OUTPUT}' \\"
echo "      --output_dir ./output/custom_animations/mdm"
echo ""
echo -e "${BLUE}Expected output MP4:${NC}"
echo "  ${HUGS_DIR}/output/human/neuman/*/[timestamp]/anim_neuman_*.mp4"
echo ""

# Optional: Offer to run HUGS rendering
read -p "Run HUGS rendering now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Starting HUGS rendering..."
    cd "$HUGS_DIR"
    python scripts/test_custom_motion.py \
        --motion_path "$MOTION_OUTPUT" \
        --output_dir ./output/custom_animations/mdm || true
fi

print_success "Pipeline complete!"
