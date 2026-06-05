#!/bin/bash
# Quick reference commands for MDM → HUGS pipeline

set -e

# Configuration
MDM_DIR="/home/sigma/skibidi/motion-diffusion-model"
HUGS_DIR="/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar"

# Function to show usage
usage() {
    cat << EOF
MDM to HUGS Pipeline Helper

Usage:
    $0 <command> [options]

Commands:
    test-all <results.npy>      - Test all coordinate transformations
    convert <results.npy>       - Convert with default rotation (90deg X)
    convert-custom <results.npy> <rx> <ry> <rz>  - Convert with custom rotation

Examples:
    # Test all transformations (recommended first time)
    $0 test-all /path/to/results.npy

    # Convert with 90° X rotation (most common fix)
    $0 convert /path/to/results.npy

    # Convert with custom rotation
    $0 convert-custom /path/to/results.npy 90 0 0

    # Convert with -90° X rotation
    $0 convert-custom /path/to/results.npy -90 0 0

For detailed documentation, see MDM_TO_HUGS_PIPELINE.txt
EOF
    exit 1
}

# Command: test-all
cmd_test_all() {
    local mdm_results="$1"
    if [ -z "$mdm_results" ] || [ ! -f "$mdm_results" ]; then
        echo "Error: File not found: $mdm_results"
        exit 1
    fi
    
    local output_dir=$(dirname "$mdm_results")/test_conversions
    echo "Testing all coordinate transformations..."
    echo "Input: $mdm_results"
    echo "Output dir: $output_dir"
    
    cd "$MDM_DIR"
    python test_coordinate_fixes.py \
        --mdm_results "$mdm_results" \
        --output_dir "$output_dir"
    
    echo ""
    echo "✓ Test files created in: $output_dir"
    echo "Next: Run each variant through HUGS and check which orientation is correct"
    echo ""
    echo "  cd $HUGS_DIR"
    echo "  python main.py --motion_file $output_dir/90deg_x.npz"
    echo "  python main.py --motion_file $output_dir/180deg_x.npz"
    echo "  # etc."
}

# Command: convert (default)
cmd_convert() {
    local mdm_results="$1"
    if [ -z "$mdm_results" ] || [ ! -f "$mdm_results" ]; then
        echo "Error: File not found: $mdm_results"
        exit 1
    fi
    
    local output_file=$(dirname "$mdm_results")/hugs_motion.npz
    echo "Converting with default 90° X rotation..."
    echo "Input: $mdm_results"
    echo "Output: $output_file"
    
    cd "$MDM_DIR"
    python mdm_to_hugs_pipeline.py \
        --mdm_results "$mdm_results" \
        --output "$output_file" \
        --rotate_x 90.0
    
    echo ""
    echo "✓ Converted file: $output_file"
    echo "Use in HUGS:"
    echo "  cd $HUGS_DIR"
    echo "  python main.py --motion_file $output_file"
}

# Command: convert-custom
cmd_convert_custom() {
    local mdm_results="$1"
    local rx="$2"
    local ry="$3"
    local rz="$4"
    
    if [ -z "$mdm_results" ] || [ ! -f "$mdm_results" ]; then
        echo "Error: File not found: $mdm_results"
        exit 1
    fi
    
    if [ -z "$rx" ] || [ -z "$ry" ] || [ -z "$rz" ]; then
        echo "Error: Must provide rotation angles (rx ry rz)"
        exit 1
    fi
    
    local output_file=$(dirname "$mdm_results")/hugs_motion_custom_${rx}_${ry}_${rz}.npz
    echo "Converting with custom rotation..."
    echo "Input: $mdm_results"
    echo "Rotation: X=${rx}°, Y=${ry}°, Z=${rz}°"
    echo "Output: $output_file"
    
    cd "$MDM_DIR"
    python mdm_to_hugs_pipeline.py \
        --mdm_results "$mdm_results" \
        --output "$output_file" \
        --rotate_x "$rx" \
        --rotate_y "$ry" \
        --rotate_z "$rz"
    
    echo ""
    echo "✓ Converted file: $output_file"
    echo "Use in HUGS:"
    echo "  cd $HUGS_DIR"
    echo "  python main.py --motion_file $output_file"
}

# Main
if [ $# -lt 1 ]; then
    usage
fi

cmd="$1"
shift

case "$cmd" in
    test-all)
        cmd_test_all "$@"
        ;;
    convert)
        cmd_convert "$@"
        ;;
    convert-custom)
        cmd_convert_custom "$@"
        ;;
    *)
        echo "Unknown command: $cmd"
        usage
        ;;
esac
