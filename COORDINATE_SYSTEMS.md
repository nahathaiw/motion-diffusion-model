# Coordinate System Visual Guide

## Understanding the Upside-Down Problem

### What MDM Uses (Y-up Convention)

```
        Y (Up)
        |
        |  Z (Forward)
        | /
        |/
--------+-------- X (Right)

Standard motion capture convention
Used by SMPL, AMASS, and most mocap systems
```

### What HUGS Expects (likely Z-up)

```
        Z (Up)
        |
        |  Y (Right)
        | /
        |/
--------+-------- X (Forward)

Common graphics convention
Z-up is typical in 3D graphics engines
```

### The Problem: Direct Mapping

```
MDM Y-axis (Up)     →  HUGS Z-axis (Up)     ← Mismatch!
MDM Z-axis (Forward) → HUGS Y-axis (Right)  ← Mismatch!
MDM X-axis (Right)   → HUGS X-axis (Forward) ← Mismatch!

Result: Avatar falls sideways or upside-down
```

## The Solution: 90° Rotation Around X-Axis

### Rotation Formula (Roll around X-axis by 90°)

```
[x']   [1    0   0] [x]
[y'] = [0   cos -sin] [y]    where θ = 90°
[z']   [0   sin  cos] [z]

Becomes:
[x']   [1   0   0] [x]       x' = x
[y'] = [0   0  -1] [y]   →   y' = -z
[z']   [0   1   0] [z]       z' = y
```

### Result After 90° X Rotation

```
MDM Y-axis (Up)      → -Z = Down? No wait...
                        Z' = Y ✓ (Up maps to Up)

MDM Z-axis (Forward) → Y' = -Z
                        X' = X ✓

So we get:
  MDM Y (Up)      → HUGS Z (Up)      ✓ Correct!
  MDM Z (Forward) → HUGS -Y (Left)   ~ Needs rotation
  MDM X (Right)   → HUGS X (Forward) ~ Needs rotation
```

## Visual Example: Avatar Orientation

### Before Transformation (Wrong - Upside Down)

```
MDM (Y-up):        HUGS Direct Map (Wrong):
    Head                Head
     |                   |
   Chest              Wrong axis!
   /   \              Foot points up
  Arm  Arm            /
    |   |           Body is upside-down ✗
   Leg  Leg
    |
   Foot
```

### After 90° X Rotation (Correct)

```
MDM (Y-up):        HUGS After Rotation (Correct):
    Head                Head (Z-axis, up) ✓
     |                   |
   Chest              Chest
   /   \              /   \
  Arm  Arm          Arm  Arm
    |   |             |   |
   Leg  Leg          Leg  Leg
    |                  |
   Foot (Y-up)        Foot (Y-up) ✓
```

## Rotation Options Explained

### 1. `--rotate_x 90.0` (Most Common)

```
Effect: Roll forward 90°
  Y → +Z ✓ (Up to Up)
  Z → -Y (Forward rotates)
  X → X

When: Avatar is upside-down or Z-axis inverted
```

### 2. `--rotate_x 180.0` (Flip)

```
Effect: Roll 180° (flip upside-down twice)
  Y → -Z (Up to Down)
  Z → -Y (Forward to Back)
  X → X

When: Opposite extreme, need full flip
```

### 3. `--rotate_x -90.0` (Opposite Roll)

```
Effect: Roll backward 90°
  Y → -Z (Up to Down)
  Z → +Y (Forward stays forward)
  X → X

When: Different Z-up convention
```

### 4. `--rotate_y 90.0` (Yaw)

```
Effect: Rotate 90° around Y-axis
  X → +Z
  Y → Y ✓ (Up stays up)
  Z → -X

When: Forward and right axes swapped
```

### 5. `--rotate_z 90.0` (Pitch)

```
Effect: Rotate 90° around Z-axis
  X → +Y
  Y → -X
  Z → Z ✓

When: X and Y axes swapped (rare)
```

## Joint Position Flow

### Example: Hip Joint Position

```
MDM (Y-up frame):       HUGS (Z-up frame):
Hip at (1, 2, 3)       Hip at (x', y', z')

After --rotate_x 90:
x' = 1 * cos(90) = 0... Actually:
x' = x
y' = z (forward becomes right)
z' = y (up becomes up)

Result: Hip at (1, 3, 2) ✓
```

## How the Scripts Handle This

### `test_coordinate_fixes.py`

Tests these transformations:
1. None (original)
2. Rotate X ±90°, 180°
3. Rotate Y 90°, 180°
4. Rotate Z 90°
5. Combinations

Each creates a separate `.npz` file for testing in HUGS.

### `mdm_to_hugs_pipeline.py`

Applies transformation to:
- All 24 joint rotations (6 features → 3D axis-angle)
- Root translation vector (3D position)

Equation used:
```
R_joint_new = R_fix @ R_joint_old @ R_fix^T
t_new = R_fix @ t_old
```

Where `R_fix` is the rotation matrix from your `--rotate_x/y/z` params.

## Debugging: How to Tell Which is Correct

When testing in HUGS, check:

### ✓ Correct Orientation (--rotate_x 90)
```
Avatar stands upright           ✓
Head above shoulders            ✓
Feet on ground (mostly)         ✓
Jump motion goes up/down        ✓
Gravity makes sense             ✓
```

### ✗ Wrong Orientation (no rotation)
```
Avatar upside-down              ✗
Feet above head                 ✗
Jump motion is sideways         ✗
Limbs twist unnaturally         ✗
```

### ✗ Wrong Rotation Axis (e.g., --rotate_y 90)
```
Avatar is rotated 90°           ✗
Head sideways                   ✗
Motion is rotated correctly but wrong axis ✗
Not quite upside-down, more sideways ✗
```

## Summary Table

| Symptom | Try First | Then Try | Then Try |
|---------|-----------|----------|----------|
| Upside-down | `--rotate_x 90` | `--rotate_x 180` | `--rotate_x -90` |
| Head sideways | `--rotate_y 90` | `--rotate_x 90` | `--rotate_z 90` |
| Forward/back swapped | `--rotate_z 90` | `--rotate_y 90` | None of above |

## Key Takeaway

For **Y-up (MDM) to Z-up (HUGS) conversion**:
- **90° rotation around X-axis** is almost always correct
- It maps Y→Z while preserving the up direction
- All rotations are applied consistently to joints and translations

If this doesn't work, the differences might be:
- Different order of transformations
- Different rotation conventions (intrinsic vs extrinsic)
- Something other than coordinate axis mismatch

Test systematically using `test_coordinate_fixes.py` to find the right one!
