# FEATURE385 Compare6/B1vnext Feature Schema Specification

> Version: Compare6/B1vnext canonical schema  
> Purpose: This document is the mandatory reference for any Codex-generated code involving FEATURE385 construction, splitting, packing, debugging, Adapter inference, or robot neutral feature building.

---

## 1. Core Principle

**FEATURE385 must always follow the Compare6/B1vnext training-time column order.**

Do **not** assume:

```text
abs_000 == feat_000
abs_001 == feat_001
...
```

This is wrong unless explicitly proven by the training columns file.

The only authoritative source for the 385-dimensional order is:

```text
FEATURE385_X2C_gpu.csv.gz.columns.json
```

Every feature must be placed according to:

```text
feature385[feat_idx] = row[source_column]
```

where `feat_idx` and `source_column` come from `FEATURE385_X2C_gpu.csv.gz.columns.json`.

---

## 2. Total Dimension

```text
FEATURE385 = 385 dimensions
```

The model input is divided into:

```text
ABS features   = 192 dims
REL features   = 190 dims
pose features  = 3 dims
Total          = 192 + 190 + 3 = 385
```

However, the final FEATURE385 vector is **not** stored as:

```text
[abs192, rel190, pose3]
```

The final FEATURE385 vector is stored in **region-interleaved order**:

```text
[brow_abs, brow_rel,
 eye_abs, eye_rel,
 mouth_abs, mouth_rel,
 jaw_abs, jaw_rel,
 global_abs, global_rel,
 pose]
```

---

## 3. Canonical FEATURE385 Layout

| Range | Dim | Region | Source | Meaning |
|---:|---:|---|---|---|
| 0:37 | 37 | brow | abs | brow absolute features |
| 37:74 | 37 | brow | rel | brow relative features |
| 74:119 | 45 | eye | abs | eye absolute features |
| 119:164 | 45 | eye | rel | eye relative features |
| 164:243 | 79 | mouth | abs | mouth absolute features |
| 243:322 | 79 | mouth | rel | mouth relative features |
| 322:350 | 28 | jaw | abs | jaw absolute features |
| 350:378 | 28 | jaw | rel | jaw relative features |
| 378:381 | 3 | global | abs | yaw, pitch, roll |
| 381:382 | 1 | global | rel | ENERGY_rel |
| 382:385 | 3 | pose | pose | yaw, pitch, roll |

### Mandatory model input split

Any inference script feeding B1vnext must split FEATURE385 exactly as:

```python
brow_abs   = f[0:37]
brow_rel   = f[37:74]
eye_abs    = f[74:119]
eye_rel    = f[119:164]
mouth_abs  = f[164:243]
mouth_rel  = f[243:322]
jaw_abs    = f[322:350]
jaw_rel    = f[350:378]
global_abs = f[378:381]
global_rel = f[381:382]
pose       = f[382:385]
```

---

## 4. Internal ABS192 and REL190 Layout

Some code may internally store ABS and REL separately. If so, the only allowed internal order is:

### ABS192 internal order

```text
abs192 = [brow_abs37, eye_abs45, mouth_abs79, jaw_abs28, global_abs3]
```

| Internal ABS range | Dim | Corresponding FEATURE385 range |
|---:|---:|---:|
| abs[0:37] | 37 | feat[0:37] |
| abs[37:82] | 45 | feat[74:119] |
| abs[82:161] | 79 | feat[164:243] |
| abs[161:189] | 28 | feat[322:350] |
| abs[189:192] | 3 | feat[378:381] |

### REL190 internal order

```text
rel190 = [brow_rel37, eye_rel45, mouth_rel79, jaw_rel28, global_rel1]
```

| Internal REL range | Dim | Corresponding FEATURE385 range |
|---:|---:|---:|
| rel[0:37] | 37 | feat[37:74] |
| rel[37:82] | 45 | feat[119:164] |
| rel[82:161] | 79 | feat[243:322] |
| rel[161:189] | 28 | feat[350:378] |
| rel[189:190] | 1 | feat[381:382] |

---

## 5. Detailed Feature Content by Region

### 5.1 Brow ABS: feat[0:37]

```text
feat_000 = AU1_abs_intensity
feat_001 = AU2_abs_intensity
feat_002 = AU4_abs_intensity
feat_003 ~ feat_032 = lmk_abs_norm_00_x/y/z ... lmk_abs_norm_09_x/y/z
feat_033 = brow_left_eye_dist
feat_034 = brow_right_eye_dist
feat_035 = brow_inner_dist
feat_036 = brow_outer_height_diff
```

### 5.2 Brow REL: feat[37:74]

```text
feat_037 = AU1_rel
feat_038 = AU2_rel
feat_039 = AU4_rel
feat_040 ~ feat_069 = lmk_rel_00_x/y/z ... lmk_rel_09_x/y/z
feat_070 = brow_left_eye_dist_rel
feat_071 = brow_right_eye_dist_rel
feat_072 = brow_inner_dist_rel
feat_073 = brow_outer_height_diff_rel
```

### 5.3 Eye ABS: feat[74:119]

```text
feat_074 = AU5_abs_intensity
feat_075 = AU6_abs_intensity
feat_076 = AU7_abs_intensity
feat_077 ~ feat_112 = lmk_abs_norm_10_x/y/z ... lmk_abs_norm_21_x/y/z
feat_113 = eye_left_open
feat_114 = eye_right_open
feat_115 = eye_left_width
feat_116 = eye_right_width
feat_117 = eye_left_ratio
feat_118 = eye_right_ratio
```

### 5.4 Eye REL: feat[119:164]

```text
feat_119 = AU5_rel
feat_120 = AU6_rel
feat_121 = AU7_rel
feat_122 ~ feat_157 = lmk_rel_10_x/y/z ... lmk_rel_21_x/y/z
feat_158 = eye_left_open_rel
feat_159 = eye_right_open_rel
feat_160 = eye_left_width_rel
feat_161 = eye_right_width_rel
feat_162 = eye_left_ratio_rel
feat_163 = eye_right_ratio_rel
```

### 5.5 Mouth ABS: feat[164:243]

```text
feat_164 = AU10_abs_intensity
feat_165 = AU12_abs_intensity
feat_166 = AU14_abs_intensity
feat_167 = AU15_abs_intensity
feat_168 = AU17_abs_intensity
feat_169 = AU20_abs_intensity
feat_170 = AU23_abs_intensity
feat_171 = AU25_abs_intensity
feat_172 ~ feat_231 = lmk_abs_norm_22_x/y/z ... lmk_abs_norm_41_x/y/z
feat_232 = mouth_width
feat_233 = mouth_open
feat_234 = mouth_left_corner_to_nose
feat_235 = mouth_right_corner_to_nose
feat_236 = mouth_left_corner_raise
feat_237 = mouth_right_corner_raise
feat_238 = upper_lip_to_lower_lip
feat_239 = upper_lip_to_nose
feat_240 = lower_lip_to_chin
feat_241 = mouth_center_to_nose
feat_242 = mouth_center_to_chin
```

### 5.6 Mouth REL: feat[243:322]

```text
feat_243 = AU10_rel
feat_244 = AU12_rel
feat_245 = AU14_rel
feat_246 = AU15_rel
feat_247 = AU17_rel
feat_248 = AU20_rel
feat_249 = AU23_rel
feat_250 = AU25_rel
feat_251 ~ feat_310 = lmk_rel_22_x/y/z ... lmk_rel_41_x/y/z
feat_311 = mouth_width_rel
feat_312 = mouth_open_rel
feat_313 = mouth_left_corner_to_nose_rel
feat_314 = mouth_right_corner_to_nose_rel
feat_315 = mouth_left_corner_raise_rel
feat_316 = mouth_right_corner_raise_rel
feat_317 = upper_lip_to_lower_lip_rel
feat_318 = upper_lip_to_nose_rel
feat_319 = lower_lip_to_chin_rel
feat_320 = mouth_center_to_nose_rel
feat_321 = mouth_center_to_chin_rel
```

### 5.7 Jaw ABS: feat[322:350]

```text
feat_322 = AU26_abs_intensity
feat_323 ~ feat_346 = lmk_abs_norm_42_x/y/z ... lmk_abs_norm_49_x/y/z
feat_347 = jaw_open
feat_348 = chin_to_nose
feat_349 = chin_to_upper_lip
```

### 5.8 Jaw REL: feat[350:378]

```text
feat_350 = AU26_rel
feat_351 ~ feat_374 = lmk_rel_42_x/y/z ... lmk_rel_49_x/y/z
feat_375 = jaw_open_rel
feat_376 = chin_to_nose_rel
feat_377 = chin_to_upper_lip_rel
```

### 5.9 Global ABS: feat[378:381]

```text
feat_378 = yaw
feat_379 = pitch
feat_380 = roll
```

### 5.10 Global REL: feat[381:382]

```text
feat_381 = ENERGY_rel
```

### 5.11 Pose: feat[382:385]

```text
feat_382 = yaw
feat_383 = pitch
feat_384 = roll
```

Note: yaw/pitch/roll appear twice:

1. `global_abs`: feat[378:381]
2. `pose`: feat[382:385]

This duplication is intentional and must not be removed unless the model and dataset are retrained.

---

## 6. Required Packing and Unpacking Functions

### 6.1 Pack internal ABS192 + REL190 + pose3 into FEATURE385

```python
import numpy as np


def pack_feature385_interleaved(abs_vec, rel_vec, pose_vec):
    a = np.asarray(abs_vec, dtype=np.float32).reshape(-1)
    r = np.asarray(rel_vec, dtype=np.float32).reshape(-1)
    p = np.asarray(pose_vec, dtype=np.float32).reshape(-1)

    if a.shape[0] != 192:
        raise ValueError(f"abs_vec must be 192 dims, got {a.shape[0]}")
    if r.shape[0] != 190:
        raise ValueError(f"rel_vec must be 190 dims, got {r.shape[0]}")
    if p.shape[0] != 3:
        raise ValueError(f"pose_vec must be 3 dims, got {p.shape[0]}")

    f = np.concatenate([
        a[0:37],      r[0:37],
        a[37:82],     r[37:82],
        a[82:161],    r[82:161],
        a[161:189],   r[161:189],
        a[189:192],   r[189:190],
        p,
    ], axis=0).astype(np.float32)

    if f.shape[0] != 385:
        raise ValueError(f"FEATURE385 must be 385 dims, got {f.shape[0]}")
    return f
```

### 6.2 Unpack FEATURE385 into internal ABS192 + REL190 + pose3

```python
import numpy as np


def unpack_feature385_interleaved(feature385):
    f = np.asarray(feature385, dtype=np.float32).reshape(-1)
    if f.shape[0] != 385:
        raise ValueError(f"FEATURE385 must be 385 dims, got {f.shape[0]}")

    brow_abs   = f[0:37]
    brow_rel   = f[37:74]
    eye_abs    = f[74:119]
    eye_rel    = f[119:164]
    mouth_abs  = f[164:243]
    mouth_rel  = f[243:322]
    jaw_abs    = f[322:350]
    jaw_rel    = f[350:378]
    global_abs = f[378:381]
    global_rel = f[381:382]
    pose       = f[382:385]

    abs_vec = np.concatenate([brow_abs, eye_abs, mouth_abs, jaw_abs, global_abs], axis=0).astype(np.float32)
    rel_vec = np.concatenate([brow_rel, eye_rel, mouth_rel, jaw_rel, global_rel], axis=0).astype(np.float32)

    return {
        "abs": abs_vec,
        "rel": rel_vec,
        "pose": pose.astype(np.float32),
        "feature385": f.astype(np.float32),
    }
```

---

## 7. Required Columns-JSON-Based Builder

Any new feature extraction code must build FEATURE385 using the training columns JSON.

### 7.1 Required builder behavior

```python
feature385[feat_idx] = row[source_column]
```

where `feat_idx` and `source_column` are read from:

```text
FEATURE385_X2C_gpu.csv.gz.columns.json
```

### 7.2 Required function signature

```python
def row_to_compare6_feature385(row: dict, columns: list[dict], rel_mode: str = "zero") -> dict:
    """Build FEATURE385 using columns.json source_column order.

    rel_mode:
        "zero"     -> fill all source == 'rel' columns with 0.0. Use for single neutral robot image.
        "from_row" -> read rel source_column from row. Use only when row contains real REL columns.
    """
```

### 7.3 Required implementation rule

```python
for item in columns:
    idx = int(item["feat_idx"])
    source = item.get("source")
    col = item["source_column"]

    if source == "rel" and rel_mode == "zero":
        feature385[idx] = 0.0
    else:
        feature385[idx] = safe_float(row[col])
```

Missing ABS or pose columns must be reported in debug output. Do not silently use `abs_000` as a fallback.

---

## 8. Rules for Robot Neutral Image Debug

When building FEATURE385 from a single robot neutral image such as `20000.jpg`:

```text
ABS columns  -> read from extracted row[source_column]
REL columns  -> fill with 0.0
pose columns -> read yaw/pitch/roll from extracted row
```

The result must satisfy:

```text
feat_000 = AU1_abs_intensity
feat_001 = AU2_abs_intensity
feat_002 = AU4_abs_intensity
...
feat_378 = yaw
feat_379 = pitch
feat_380 = roll
feat_381 = 0.0
feat_382 = yaw
feat_383 = pitch
feat_384 = roll
```

---

## 9. Rules for Human-to-Robot Adapter

The Adapter must not feed raw human ABS into B1vnext.

Correct logic:

```text
human neutral image + human expression image
→ human REL
→ robot-style REL190
→ robot-style ABS192 = robot_neutral_ABS192 + mapped REL delta
→ FEATURE385 = pack_feature385_interleaved(robot_style_ABS192, robot_style_REL190, pose3)
→ B1vnext
```

The Adapter's internal REL190 must follow the exact internal REL order:

```text
[brow_rel37, eye_rel45, mouth_rel79, jaw_rel28, global_rel1]
```

The final Adapter output must follow the exact FEATURE385 order:

```text
[brow_abs, brow_rel,
 eye_abs, eye_rel,
 mouth_abs, mouth_rel,
 jaw_abs, jaw_rel,
 global_abs, global_rel,
 pose]
```

---

## 10. REL-to-ABS Mapping Rule for Adapter

When constructing `robot_style_abs` from `robot_style_rel`:

```python
abs_delta = np.zeros(192, dtype=np.float32)
abs_delta[0:37] = rel[0:37]        # brow
abs_delta[37:82] = rel[37:82]      # eye
abs_delta[82:161] = rel[82:161]    # mouth
abs_delta[161:189] = rel[161:189]  # jaw
abs_delta[189:192] = 0.0           # global_abs remains neutral

robot_style_abs = robot_neutral_abs + abs_delta
```

Do not copy `global_rel` into `global_abs`.

---

## 11. Mandatory Validation Checks

Every script that builds FEATURE385 must check:

```python
assert feature385.shape == (385,)
assert brow_abs.shape == (37,)
assert brow_rel.shape == (37,)
assert eye_abs.shape == (45,)
assert eye_rel.shape == (45,)
assert mouth_abs.shape == (79,)
assert mouth_rel.shape == (79,)
assert jaw_abs.shape == (28,)
assert jaw_rel.shape == (28,)
assert global_abs.shape == (3,)
assert global_rel.shape == (1,)
assert pose.shape == (3,)
```

If internal ABS/REL vectors are used:

```python
assert abs_vec.shape == (192,)
assert rel_vec.shape == (190,)
assert pose_vec.shape == (3,)
```

---

## 12. Strict Prohibitions

Codex must not generate code that:

```text
1. Treats FEATURE385 as [abs192, rel190, pose3].
2. Assumes abs_000 == feat_000.
3. Places yaw/pitch/roll at feat_000/001/002.
4. Uses abs_000...abs_191 directly as FEATURE385 columns.
5. Silently ignores missing source_column fields.
6. Reorders FEATURE385 without updating model input split.
7. Removes duplicated yaw/pitch/roll from global_abs and pose.
8. Feeds raw human ABS directly into B1vnext.
9. Copies global_rel into global_abs.
10. Changes motor30 output order without retraining and documentation.
```

---

## 13. Required Debug Fields

Any debug output should include:

```json
{
  "feature_layout": "compare6_interleaved",
  "feature_build_method": "columns_json_source_column",
  "feature385_shape": [385],
  "missing_columns": [],
  "num_missing_columns": 0,
  "rel_columns_filled_zero": true,
  "feature_layout_check_passed": true
}
```

For Adapter debug:

```json
{
  "feature_layout": "compare6_interleaved",
  "feature385_pack_order": "brow_abs,brow_rel,eye_abs,eye_rel,mouth_abs,mouth_rel,jaw_abs,jaw_rel,global_abs,global_rel,pose",
  "rel_to_abs_mapping_used": true,
  "rel_to_abs_mapping_type": "default_regionwise_189_to_192",
  "global_abs_kept_neutral": true,
  "feature_layout_check_passed": true
}
```

---

## 14. Minimal Acceptance Test

Use `20000.jpg` as a sanity check.

Expected diagnosis:

```text
training original FEATURE385 -> B1vnext -> motor30 should closely match metadata ctrl_value.
newly extracted FEATURE385 -> B1vnext -> motor30 should also be close after correct columns-json-based construction.
```

If the training original prediction is accurate but newly extracted prediction is not, then:

```text
The current extraction/build process is not equivalent to training FEATURE385 construction.
```

If `feat_000`, `feat_001`, `feat_002` become yaw/pitch/roll, the build is wrong.

Correct:

```text
feat_000 = AU1_abs_intensity
feat_001 = AU2_abs_intensity
feat_002 = AU4_abs_intensity
```

---

## 15. One-Sentence Rule

**Always build FEATURE385 by `feat_idx` and `source_column` from `FEATURE385_X2C_gpu.csv.gz.columns.json`; never build it by guessing array position.**
