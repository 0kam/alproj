# Matching Methods Gallery

This page shows the matching results for all available methods. All images were generated using the same input pair (5616x3744 pixel photograph and simulated image) on CPU (M4 Pro MBP).

See [Usage](usage.md) for details on how to use each method.

## Built-in Methods (No Extra Dependencies)

### AKAZE
- **Time:** ~1 sec | **Matches:** 205

![](_static/matched_akaze.png)

### SIFT
- **Time:** ~3 sec | **Matches:** 333

![](_static/matched_sift.png)

## LightGlue-based Methods (Lightweight)

These methods can handle full-resolution images efficiently.

### SIFT-LightGlue
- **Time:** ~2 sec | **Matches:** 503

![](_static/matched_sift_lightglue.png)

### SuperPoint-LightGlue
- **Time:** ~3 sec | **Matches:** 965

![](_static/matched_superpoint_lightglue.png)

### MiniMa-SuperPoint-LightGlue
- **Time:** ~3 sec | **Matches:** 975

![](_static/matched_minima_superpoint_lightglue.png)

## Dense Matching Methods (High Match Count)

These methods automatically resize images to 640px when `resize` is not specified to prevent out-of-memory errors.

### Tiny-RoMa
- **Time:** ~2 sec | **Matches:** 2046

![](_static/matched_tiny_roma.png)

### LoFTR
- **Time:** ~3 sec | **Matches:** 2237

![](_static/matched_loftr.png)

### MiniMa-LoFTR
- **Time:** ~2 sec | **Matches:** 2632

![](_static/matched_minima_loftr.png)s

### RDD
- **Time:** ~3 sec | **Matches:** 1510

![](_static/matched_rdd.png)

### UFM
- **Time:** ~12 sec | **Matches:** 2048

![](_static/matched_ufm.png)

### MAST3R
- **Time:** ~17 sec | **Matches:** 3497

![](_static/matched_master.png)

### RoMa
- **Time:** ~21 sec | **Matches:** 2048

![](_static/matched_roma.png)

### MiniMa-RoMa
- **Time:** ~25 sec | **Matches:** 9999

![](_static/matched_minima_roma.png)
