# Matching Methods Gallery

This page shows the matching results for all available methods. All images were generated using the same input pair (5616x3744 pixel photograph and simulated image) on CPU (M4 Pro MBP).

Any method supported by [vismatch](https://github.com/gmberton/vismatch) (70+ models) can be used. See [Usage](usage.md) for details.

## Built-in Methods (No Extra Dependencies)

### AKAZE
- **Time:** ~1 sec | **Matches:** 176

![](_static/matched_akaze.png)

### SIFT
- **Time:** ~4 sec | **Matches:** 331

![](_static/matched_sift.png)

## LightGlue-based Methods (Lightweight)

These methods can handle full-resolution images efficiently.

### SIFT-LightGlue
- **Time:** ~3 sec | **Matches:** 518

![](_static/matched_sift_lightglue.png)

### SuperPoint-LightGlue
- **Time:** ~4 sec | **Matches:** 962

![](_static/matched_superpoint_lightglue.png)

### ALIKED-LightGlue
- **Time:** ~6 sec | **Matches:** 815

![](_static/matched_aliked_lightglue.png)

### MiniMa-SuperPoint-LightGlue
- **Time:** ~29 sec | **Matches:** 973

![](_static/matched_minima_superpoint_lightglue.png)

## Dense Matching Methods (High Match Count)

These methods automatically resize images to 640px when `resize` is not specified to prevent out-of-memory errors.

### Tiny-RoMa
- **Time:** ~2 sec | **Matches:** 2047

![](_static/matched_tiny_roma.png)

### LoFTR
- **Time:** ~2 sec | **Matches:** 2198

![](_static/matched_loftr.png)

### MiniMa-LoFTR
- **Time:** ~3 sec | **Matches:** 2629

![](_static/matched_minima_loftr.png)

### UFM
- **Time:** ~10 sec | **Matches:** 2048

![](_static/matched_ufm.png)

### RDD
- **Time:** ~14 sec | **Matches:** 1494

![](_static/matched_rdd.png)

### RoMa
- **Time:** ~23 sec | **Matches:** 2048

![](_static/matched_roma.png)

### MAST3R
- **Time:** ~327 sec | **Matches:** 3515

![](_static/matched_master.png)

### MiniMa-RoMa
- **Requires CUDA** (large model). Use `minima-roma-tiny` for CPU.

![](_static/matched_minima_roma.png)

## Cross-Modal Methods (Robust to Appearance Changes)

These methods are designed for matching across different modalities, seasons, and resolutions.

### MatchAnything-ELoFTR
- **Time:** ~2 sec | **Matches:** 2547

![](_static/matched_matchanything_eloftr.png)

### MatchAnything-RoMa
- **Time:** ~23 sec | **Matches:** 4999

![](_static/matched_matchanything_roma.png)

### GIM-DKM
- **Time:** ~34 sec | **Matches:** 2048

![](_static/matched_gim_dkm.png)
