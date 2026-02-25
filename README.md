# INTACT: Integration of Trajectory Sets for Cell Tracking

INTACT is a Python tool for fusing multiple sets of trajectory (tracklet) data obtained from cell tracking. It addresses common scenarios where tracking results need to be combined across different time batches, spatial regions, or multiple views, while preserving trajectory continuity and reliability.

## Features

INTACT supports three main integration scenarios:

1. **Temporal Integration** – Fuse trajectory sets from different yet overlapping time batches. The algorithm leverages spatiotemporal information from adjacent frames to correctly link tracklets across time gaps or overlaps.
2. **Spatial Integration** – Fuse trajectory sets from overlapping spatial regions (e.g., tiles in a large image). This minimizes loss of tracking accuracy caused by cells at region boundaries and ensures seamless trajectories across the whole field of view.
3. **Multi‑view Stitching** – Fuse tracking results obtained from different views of the same specimen. By accounting for spatial alignment between views, INTACT maintains the integrity and reliability of trajectories as much as possible.

## Installation

```bash
pip install intact
```
(Provide actual installation instructions if the package is published.)

## Usage

The main function is `tracklets_integration` from the `integ` module. Below are typical calls for different scenarios.

### Parameters

| Parameter        | Description                                                                                                                                                                                                                                                                                                                                                             |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `csv_path1`      | Path to the first CSV file containing tracklet data.                                                                                                                                                                                                                                                                                                                   |
| `csv_path2`      | Path to the second CSV file containing tracklet data.                                                                                                                                                                                                                                                                                                                  |
| `seg_indice`     | Format of the `voxIdx` column in the CSV files. Options:<br>- `'matlab'`: indices are 1‑based (MATLAB style).<br>- `'python'`: indices are 0‑based (Python style).                                                                                                                                                                                                    |
| `output_path`    | Directory where the fused results will be saved.                                                                                                                                                                                                                                                                                                                       |
| `resolution`     | Voxel resolution in the order (z, y, x) for 3D data, or (y, x) for 2D data. For example, if the z‑step is 5 times larger than the xy pixel size, use `[5, 1, 1]`.                                                                                                                                                                                                     |
| `img_shape`      | Dimensions of the combined image (z, y, x) or (y, x) depending on data dimensionality.                                                                                                                                                                                                                                                                                 |
| `img_shape1`     | (Only for spatial/multi‑view integration) Dimensions of the first sub‑block **before** alignment.                                                                                                                                                                                                                                                                     |
| `img_shape2`     | (Only for spatial/multi‑view integration) Dimensions of the second sub‑block **before** alignment.                                                                                                                                                                                                                                                                    |
| `spatial_shift`  | Translation offsets needed to bring the two blocks into a common coordinate system. Format: `[[z_off1, y_off1, x_off1], [z_off2, y_off2, x_off2]]` (or 2D equivalent). For temporal integration this is typically `[[0,0],[0,0]]` and can be omitted.                                                                                                                |
| `temporal_shift` | Frame offsets needed to align the temporal axes. Format: `[shift1, shift2]`. For spatial integration this is typically `[0,0]` and can be omitted.                                                                                                                                                                                                                     |
| `mode`           | Integration mode:<br>- `'temporal'`: temporal integration (batches overlapping in time).<br>- `'spatial'`: spatial integration or multi‑view stitching (blocks overlapping in space).<br>- `'simple'`: simple fusion of two results that are already in the same coordinate system (both spatially and temporally).                                                 |
| `solver`         | Optimization solver to use. Options: `'pulp'`, `'ortools'`, `'gurobi'`, `'bb'`. Note that `'gurobi'` requires a valid license.                                                                                                                                                                                                                                         |
| `saveSeg`        | Boolean. If `True`, the output CSV will include the `voxIdx` column (segmentation indices). If `False`, this column is omitted to save space.                                                                                                                                                                                                                         |

### CSV Format Requirements

The input CSV files **must** contain the following columns:
- `ID`: unique identifier of the tracklet (integer)
- `frames`: frame number (integer, 0‑based or 1‑based, but must be consistent with `temporal_shift`)
- `x`, `y`, `z`: coordinates of the cell (floats). For 2D data, `z` can be omitted or set to 0.
- `parents`: parent information (typically an integer, may be 0 for no parent)
- `voxIdx`: segmentation index of the cell (format controlled by `seg_indice`)

## Dependencies

- Python 3.7+
- NumPy
- pandas
- PuLP (optional, if using `pulp` solver)
- ortools (optional, if using `ortools` solver)
- gurobipy (optional, if using `gurobi` solver)
