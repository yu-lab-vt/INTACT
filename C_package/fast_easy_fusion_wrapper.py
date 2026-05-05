import os
import sys
import importlib
import numpy as np
from scipy import stats


def _ensure_c_package_on_path():
    """
    Make importing fast_easy_fusion more robust.

    Supported layouts:
    1. wrapper.py and fast_easy_fusion.* are in the same folder.
    2. wrapper.py is in project root, and fast_easy_fusion.* is in ./C_package.
    3. wrapper.py is inside C_package.
    """
    if "__file__" in globals():
        base_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        base_dir = os.getcwd()

    candidates = [
        base_dir,
        os.path.join(base_dir, "C_package"),
        os.path.join(os.path.dirname(base_dir), "C_package"),
    ]

    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


_ensure_c_package_on_path()


try:
    _fast_easy_fusion = importlib.import_module("fast_easy_fusion")
except ImportError as e:
    raise ImportError(
        "Failed to import fast_easy_fusion. Please make sure the compiled "
        "C++ extension fast_easy_fusion is available in the current directory "
        "or in C_package."
    ) from e


if not hasattr(_fast_easy_fusion, "solve_easy_fusion_cpp"):
    raise ImportError(
        "fast_easy_fusion was imported, but solve_easy_fusion_cpp was not found. "
        "Please check whether the C++ extension was compiled with this binding."
    )

solve_easy_fusion_cpp = _fast_easy_fusion.solve_easy_fusion_cpp
build_easy_arcs_debug_cpp = getattr(_fast_easy_fusion, "build_easy_arcs_debug_cpp", None)


def _as_int64_1d(x, name):
    arr = np.asarray(x, dtype=np.int64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return np.ascontiguousarray(arr, dtype=np.int64)


def _normalize_matches(matches, n1=None, n2=None):
    """
    Convert matches to a contiguous int64 array with shape (K, 2).

    Accepted inputs:
    - None
    - []
    - shape (K, 2)
    - shape (K, 3), where the third column is ignored
    - shape (2,) or (3,), interpreted as one match
    """
    if matches is None:
        matches_ij = np.empty((0, 2), dtype=np.int64)
        return matches_ij

    arr = np.asarray(matches)

    if arr.size == 0:
        matches_ij = np.empty((0, 2), dtype=np.int64)
        return matches_ij

    if arr.ndim == 1:
        if arr.size < 2:
            raise ValueError(
                f"matches must contain at least two columns/values, got shape {arr.shape}."
            )
        arr = arr.reshape(1, -1)

    if arr.ndim != 2:
        raise ValueError(f"matches must be a 2D array, got shape {arr.shape}.")

    if arr.shape[1] < 2:
        raise ValueError(
            f"matches must have at least 2 columns: id1, id2. Got shape {arr.shape}."
        )

    matches_ij = np.ascontiguousarray(arr[:, :2], dtype=np.int64)

    if n1 is not None and n2 is not None and matches_ij.shape[0] > 0:
        bad1 = (matches_ij[:, 0] < 0) | (matches_ij[:, 0] >= n1)
        bad2 = (matches_ij[:, 1] < 0) | (matches_ij[:, 1] >= n2)

        if np.any(bad1) or np.any(bad2):
            bad_rows = np.where(bad1 | bad2)[0]
            preview = matches_ij[bad_rows[:10]]
            raise IndexError(
                "matches contain out-of-bound ids. "
                f"n1={n1}, n2={n2}, bad_rows_preview={bad_rows[:10].tolist()}, "
                f"bad_matches_preview={preview.tolist()}"
            )

    return matches_ij


def _infer_voxel_dim(vox_all):
    """
    Infer voxel dimension from the first non-empty voxel array.
    """
    for i, v in enumerate(vox_all):
        arr = np.asarray(v)

        if arr.ndim == 2:
            return int(arr.shape[1])

        if arr.ndim == 1 and arr.size == 0:
            continue

        raise ValueError(
            f"Invalid voxel array at index {i}. Expected shape (num_voxels, dim), "
            f"or an empty array. Got shape {arr.shape}."
        )

    raise ValueError(
        "Cannot infer voxel dimension because all voxel arrays are empty."
    )


def pack_voxels(movieInfo1_partial, movieInfo2_partial):
    """
    Pack all voxel coordinates into a flat contiguous int32 array.

    Returns
    -------
    vox_flat : np.ndarray, shape (total_num_voxels, dim), int32
    offsets : np.ndarray, shape (num_cells + 1,), int64
        voxels of cell i are vox_flat[offsets[i]:offsets[i+1]]
    dim : int
        voxel coordinate dimension, usually 2 or 3
    """
    if "vox" not in movieInfo1_partial:
        raise KeyError("movieInfo1_partial must contain key 'vox'.")
    if "vox" not in movieInfo2_partial:
        raise KeyError("movieInfo2_partial must contain key 'vox'.")

    vox_all = list(movieInfo1_partial["vox"]) + list(movieInfo2_partial["vox"])

    if len(vox_all) == 0:
        raise ValueError("No voxels found.")

    dim = _infer_voxel_dim(vox_all)

    vox_norm = []
    lengths = np.empty(len(vox_all), dtype=np.int64)

    for i, v in enumerate(vox_all):
        arr = np.asarray(v)

        if arr.ndim == 1 and arr.size == 0:
            arr = np.empty((0, dim), dtype=np.int32)
        elif arr.ndim == 2:
            if arr.shape[1] != dim:
                raise ValueError(
                    f"All voxel arrays must have the same dimension. "
                    f"Expected dim={dim}, but voxel array {i} has shape {arr.shape}."
                )
            arr = np.ascontiguousarray(arr, dtype=np.int32)
        else:
            raise ValueError(
                f"Invalid voxel array at index {i}. Expected shape (num_voxels, {dim}), "
                f"got shape {arr.shape}."
            )

        vox_norm.append(arr)
        lengths[i] = arr.shape[0]

    offsets = np.empty(len(vox_norm) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])

    total_voxels = int(offsets[-1])
    vox_flat = np.empty((total_voxels, dim), dtype=np.int32)

    pos = 0
    for arr in vox_norm:
        n = arr.shape[0]
        if n > 0:
            vox_flat[pos:pos + n] = arr
        pos += n

    return (
        np.ascontiguousarray(vox_flat, dtype=np.int32),
        np.ascontiguousarray(offsets, dtype=np.int64),
        int(dim),
    )


def _prepare_common_inputs(movieInfo1_partial, movieInfo2_partial, matches):
    """
    Shared input preparation for solve and debug wrappers.
    """
    required_keys = ["frames", "parents", "vox"]
    for key in required_keys:
        if key not in movieInfo1_partial:
            raise KeyError(f"movieInfo1_partial must contain key '{key}'.")
        if key not in movieInfo2_partial:
            raise KeyError(f"movieInfo2_partial must contain key '{key}'.")

    frames1 = _as_int64_1d(movieInfo1_partial["frames"], "frames1")
    parents1 = _as_int64_1d(movieInfo1_partial["parents"], "parents1")
    frames2 = _as_int64_1d(movieInfo2_partial["frames"], "frames2")
    parents2 = _as_int64_1d(movieInfo2_partial["parents"], "parents2")

    n1 = len(frames1)
    n2 = len(frames2)
    N = n1 + n2

    if len(parents1) != n1:
        raise ValueError(
            f"movieInfo1_partial['parents'] length must match frames length. "
            f"Got len(parents1)={len(parents1)}, len(frames1)={n1}."
        )

    if len(parents2) != n2:
        raise ValueError(
            f"movieInfo2_partial['parents'] length must match frames length. "
            f"Got len(parents2)={len(parents2)}, len(frames2)={n2}."
        )

    if N <= 0:
        raise ValueError("Total number of cells is zero.")

    if len(movieInfo1_partial["vox"]) != n1:
        raise ValueError(
            f"movieInfo1_partial['vox'] length must match frames length. "
            f"Got len(vox)={len(movieInfo1_partial['vox'])}, len(frames)={n1}."
        )

    if len(movieInfo2_partial["vox"]) != n2:
        raise ValueError(
            f"movieInfo2_partial['vox'] length must match frames length. "
            f"Got len(vox)={len(movieInfo2_partial['vox'])}, len(frames)={n2}."
        )

    matches_ij = _normalize_matches(matches, n1=n1, n2=n2)

    vox_flat, vox_offsets, dim = pack_voxels(movieInfo1_partial, movieInfo2_partial)

    threshold = float(stats.chi2.ppf(1.0 - 0.01 / N, 1) / 2.0)

    return (
        frames1,
        parents1,
        frames2,
        parents2,
        matches_ij,
        vox_flat,
        vox_offsets,
        dim,
        threshold,
    )


def solve_easy_fusion_cpp_wrapper(movieInfo1_partial, movieInfo2_partial, matches):
    """
    Python wrapper for solve_easy_fusion_cpp.

    Returns
    -------
    tracks_easy : list[np.ndarray]
    hard_subgraphs : object returned by C++ binding
    edges : object returned by C++ binding
    """
    (
        frames1,
        parents1,
        frames2,
        parents2,
        matches_ij,
        vox_flat,
        vox_offsets,
        dim,
        threshold,
    ) = _prepare_common_inputs(movieInfo1_partial, movieInfo2_partial, matches)

    tracks_easy, hard_subgraphs, edges = solve_easy_fusion_cpp(
        frames1,
        parents1,
        frames2,
        parents2,
        matches_ij,
        vox_flat,
        vox_offsets,
        int(dim),
        float(threshold),
    )

    tracks_easy = [np.asarray(t, dtype=np.int64) for t in tracks_easy]

    return tracks_easy, hard_subgraphs, edges


def build_easy_arcs_debug_cpp_wrapper(movieInfo1_partial, movieInfo2_partial, matches):
    """
    Python wrapper for build_easy_arcs_debug_cpp.

    This requires the C++ extension to expose build_easy_arcs_debug_cpp.
    """
    if build_easy_arcs_debug_cpp is None:
        raise ImportError(
            "fast_easy_fusion was imported, but build_easy_arcs_debug_cpp was not found. "
            "Please check whether the C++ extension was compiled with the debug binding."
        )

    (
        frames1,
        parents1,
        frames2,
        parents2,
        matches_ij,
        vox_flat,
        vox_offsets,
        dim,
        threshold,
    ) = _prepare_common_inputs(movieInfo1_partial, movieInfo2_partial, matches)

    detection_arcs, transition_arcs, easy_subgraphs, hard_subgraphs, edges = (
        build_easy_arcs_debug_cpp(
            frames1,
            parents1,
            frames2,
            parents2,
            matches_ij,
            vox_flat,
            vox_offsets,
            int(dim),
            float(threshold),
        )
    )

    detection_arcs = np.asarray(detection_arcs)
    transition_arcs = np.asarray(transition_arcs)

    return (
        detection_arcs,
        transition_arcs,
        easy_subgraphs,
        hard_subgraphs,
        edges,
    )