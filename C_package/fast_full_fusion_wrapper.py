import numpy as np
from scipy import stats

from fast_full_fusion import solve_fusion_cpp


def pack_voxels(movieInfo1_partial, movieInfo2_partial):
    vox_all = list(movieInfo1_partial["vox"]) + list(movieInfo2_partial["vox"])

    if len(vox_all) == 0:
        raise ValueError("No voxels found.")

    dim = None
    for v in vox_all:
        v = np.asarray(v)
        if v.ndim == 2 and v.shape[0] > 0:
            dim = int(v.shape[1])
            break

    if dim is None:
        raise ValueError("Cannot infer voxel dimension.")

    lengths = np.asarray([len(np.asarray(v)) for v in vox_all], dtype=np.int64)

    offsets = np.empty(len(vox_all) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])

    vox_flat = np.empty((int(offsets[-1]), dim), dtype=np.int32)

    pos = 0
    for i, v in enumerate(vox_all):
        v = np.asarray(v, dtype=np.int32)

        if v.ndim != 2 or v.shape[1] != dim:
            raise ValueError(
                f"All vox arrays must have shape (n, {dim}), "
                f"but vox[{i}] has shape {v.shape}."
            )

        n = len(v)
        vox_flat[pos:pos + n] = v
        pos += n

    return (
        np.ascontiguousarray(vox_flat, dtype=np.int32),
        np.ascontiguousarray(offsets, dtype=np.int64),
        int(dim),
    )


def normalize_matches(matches):
    arr = np.asarray(matches)

    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"matches must have shape (M, >=2), got {arr.shape}")

    return np.ascontiguousarray(arr[:, :2], dtype=np.int64)


def solve_fusion_cpp_wrapper(
    movieInfo1_partial,
    movieInfo2_partial,
    matches,
    max_bb_nodes=1000000,
    batch_branch_size=2,
    use_initial_greedy=True,
    use_conflict_blocks=True,
    conflict_blocks_star_only=True,
):
    frames1 = np.ascontiguousarray(movieInfo1_partial["frames"], dtype=np.int64)
    parents1 = np.ascontiguousarray(movieInfo1_partial["parents"], dtype=np.int64)

    frames2 = np.ascontiguousarray(movieInfo2_partial["frames"], dtype=np.int64)
    parents2 = np.ascontiguousarray(movieInfo2_partial["parents"], dtype=np.int64)

    matches_ij = normalize_matches(matches)

    vox_flat, vox_offsets, dim = pack_voxels(
        movieInfo1_partial,
        movieInfo2_partial,
    )

    N = len(frames1) + len(frames2)
    threshold = float(stats.chi2.ppf(1.0 - 0.01 / N, 1) / 2.0)

    tracks, summary, hard_stats = solve_fusion_cpp(
        frames1,
        parents1,
        frames2,
        parents2,
        matches_ij,
        vox_flat,
        vox_offsets,
        int(dim),
        float(threshold),
        int(max_bb_nodes),
        int(batch_branch_size),
        bool(use_initial_greedy),
        bool(use_conflict_blocks),
        bool(conflict_blocks_star_only),
    )

    tracks = [np.asarray(t, dtype=np.int64) for t in tracks]
    return tracks, summary, hard_stats