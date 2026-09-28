import numpy as np
import pandas as pd
import copy
import os
from typing import Dict, List, Tuple, Any, Set, Optional
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'C_package'))
import ctypes
from scipy import stats
import warnings
from collections import deque, defaultdict
import heapq
import ctypes
from itertools import product
try:
    from C_package.fast_full_fusion_wrapper import solve_fusion_cpp_wrapper
except Exception:
    solve_fusion_cpp_wrapper = None
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_C_PACKAGE_DIR = os.path.join(_THIS_DIR, "C_package")
if _C_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _C_PACKAGE_DIR)

# from fast_subgraphs import build_subgraphs_cpp
try:
    from C_package.fast_easy_fusion_wrapper import build_easy_arcs_cpp_wrapper
except ImportError:
    # Backward-compatible name used during debugging.
    from C_package.fast_easy_fusion_wrapper import build_easy_arcs_debug_cpp_wrapper as build_easy_arcs_cpp_wrapper

# =========================
# Global C solver loader
# =========================

_CINDA = None
_EDT3D = None

def get_edt3d_lib():
    global _EDT3D

    if _EDT3D is not None:
        return _EDT3D

    if sys.platform == "win32":
        lib_path = os.path.join(_C_PACKAGE_DIR, "edt_3d.dll")
    else:
        lib_path = os.path.join(_C_PACKAGE_DIR, "libedt3d.so")

    _EDT3D = ctypes.CDLL(lib_path)

    _EDT3D.edt_3d.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    ]
    _EDT3D.edt_3d.restype = None

    return _EDT3D

def get_cinda_solver():
    global _CINDA

    if _CINDA is not None:
        return _CINDA

    if sys.platform == 'win32':
        lib_path = os.path.join('C_package', 'lib_cinda_funcs.dll')
    else:
        lib_path = os.path.join('C_package', 'lib_cinda_funcs.so')

    _CINDA = ctypes.CDLL(lib_path)

    _CINDA.pyCS2.argtypes = (
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    )
    _CINDA.pyCS2.restype = ctypes.POINTER(ctypes.c_longlong)

    return _CINDA

def solve_fusion_v2(movieInfo1_partial, movieInfo2_partial, matches, return_stats=False):
    if solve_fusion_cpp_wrapper is not None:
        tracks, summary, hard_stats = solve_fusion_cpp_wrapper(
            movieInfo1_partial,
            movieInfo2_partial,
            matches,
            max_bb_nodes=1000000,
            batch_branch_size=2,
            use_initial_greedy=True,
            use_conflict_blocks=True,
            conflict_blocks_star_only=True,
        )

        print_fusion_stats_table(summary, hard_stats)

        if return_stats:
            return tracks, summary, hard_stats

        return tracks


def solve_fusion(movieInfo1_partial, movieInfo2_partial, matches, return_stats=False):
    """
    Build subgraphs, solve easy subgraphs and hard subgraphs.

    If return_stats=True, also return compact statistics.
    """
    t_total = time.perf_counter()

    print("\nStep 4-1 + Step 4-2a: Building easy arcs with C++...")
    n1 = len(movieInfo1_partial["frames"])
    n2 = len(movieInfo2_partial["frames"])
    N = n1 + n2
    detection_arcs, transition_arcs, easy_subgraphs, hard_subgraphs, edges = (
        build_easy_arcs_cpp_wrapper(
            movieInfo1_partial,
            movieInfo2_partial,
            matches,
        )
    )

    easy_total_nodes = int(sum(len(g) for g in easy_subgraphs))
    hard_node_sizes = [int(len(g)) for g in hard_subgraphs]

    detection_arcs = np.ascontiguousarray(detection_arcs, dtype=np.float64)
    transition_arcs = np.ascontiguousarray(transition_arcs, dtype=np.float64)

    transition_arcs = recompute_2d_transition_costs_by_edt(
        transition_arcs=transition_arcs,
        movieInfo1_partial=movieInfo1_partial,
        movieInfo2_partial=movieInfo2_partial,
        verbose=True,
    )

    print("\nStep 4-2b: Solving easy subgraphs with Python mcc4mot...")
    trajectories, costs = mcc4mot(detection_arcs, transition_arcs)

    tracks_easy = []
    for track in trajectories:
        track = np.asarray(track, dtype=int)
        track = track[track < N]  # remove pseudo/control nodes
        if len(track) > 0:
            tracks_easy.append(track)

    # Convert matches to merged/global node indexing.
    matches_arr = np.asarray(matches, dtype=np.int64).copy()
    if matches_arr.size == 0:
        matches_arr = np.empty((0, 2), dtype=np.int64)
    else:
        if matches_arr.ndim != 2 or matches_arr.shape[1] < 2:
            raise ValueError(f"matches must have shape (M, >=2), got {matches_arr.shape}")
        matches_arr = matches_arr[:, :2].copy()
        matches_arr[:, 1] += n1

    # Optional sanity check: if both endpoints of a matched pair are already
    # selected by easy subgraphs, warn the user. This preserves your previous
    # check but avoids repeated np.concatenate inside the loop.
    if len(tracks_easy) > 0:
        selected_easy = set(np.concatenate(tracks_easy).astype(int).tolist())
    else:
        selected_easy = set()

    matching_rows = [
        i
        for i, row in enumerate(matches_arr)
        if int(row[0]) in selected_easy and int(row[1]) in selected_easy
    ]
    if len(matching_rows) > 0:
        warnings.warn(
            f"Warning: Some matches are included in easy subgraphs. Rows: {matching_rows}"
        )

    print("\nStep 4-3: Solving hard subgraphs...")
    tracks_hard = []
    hard_stats = []

    for hard_id, subgraph_nodes in enumerate(hard_subgraphs):
        current_ids = set(subgraph_nodes)
        mask = np.isin(matches_arr[:, 0], list(current_ids))
        matches_hard = matches_arr[mask]

        subtracks_hard, sub_stats = solve_hard_fusion(
            current_ids,
            movieInfo1_partial,
            movieInfo2_partial,
            matches_hard,
            edges,
            print_block_summary=False,
            hard_id = hard_id
        )

        tracks_hard.extend(subtracks_hard)
        hard_stats.append(sub_stats)

    final_tracks = tracks_easy + tracks_hard

    total_time = time.perf_counter() - t_total

    summary = {
        "total_step4_5_time_sec": float(total_time),
        "num_easy_subgraphs": int(len(easy_subgraphs)),
        "num_hard_subgraphs": int(len(hard_subgraphs)),
        "easy_total_nodes": int(easy_total_nodes),
        "hard_node_sizes": hard_node_sizes,
    }

    print_fusion_stats_table(summary, hard_stats)

    if return_stats:
        return final_tracks, summary, hard_stats

    return final_tracks

def build_subgraphs(movieInfo1_partial, movieInfo2_partial, matches):
    """
    Build joint graph, extract connected components, and classify into easy vs hard subgraphs.

    Parameters
    ----------
    movieInfo1_partial : dict
        Information for movie 1 (frames, parents, etc.)
    movieInfo2_partial : dict
        Information for movie 2 (frames, parents, etc.)
    matches : array-like
        Cross-movie matches of shape (M, 3) where each row is (id1, id2, cost)

    Returns
    -------
    easy_subgraphs : list of lists
        List of node lists for easy subgraphs (no conflicts within a frame)
    hard_subgraphs : list of lists
        List of node lists for hard subgraphs (contains conflicts)
    """
    # -------------------------------------------------
    # Basic sizes and index mapping
    # -------------------------------------------------
    n1 = len(movieInfo1_partial['frames'])
    n2 = len(movieInfo2_partial['frames'])
    N = n1 + n2  # total number of nodes

    def m2(j):
        return n1 + j

    # -------------------------------------------------
    # Build adjacency lists: directed for parent-child, undirected for connectivity
    # -------------------------------------------------
    dir_edges = [[] for _ in range(N)]  # Directed edges (parent -> child)
    undir_edges = [[] for _ in range(N)]  # Undirected edges (both directions)

    children1 = {i: [] for i in range(n1)}
    for i, p in enumerate(movieInfo1_partial['parents']):
        if p != -1:
            children1[p].append(i)
            
    children2 = {j: [] for j in range(n2)}
    for j, p in enumerate(movieInfo2_partial['parents']):
        if p != -1:
            children2[p].append(j)

    # -------------------------------------------------
    # Intra-movie parent-child edges (directed)
    # -------------------------------------------------
    for i, p in enumerate(movieInfo1_partial['parents']):
        if p != -1:
            dir_edges[p].append(i)
            # For undirected connectivity, add bidirectional edges
            undir_edges[p].append(i)
            undir_edges[i].append(p)

    for j, p in enumerate(movieInfo2_partial['parents']):
        if p != -1:
            src = m2(p)
            dst = m2(j)
            dir_edges[src].append(dst)
            # For undirected connectivity, add bidirectional edges
            undir_edges[src].append(dst)
            undir_edges[dst].append(src)

    # -------------------------------------------------
    # Cross-movie edges induced by matches (add to both directed and undirected)
    # -------------------------------------------------
    for i1, i2 in matches:
        i1 = int(i1)
        i2 = int(i2)

        u = i1
        v = m2(i2)
        
        undir_edges[u].append(v)
        undir_edges[v].append(u)

        # m2 cell connects to parent of m1 cell (if exists)
        p1 = movieInfo1_partial['parents'][i1]
        p2 = movieInfo2_partial['parents'][i2]
        childs1 = children1.get(i1, [])
        childs2 = children2.get(i2, [])

        if p1 != -1:
            # Add bidirectional edges for undirected connectivity
            undir_edges[v].append(p1)
            undir_edges[p1].append(v)
            dir_edges[p1].append(v)

        if p2 != -1:
            undir_edges[u].append(m2(p2))
            undir_edges[m2(p2)].append(u)
            dir_edges[m2(p2)].append(u)
        
        for c1 in childs1:
            undir_edges[v].append(c1)
            undir_edges[c1].append(v)
            dir_edges[v].append(c1)

        for c2 in childs2:
            undir_edges[u].append(m2(c2))
            undir_edges[m2(c2)].append(u)
            dir_edges[u].append(m2(c2))


    for i in range(len(dir_edges)):
        if dir_edges[i]: 
            dir_edges[i] = list(set(dir_edges[i]))   
    # -------------------------------------------------
    # Connected components via BFS on UNDIRECTED edges
    # -------------------------------------------------
    visited = bytearray(N)  # much faster than set
    subgraphs = []

    for start in range(N):
        if visited[start]:
            continue

        queue = deque([start])
        visited[start] = 1
        nodes = []

        while queue:
            u = queue.popleft()
            nodes.append(u)
            for v in undir_edges[u]:  # Use undirected edges for connectivity
                if not visited[v]:
                    visited[v] = 1
                    queue.append(v)

        subgraphs.append(nodes)

    # -------------------------------------------------
    # Easy / hard classification
    # -------------------------------------------------
    easy_subgraphs = []
    hard_subgraphs = []

    frames1 = np.array(movieInfo1_partial['frames'], dtype=int)
    frames2 = np.array(movieInfo2_partial['frames'], dtype=int)

    for nodes in subgraphs:
        frame_count = defaultdict(lambda: [0, 0])  # [m1_count, m2_count]
        easy = True

        for u in nodes:
            if u < n1:
                f = frames1[u]
                frame_count[f][0] += 1
                if frame_count[f][0] > 1:
                    easy = False
                    break
            else:
                f = frames2[u - n1]
                frame_count[f][1] += 1
                if frame_count[f][1] > 1:
                    easy = False
                    break

        if easy:
            easy_subgraphs.append(nodes)
        else:
            hard_subgraphs.append(nodes)

    return easy_subgraphs, hard_subgraphs, dir_edges

def pruning(subgraphs, movieInfo1_partial, movieInfo2_partial, edges):
    
    n1 = len(movieInfo1_partial['frames'])
    n2 = len(movieInfo2_partial['frames'])
    N = n1 + n2  # total number of nodes
    frames1 = np.array(movieInfo1_partial['frames'], dtype=int)
    frames2 = np.array(movieInfo2_partial['frames'], dtype=int)
    # Build detection_arcs and transition_arcs for easy subgraphs
    # -------------------------------------------------
    # Calculate chi-squared threshold for cost calculation
    threshold = stats.chi2.ppf(1 - 0.01 / N, 1) / 2
    big = (-(np.max(frames2) - np.min(frames1) + 1) * (-2 * threshold - 0.00001))+1
    detection_arcs_original = np.full((N, 3), [big, 2 * threshold, -2 * threshold - 0.00001], dtype=np.float32)
    detection_arcs_control = np.full((len(subgraphs), 3), [threshold, threshold, -2 * threshold - 0.00001], dtype=np.float32)
    detection_arcs = np.concatenate((detection_arcs_original, detection_arcs_control), axis=0)
    # detection_arcs = np.full((len(subgraphs)+N, 3), [threshold, threshold, -2 * threshold - 0.00001], dtype=np.float32)
    ids = np.arange(detection_arcs.shape[0])
    detection_arcs = np.column_stack((ids, detection_arcs))

    transition_arcs = []

    for i, subgraph in enumerate(subgraphs):
        subgraph = np.array(subgraph, dtype=np.int32)
        mask = subgraph < n1
        indices_part1 = subgraph[mask]
        indices_part2 = subgraph[~mask] - n1

        if len(indices_part1) > 0:
            partial_frames1 = np.array(frames1)[indices_part1] 
        else:
            partial_frames1 = None     
        if len(indices_part2) > 0:
            partial_frames2 = np.array(frames2)[indices_part2]
        else:
            partial_frames2 = None

        min_frame = min([np.min(p) for p in [partial_frames1, partial_frames2] if p is not None], default=-np.inf)
        max_frame = max([np.max(p) for p in [partial_frames1, partial_frames2] if p is not None], default=np.inf)

        if len(indices_part1) > 0:
            max_idx_in_part1 = indices_part1[np.where(partial_frames1 == max_frame)[0]]
            min_idx_in_part1 = indices_part1[np.where(partial_frames1 == min_frame)[0]]
        else:
            max_idx_in_part1 = None
            min_idx_in_part1 = None

        if len(indices_part2) > 0:
            max_idx_in_part2 = indices_part2[np.where(partial_frames2 == max_frame)[0]] + n1
            min_idx_in_part2 = indices_part2[np.where(partial_frames2 == min_frame)[0]] + n1
        else:
            max_idx_in_part2 = None
            min_idx_in_part2 = None

        first_frame_nodes = np.concatenate([x for x in [min_idx_in_part1, min_idx_in_part2] if x is not None])
        last_frame_nodes = np.concatenate([x for x in [max_idx_in_part1, max_idx_in_part2] if x is not None])

        # Add transition arcs: calculate the cost between nodes using their indices from combined graph
        for u in subgraph:
            for v in edges[u]:
                # Determine cost based on origin (movieInfo1 or movieInfo2)
                if u < n1 and v < n1:  # Both in movieInfo1
                    cost = ovDistanceRegion(movieInfo1_partial['vox'][u], movieInfo1_partial['vox'][v])[0]
                elif u >= n1 and v >= n1:  # Both in movieInfo2
                    cost = ovDistanceRegion(movieInfo2_partial['vox'][u - n1], movieInfo2_partial['vox'][v - n1])[0]
                elif u < n1 and v >= n1:  # One in movieInfo1, other in movieInfo2
                    cost = ovDistanceRegion(movieInfo1_partial['vox'][u], movieInfo2_partial['vox'][v - n1])[0]
                elif u >= n1 and v < n1:  # One in movieInfo2, other in movieInfo1
                    cost = ovDistanceRegion(movieInfo2_partial['vox'][u - n1], movieInfo1_partial['vox'][v])[0]
                transition_arcs.append([u, v, cost])

        # Add edges from pseudo-node to the first frame nodes of each subgraph
        for first_frame_node in first_frame_nodes:
            transition_arcs.append([N+i, first_frame_node, 0.0])
        for last_frame_node in last_frame_nodes:
            detection_arcs[last_frame_node, 2] = threshold

    # Call mcc4mot to solve the optimization problem
    transition_arcs = np.array(transition_arcs, dtype=np.float32)
    trajectories, costs = mcc4mot(detection_arcs, transition_arcs)
    trajectories_filtered = [track[track < N] for track in trajectories]
    return trajectories_filtered, costs

def find_feasible_solution_greedy(
    root_sol,
    root_violations,
    matches_hard,
    node_in,
    node_out,
    base_detection_arcs,
    base_transition_arcs,
    max_iter=100,
    verbose=True,
):
    """
    Find a feasible incumbent before exact B&B.

    This is only used to improve the initial upper bound. The exact B&B still
    runs afterwards, so global optimality is not affected.
    """
    if root_sol is None or root_violations is None:
        return None, float("inf")

    current_detection = base_detection_arcs.copy()
    current_transition = base_transition_arcs.copy()
    current_fixed = {}
    current_sol = root_sol
    current_violations = list(root_violations)

    best_feasible_sol = None
    best_feasible_cost = float("inf")
    flow_calls = 0

    if verbose:
        print(f"[INIT-GREEDY] start: violations={len(current_violations)}")

    for it in range(max_iter):
        current_violations = [m for m in current_violations if m not in current_fixed]

        if len(current_violations) == 0:
            node = {
                "fixed": current_fixed,
                "detection": current_detection,
                "transition": current_transition,
            }
            sol, cost, violations = solve_flow_node(node, matches_hard, node_in, node_out)
            flow_calls += 1
            if sol is not None and violations is not None and len(violations) == 0:
                best_feasible_sol = sol
                best_feasible_cost = cost
            break

        match_id = current_violations[0]
        u, v = matches_hard[match_id][:2]
        u = int(u)
        v = int(v)

        candidates = []
        for choice in [0, 1]:
            # choice=0: forbid v; choice=1: forbid u.
            forbid_node = v if choice == 0 else u

            cand_detection = current_detection.copy()
            cand_transition = current_transition.copy()
            cand_detection, cand_transition = apply_forbid(
                cand_detection,
                cand_transition,
                forbid_node,
                node_in,
                node_out,
            )

            cand_fixed = dict(current_fixed)
            cand_fixed[match_id] = choice

            cand_node = {
                "fixed": cand_fixed,
                "detection": cand_detection,
                "transition": cand_transition,
            }
            cand_sol, cand_cost, cand_violations = solve_flow_node(
                cand_node,
                matches_hard,
                node_in,
                node_out,
            )
            flow_calls += 1

            if cand_sol is None or cand_violations is None:
                continue

            cand_violations = [m for m in cand_violations if m not in cand_fixed]
            candidates.append(
                (
                    len(cand_violations),
                    cand_cost,
                    choice,
                    cand_fixed,
                    cand_detection,
                    cand_transition,
                    cand_sol,
                    cand_violations,
                )
            )

        if len(candidates) == 0:
            break

        # Prefer reducing violations; if tied, prefer lower cost.
        candidates.sort(key=lambda x: (x[0], x[1]))
        (
            remain_viol_count,
            cand_cost,
            choice,
            current_fixed,
            current_detection,
            current_transition,
            current_sol,
            current_violations,
        ) = candidates[0]

        if remain_viol_count == 0:
            best_feasible_sol = current_sol
            best_feasible_cost = cand_cost
            break

    if verbose:
        if best_feasible_sol is not None:
            print(
                f"[INIT-GREEDY] found feasible: "
                f"cost={best_feasible_cost:.6f}, flow_calls={flow_calls}"
            )
        else:
            print(f"[INIT-GREEDY] failed to find feasible, flow_calls={flow_calls}")

    return best_feasible_sol, best_feasible_cost

def select_batch_violations(violations, min_switch_cost, batch_size=2):
    """
    Fallback match-level branching.

    This remains exact and is used when no safe conflict block covers the
    current violated matches.
    """
    violations = list(violations)
    if len(violations) == 0:
        return []

    batch_size = max(1, min(int(batch_size), len(violations)))
    return sorted(
        violations,
        key=lambda m: min_switch_cost.get(m, 0.0),
        reverse=True,
    )[:batch_size]

def generate_exact_batch_assignments(selected_matches):
    """
    Enumerate all 2^k assignments for selected matches.

    choice=0 means forbid the second endpoint v.
    choice=1 means forbid the first endpoint u.
    """
    selected_matches = list(selected_matches)
    for bits in product([0, 1], repeat=len(selected_matches)):
        yield {mid: choice for mid, choice in zip(selected_matches, bits)}

def build_batch_child(node, assignment, matches_hard, node_in, node_out):
    """
    Build one fallback B&B child by applying a complete assignment for several
    individual matches.
    """
    child_fixed = dict(node["fixed"])
    child_detection = node["detection"].copy()
    child_transition = node["transition"].copy()

    forbid_nodes = []
    for match_id, choice in assignment.items():
        match_id = int(match_id)
        choice = int(choice)

        if match_id in child_fixed:
            if child_fixed[match_id] != choice:
                return None, None
            continue

        u, v = matches_hard[match_id][:2]
        u = int(u)
        v = int(v)
        forbid_node = v if choice == 0 else u

        child_fixed[match_id] = choice
        forbid_nodes.append(forbid_node)

    child_detection, child_transition = apply_forbid_many(
        child_detection,
        child_transition,
        forbid_nodes,
        node_in,
        node_out,
    )
    if child_detection is None:
        return None, None

    child = {
        "fixed": child_fixed,
        "detection": child_detection,
        "transition": child_transition,
    }
    child_key = tuple(sorted(child_fixed.items()))
    child["depth"] = node.get("depth", 0) + 1
    return child, child_key

def _cell_side(cell_id, n1):
    return 0 if int(cell_id) < n1 else 1

def _cell_frame(cell_id, n1, frames1, frames2):
    cell_id = int(cell_id)
    if cell_id < n1:
        return int(frames1[cell_id])
    return int(frames2[cell_id - n1])

def build_frame_conflict_blocks(
    matches_hard,
    movieInfo1_partial,
    movieInfo2_partial,
    star_only=True,
):
    """
    Build same-frame bipartite conflict blocks from matches_hard.

    A block is a connected component in the bipartite graph formed by matches
    at the same frame.  For speed and safety, the default `star_only=True`
    keeps only k-vs-1 or 1-vs-k blocks.  This exactly covers cases like:

        {843, 893} vs {764}
        {1558} vs {1478, 1584}

    and avoids prematurely compressing general k-vs-m blocks, where mixed
    selections might be meaningful in some datasets.

    Returns
    -------
    conflict_blocks : list[dict]
        Each block has keys: block_id, frame, left_nodes, right_nodes,
        match_ids, kind.
    match_to_block : dict[int, int]
        Map from match_id to block_id.
    """
    matches_hard = np.asarray(matches_hard)
    n1 = len(movieInfo1_partial["frames"])
    frames1 = np.asarray(movieInfo1_partial["frames"], dtype=int)
    frames2 = np.asarray(movieInfo2_partial["frames"], dtype=int)

    # frame -> adjacency over original cell ids, using only same-frame matches.
    frame_adj = defaultdict(lambda: defaultdict(set))
    frame_match_ids = defaultdict(list)

    for mid, row in enumerate(matches_hard):
        u = int(row[0])
        v = int(row[1])
        fu = _cell_frame(u, n1, frames1, frames2)
        fv = _cell_frame(v, n1, frames1, frames2)
        if fu != fv:
            # Keep cross-frame matches as normal match-level constraints.
            continue
        if _cell_side(u, n1) == _cell_side(v, n1):
            # Not a cross-movie bipartite match; do not compress.
            continue

        f = fu
        frame_adj[f][u].add(v)
        frame_adj[f][v].add(u)
        frame_match_ids[f].append(mid)

    conflict_blocks = []
    match_to_block = {}

    # Helper map from unordered endpoint pair to match id.
    pair_to_match = {}
    for mid, row in enumerate(matches_hard):
        u = int(row[0])
        v = int(row[1])
        pair_to_match[frozenset((u, v))] = mid

    for f in sorted(frame_adj.keys()):
        adj = frame_adj[f]
        visited = set()

        for start in list(adj.keys()):
            if start in visited:
                continue

            q = deque([start])
            visited.add(start)
            comp_nodes = []

            while q:
                x = q.popleft()
                comp_nodes.append(x)
                for y in adj[x]:
                    if y not in visited:
                        visited.add(y)
                        q.append(y)

            if len(comp_nodes) < 2:
                continue

            left_nodes = sorted([x for x in comp_nodes if _cell_side(x, n1) == 0])
            right_nodes = sorted([x for x in comp_nodes if _cell_side(x, n1) == 1])

            if len(left_nodes) == 0 or len(right_nodes) == 0:
                continue

            block_match_ids = []
            for u in left_nodes:
                for v in right_nodes:
                    mid = pair_to_match.get(frozenset((u, v)))
                    if mid is not None:
                        block_match_ids.append(int(mid))

            block_match_ids = sorted(set(block_match_ids))
            if len(block_match_ids) <= 1:
                continue

            # Default exact-safe compression: only k-vs-1 or 1-vs-k blocks.
            if star_only and not (len(left_nodes) == 1 or len(right_nodes) == 1):
                continue

            block_id = len(conflict_blocks)
            if len(left_nodes) == 1 and len(right_nodes) > 1:
                kind = "1-vs-k"
            elif len(left_nodes) > 1 and len(right_nodes) == 1:
                kind = "k-vs-1"
            else:
                kind = "k-vs-m"

            block = {
                "block_id": block_id,
                "frame": int(f),
                "left_nodes": tuple(int(x) for x in left_nodes),
                "right_nodes": tuple(int(x) for x in right_nodes),
                "match_ids": tuple(int(x) for x in block_match_ids),
                "kind": kind,
            }
            conflict_blocks.append(block)

            for mid in block_match_ids:
                match_to_block[int(mid)] = block_id

    return conflict_blocks, match_to_block

def count_root_conflict_units(root_violations, match_to_block):
    """
    Count conflict units.

    If several violated matches belong to the same same-frame conflict block,
    count them as one conflict unit.
    Matches outside conflict blocks are counted individually.
    """
    used_blocks = set()
    single_count = 0

    for mid in root_violations:
        mid = int(mid)
        bid = match_to_block.get(mid)

        if bid is None:
            single_count += 1
        else:
            used_blocks.add(int(bid))

    return len(used_blocks) + single_count

def print_fusion_stats_table(summary, hard_stats):
    """
    Print compact final statistics table.
    Also print:
      1. hard_total_nodes
      2. the row with the largest max_depth
    """
    print("\n" + "=" * 100)
    print("Fusion Step 4&5 Statistics")
    print("=" * 100)

    print(
        f"total_step4_5_time_sec: "
        f"{summary.get('total_step4_5_time_sec', np.nan):.4f}"
    )

    rows = []
    for s in hard_stats:
        rows.append({
            "hard_id": s.get("hard_id", -1),
            "nodes": s.get("num_nodes", s.get("nodes", 0)),
            "matches": s.get("num_matches", s.get("matches", 0)),
            "root_satisfied_matches": s.get("root_satisfied_matches", 0),
            "root_violated_matches": s.get("root_violated_matches", 0),
            "max_depth": s.get("max_depth", 0),
            "branches_explored": s.get("branches_explored", 0),
            "flow_calls": s.get("flow_calls", 0),
            "certified": s.get("certified_optimal", s.get("certified", False)),
            "reason": s.get("termination_reason", s.get("reason", "")),
            "time_flow_sec": s.get("time_solve_flow", s.get("time_flow_sec", 0.0)),
        })

    # 优先从 hard_stats 统计 hard_total_nodes；
    # 如果 hard_stats 为空，则尝试从 summary["hard_node_sizes"] 中统计。
    if len(rows) > 0:
        hard_total_nodes = int(sum(int(r["nodes"]) for r in rows))
    else:
        hard_total_nodes = int(sum(summary.get("hard_node_sizes", [])))

    print(
        f"easy_subgraphs: {summary.get('num_easy_subgraphs', 0)}, "
        f"hard_subgraphs: {summary.get('num_hard_subgraphs', 0)}, "
        f"easy_total_nodes: {summary.get('easy_total_nodes', 0)}, "
        f"hard_total_nodes: {hard_total_nodes}"
    )

    if len(rows) == 0:
        print("No hard subgraphs.")
        print("=" * 100)
        return

    df = pd.DataFrame(rows)

    print("\n[HARD_STATS]")
    print(df.to_string(index=False))

    # 单独输出 max_depth 最大的那一行
    # 如果有多个 hard subgraph 的 max_depth 相同，这里默认输出第一个。
    max_depth_idx = df["max_depth"].astype(float).idxmax()
    max_depth_row = df.loc[[max_depth_idx]]

    print("\n[MAX_DEPTH_MAX_ROW]")
    print(max_depth_row.to_string(index=False))

    print("=" * 100)
    
def summarize_conflict_blocks(conflict_blocks, n1, max_print=20):
    """Print a compact summary of conflict blocks used for B&B branching."""
    print(
        f"[BLOCKS] usable_same_frame_blocks={len(conflict_blocks)} "
        f"(default: star blocks only)"
    )
    for block in conflict_blocks[:max_print]:
        left_local = [int(x) for x in block["left_nodes"]]
        right_local = [int(x) - n1 for x in block["right_nodes"]]
        # print(
        #     f"  block[{block['block_id']}] frame={block['frame']} "
        #     f"kind={block['kind']} "
        #     f"matches={list(block['match_ids'])} "
        #     f"M1={left_local} vs M2={right_local}"
        # )
    if len(conflict_blocks) > max_print:
        print(f"  ... {len(conflict_blocks) - max_print} more blocks not printed")

def select_conflict_block(
    violations,
    fixed,
    conflict_blocks,
    match_to_block,
    min_switch_cost,
):
    """
    Select one unresolved conflict block for branching.

    Preference:
      1) covers many current violations;
      2) has larger estimated correction impact;
      3) contains more matches.
    """
    unresolved = [int(m) for m in violations if int(m) not in fixed]
    if len(unresolved) == 0:
        return None

    candidate_block_ids = set()
    for mid in unresolved:
        bid = match_to_block.get(mid)
        if bid is not None:
            candidate_block_ids.add(int(bid))

    if len(candidate_block_ids) == 0:
        return None

    unresolved_set = set(unresolved)
    best_key = None
    best_block = None

    for bid in candidate_block_ids:
        block = conflict_blocks[bid]
        mids = list(block["match_ids"])
        unresolved_in_block = [m for m in mids if m in unresolved_set]
        if len(unresolved_in_block) == 0:
            continue

        impact = sum(max(float(min_switch_cost.get(m, 0.0)), 0.0) for m in unresolved_in_block)
        key = (len(unresolved_in_block), impact, len(mids), -int(block["frame"]))

        if best_key is None or key > best_key:
            best_key = key
            best_block = block

    return best_block

def _assignment_for_forbid_node(match_id, forbid_node, matches_hard):
    """
    Convert a forbidden original node into the old match-level choice encoding.

    choice=0 means forbid v.
    choice=1 means forbid u.
    """
    u, v = matches_hard[int(match_id)][:2]
    u = int(u)
    v = int(v)
    forbid_node = int(forbid_node)

    if forbid_node == v:
        return 0
    if forbid_node == u:
        return 1
    return None

def build_block_child(node, block, choose_side, matches_hard, node_in, node_out):
    """
    Build one B&B child from a conflict block.

    choose_side=0: choose the left/M1 side, forbid all right/M2 nodes.
    choose_side=1: choose the right/M2 side, forbid all left/M1 nodes.

    The node still stores `fixed` in match-level encoding so that existing
    violation filtering and state caching continue to work.
    """
    choose_side = int(choose_side)
    if choose_side not in (0, 1):
        return None, None

    if choose_side == 0:
        forbid_nodes = list(block["right_nodes"])
    else:
        forbid_nodes = list(block["left_nodes"])

    forbid_set = set(int(x) for x in forbid_nodes)

    child_fixed = dict(node["fixed"])

    for mid in block["match_ids"]:
        u, v = matches_hard[int(mid)][:2]
        u = int(u)
        v = int(v)

        if u in forbid_set and v in forbid_set:
            return None, None
        if u not in forbid_set and v not in forbid_set:
            # This should not happen for a proper bipartite conflict block.
            return None, None

        forbid_node = u if u in forbid_set else v
        choice = _assignment_for_forbid_node(mid, forbid_node, matches_hard)
        if choice is None:
            return None, None

        if int(mid) in child_fixed:
            if int(child_fixed[int(mid)]) != int(choice):
                return None, None
        else:
            child_fixed[int(mid)] = int(choice)

    child_detection = node["detection"].copy()
    child_transition = node["transition"].copy()
    child_detection, child_transition = apply_forbid_many(
        child_detection,
        child_transition,
        forbid_nodes,
        node_in,
        node_out,
    )
    if child_detection is None:
        return None, None

    child = {
        "fixed": child_fixed,
        "detection": child_detection,
        "transition": child_transition,
    }
    child_key = tuple(sorted(child_fixed.items()))
    child["depth"] = node.get("depth", 0) + 1
    return child, child_key

def compute_corrected_lower_bound_by_blocks(
    current_cost,
    violations,
    correction_data,
    best_cost,
    conflict_blocks=None,
    match_to_block=None,
):
    if not violations:
        return current_cost

    if not conflict_blocks or not match_to_block:
        return compute_corrected_lower_bound(
            current_cost,
            violations,
            correction_data,
            best_cost,
        )

    viol_set = set(int(m) for m in violations)
    used_blocks = set()
    min_fix_cost = 0.0

    for mid in viol_set:
        bid = match_to_block.get(mid)

        if bid is None:
            min_fix_cost += float(correction_data.get(mid, 0.0))
            continue

        if bid in used_blocks:
            continue

        used_blocks.add(bid)
        block = conflict_blocks[int(bid)]

        vals = [
            float(correction_data.get(m, 0.0))
            for m in block["match_ids"]
            if int(m) in viol_set
        ]

        if vals:
            min_fix_cost += max(vals)

    corrected_lb = current_cost + min_fix_cost
    if corrected_lb >= best_cost:
        return float("inf")

    return corrected_lb

def solve_hard_fusion(hard_ids,
                      movieInfo1_partial,
                      movieInfo2_partial,
                      matches_hard,
                      edges,
                      max_bb_nodes=1000000,
                      batch_branch_size=2,
                      use_initial_greedy=True,
                      use_conflict_blocks=True,
                      conflict_blocks_star_only=True,
                      debug_build_graph=False,
                      print_block_summary=False,
                      hard_id = -1,
                      tol=1e-6):
    """
    Exact Branch-and-Bound + Min-Cost Flow for hard fusion.

    Main change:
        - Prefer same-frame conflict-block branching over individual-match
          branching.  A star conflict block such as {843, 893} vs {764} is
          branched as two children:
              choose {843, 893}, forbid {764}
              choose {764}, forbid {843, 893}
        - Fallback to the old exact match-level batch branching when no safe
          block covers the current violations.
        - The block mode is exact-safe by default because it only compresses
          k-vs-1 / 1-vs-k blocks.  General k-vs-m blocks are left to fallback
          match-level branching unless conflict_blocks_star_only=False.
    """
    flow_calls = 0
    time_solve_flow = 0.0

    base_detection_arcs, base_transition_arcs, expand_to_original, \
        node_in, node_out, node_single, matches_hard, threshold = build_base_graph(
            hard_ids,
            movieInfo1_partial,
            movieInfo2_partial,
            matches_hard,
            edges,
            debug=debug_build_graph,
        )

    n1 = len(movieInfo1_partial["frames"])

    if use_conflict_blocks:
        conflict_blocks, match_to_block = build_frame_conflict_blocks(
            matches_hard,
            movieInfo1_partial,
            movieInfo2_partial,
            star_only=conflict_blocks_star_only,
        )
    else:
        conflict_blocks, match_to_block = [], {}

    if print_block_summary and len(matches_hard) > 0:
        summarize_conflict_blocks(conflict_blocks, n1)

    best_cost = float("inf")
    best_solution = None
    flow_cache = {}

    pq = []
    counter = 0

    root = {
        "fixed": {},
        "detection": base_detection_arcs,
        "transition": base_transition_arcs,
        "depth": 0,
    }

    t0 = time.perf_counter()
    root_sol, root_cost, root_violations = solve_flow_node(
        root,
        matches_hard,
        node_in,
        node_out,
    )
    time_solve_flow += time.perf_counter() - t0
    flow_calls += 1

    if root_violations is None:
        root_violations = []

    
    if root_violations is None:
        root_violations = []

    root_violations = set(int(m) for m in root_violations)

    num_matches = len(matches_hard)

    root_violated_matches = len(root_violations)
    root_satisfied_matches = num_matches - root_violated_matches

    # Step 1: initial feasible incumbent.
    if root_sol is not None and root_violations is not None and len(root_violations) == 0:
        best_cost = root_cost
        best_solution = root_sol
        print(f"[INIT-ROOT] root is feasible: cost={best_cost:.6f}")

    elif (
        use_initial_greedy
        and root_sol is not None
        and root_violations is not None
        and len(root_violations) > 0
    ):
        greedy_sol, greedy_cost = find_feasible_solution_greedy(
            root_sol=root_sol,
            root_violations=root_violations,
            matches_hard=matches_hard,
            node_in=node_in,
            node_out=node_out,
            base_detection_arcs=base_detection_arcs,
            base_transition_arcs=base_transition_arcs,
            max_iter=100,
            verbose=True,
        )

        if greedy_sol is not None and greedy_cost < best_cost:
            print(
                f"[INIT-BEST] improved initial feasible: "
                f"new_best={greedy_cost:.6f}"
            )
            best_cost = greedy_cost
            best_solution = greedy_sol

    min_switch_cost, correction_data, cost_diff, u_switch_cost, v_switch_cost = \
        initialize_correction_data_fast(
            base_transition_arcs,
            matches_hard,
            node_in,
            node_out,
            threshold,
        )

    if root_sol is not None and root_violations is not None:
        corrected_root_lb = compute_corrected_lower_bound_by_blocks(
            root_cost,
            root_violations,
            correction_data,
            best_cost,
            conflict_blocks=conflict_blocks,
            match_to_block=match_to_block,
        )
    else:
        corrected_root_lb = float("inf")

    # root_key = tuple(sorted(root["fixed"].items()))
    # seen_states = {root_key}
    root_violations_list = [] if root_violations is None else [int(m) for m in root_violations]

    root_key = tuple(sorted(root["fixed"].items()))
    flow_cache[root_key] = (root_sol, root_cost, root_violations_list)
    seen_states = {root_key}

    root_violations = set(root_violations_list)

    if corrected_root_lb != float("inf") and corrected_root_lb < best_cost - tol:
        heapq.heappush(pq, (corrected_root_lb, counter, root, root_violations))
        counter += 1

    explored = 0
    certified_optimal = False
    termination_reason = "unknown"
    final_min_lb = None
    block_branches = 0
    fallback_match_branches = 0
    max_depth = 0
    print(
        f"[BB-START] hard_nodes={len(hard_ids)}, matches={len(matches_hard)}, "
        f"blocks={len(conflict_blocks)}, batch_branch_size={batch_branch_size}, "
        f"root_lb={corrected_root_lb:.6f}, initial_best={best_cost:.6f}"
    )

    while pq and explored < max_bb_nodes:
        lb, _, node, _ = heapq.heappop(pq)

        if lb >= best_cost - tol:
            certified_optimal = True
            termination_reason = "min_heap_lb_reached_best"
            final_min_lb = lb
            break

        key = tuple(sorted(node["fixed"].items()))
        explored += 1
        max_depth = max(max_depth, node.get("depth", 0))
        if key in flow_cache:
            sol, cost, violations = flow_cache[key]
        else:
            t0 = time.perf_counter()
            sol, cost, violations = solve_flow_node(
                node,
                matches_hard,
                node_in,
                node_out,
            )
            time_solve_flow += time.perf_counter() - t0
            flow_calls += 1
            flow_cache[key] = (sol, cost, violations)

        if sol is None or violations is None:
            continue

        violations = [int(m) for m in violations if int(m) not in node["fixed"]]

        if len(violations) == 0:
            if cost < best_cost - tol:
                print(
                    f"[BB-BEST] old_best={best_cost:.6f}, "
                    f"new_best={cost:.6f}, explored={explored}, "
                    f"fixed={len(node['fixed'])}, pq={len(pq)}"
                )
                best_cost = cost
                best_solution = sol
            continue

        if cost >= best_cost - tol:
            continue

        children_to_solve = []

        # =============================================================
        # 1) Prefer exact-safe conflict-block branching.
        # =============================================================
        block = None
        if use_conflict_blocks and len(conflict_blocks) > 0:
            block = select_conflict_block(
                violations=violations,
                fixed=node["fixed"],
                conflict_blocks=conflict_blocks,
                match_to_block=match_to_block,
                min_switch_cost=min_switch_cost,
            )

        if block is not None:
            block_branches += 1
            for choose_side in (0, 1):
                child, child_key = build_block_child(
                    node,
                    block,
                    choose_side,
                    matches_hard,
                    node_in,
                    node_out,
                )
                if child is None:
                    continue
                if child_key in seen_states:
                    continue
                seen_states.add(child_key)
                children_to_solve.append((child, child_key))

        # =============================================================
        # 2) Fallback: old match-level exact batch branching.
        # =============================================================
        if len(children_to_solve) == 0:
            fallback_match_branches += 1
            selected_matches = select_batch_violations(
                violations,
                min_switch_cost,
                batch_size=batch_branch_size,
            )

            if len(selected_matches) == 0:
                continue

            for assignment in generate_exact_batch_assignments(selected_matches):
                tmp_fixed = dict(node["fixed"])
                conflict = False
                for mid, choice in assignment.items():
                    if mid in tmp_fixed and tmp_fixed[mid] != choice:
                        conflict = True
                        break
                    tmp_fixed[mid] = choice
                if conflict:
                    continue

                child_key = tuple(sorted(tmp_fixed.items()))
                if child_key in seen_states:
                    continue

                child, child_key = build_batch_child(
                    node,
                    assignment,
                    matches_hard,
                    node_in,
                    node_out,
                )
                if child is None:
                    continue
                if child_key in seen_states:
                    continue
                seen_states.add(child_key)
                children_to_solve.append((child, child_key))

        for child, child_key in children_to_solve:
            t0 = time.perf_counter()
            child_sol, child_cost, child_violations = solve_flow_node(
                child,
                matches_hard,
                node_in,
                node_out,
            )
            time_solve_flow += time.perf_counter() - t0
            flow_calls += 1

            flow_cache[child_key] = (child_sol, child_cost, child_violations)

            if child_sol is None or child_violations is None:
                continue

            child_violations = [
                int(m) for m in child_violations
                if int(m) not in child["fixed"]
            ]

            if len(child_violations) == 0:
                if child_cost < best_cost - tol:
                    print(
                        f"[BB-BEST] old_best={best_cost:.6f}, "
                        f"new_best={child_cost:.6f}, explored={explored}, "
                        f"fixed={len(child['fixed'])}, pq={len(pq)}"
                    )
                    best_cost = child_cost
                    best_solution = child_sol
                continue

            child_lb = compute_corrected_lower_bound_by_blocks(
                child_cost,
                child_violations,
                correction_data,
                best_cost,
                conflict_blocks=conflict_blocks,
                match_to_block=match_to_block,
            )

            if child_lb == float("inf") or child_lb >= best_cost - tol:
                continue

            heapq.heappush(pq, (child_lb, counter, child, child_violations))
            counter += 1

    if certified_optimal:
        pass
    elif len(pq) == 0:
        certified_optimal = True
        termination_reason = "priority_queue_empty"
        final_min_lb = best_cost
    elif explored >= max_bb_nodes:
        certified_optimal = False
        termination_reason = "max_bb_nodes_reached"
        final_min_lb = pq[0][0] if len(pq) > 0 else best_cost
    else:
        certified_optimal = False
        termination_reason = "unknown_exit"
        final_min_lb = pq[0][0] if len(pq) > 0 else best_cost

    gap = max(0.0, best_cost - final_min_lb) if final_min_lb is not None else float("inf")

    print(
        f"[BB-END] hard_nodes={len(hard_ids)}, "
        f"matches={len(matches_hard)}, "
        f"blocks={len(conflict_blocks)}, "
        f"explored={explored}, "
        f"pq={len(pq)}, "
        f"seen={len(seen_states)}, "
        f"flow_calls={flow_calls}, "
        f"block_branches={block_branches}, "
        f"fallback_branches={fallback_match_branches}, "
        f"best={best_cost:.6f}, "
        f"final_min_lb={final_min_lb:.6f}, "
        f"gap={gap:.6f}, "
        f"certified_optimal={certified_optimal}, "
        f"reason={termination_reason}, "
        f"time_flow={time_solve_flow:.3f}s"
    )

    if not certified_optimal:
        warnings.warn(
            f"B&B did not certify optimality. "
            f"best={best_cost:.6f}, "
            f"final_min_lb={final_min_lb:.6f}, "
            f"gap={gap:.6f}, "
            f"reason={termination_reason}"
        )

    stats = {
        "hard_id": int(hard_id),
        "num_nodes": int(len(hard_ids)),
        "num_matches": int(num_matches),

        "root_satisfied_matches": int(root_satisfied_matches),
        "root_violated_matches": int(root_violated_matches),

        "max_depth": int(max_depth),   # ← 现在是 block depth
        "branches_explored": int(explored),
        "flow_calls": int(flow_calls),

        "block_branches": int(block_branches),
        "fallback_branches": int(fallback_match_branches),

        "certified_optimal": bool(certified_optimal),
        "termination_reason": termination_reason,
        "time_solve_flow": float(time_solve_flow),
    }

    if best_solution is None or best_cost == float("inf"):
        warnings.warn("No feasible solution found.")
        return [], stats


    trajectories_original = []
    for path in best_solution:
        cells = []
        for nid in path:
            if nid in expand_to_original:
                cells.append(expand_to_original[nid])
        cells_unique = list(dict.fromkeys(cells))
        trajectories_original.append(np.array(cells_unique, dtype=int))

    return trajectories_original, stats

def build_base_graph(all_hard_ids,
                     movieInfo1_partial,
                     movieInfo2_partial,
                     matches_hard,
                     edges,
                     debug=False):
    """
    Build the base detection_arcs and transition_arcs graph.
    """

    BIG = 1e3
    n1 = len(movieInfo1_partial["frames"])
    n2 = len(movieInfo2_partial["frames"])
    N = n1 + n2

    threshold = stats.chi2.ppf(1 - 0.01 / N, 1) / 2
    threshold2 = -2 * threshold - 1e-6

    # First, identify and merge identical cells in matches_hard
    # Two cells in a match are considered identical if their vox arrays are the same
    # (vox arrays are n×3 arrays, row order doesn't matter)
    
    matches_to_remove = []
    merged_cells = {}  # Map from removed cell to kept cell
    
    for i, (u, v, *_) in enumerate(matches_hard):
        # Get vox arrays for both cells
        if u < n1:
            vox_u = movieInfo1_partial["vox"][u]
        else:
            vox_u = movieInfo2_partial["vox"][u - n1]
            
        if v < n1:
            vox_v = movieInfo1_partial["vox"][v]
        else:
            vox_v = movieInfo2_partial["vox"][v - n1]
        
        # Quick check: compare lengths first (fast)
        if len(vox_u) != len(vox_v):
            continue
            
        # Detailed check: compare vox arrays (order of rows doesn't matter)
        # Sort rows to make comparison easier
        vox_u_sorted = np.sort(vox_u.view([('', vox_u.dtype)] * vox_u.shape[1]), axis=0).view(vox_u.dtype).reshape(-1, vox_u.shape[1])
        vox_v_sorted = np.sort(vox_v.view([('', vox_v.dtype)] * vox_v.shape[1]), axis=0).view(vox_v.dtype).reshape(-1, vox_v.shape[1])
        
        if np.array_equal(vox_u_sorted, vox_v_sorted):
            # Cells are identical, mark for merging
            matches_to_remove.append(i)
            # choose one
            merged_cells[v] = u

    
    # Remove the marked matches from matches_hard
    if matches_to_remove:
        matches_hard = np.delete(matches_hard, matches_to_remove, axis=0)

    # Update edges by merging cells that were identified as identical
    # For each merged cell, combine its edges with the kept cell
    edges_copy = edges.copy()

    # Convert edges list to adjacency dictionary first
    edges_dict = {}
    for u, v in enumerate(edges_copy):
        if u not in edges_dict:
            edges_dict[u] = set()
            edges_dict[u].update(v)

    # Now use edges_dict instead of edges_copy
    for removed_cell, kept_cell in merged_cells.items():
        if removed_cell in edges_dict and kept_cell in edges_dict:
            # Merge edges: union of both sets, remove duplicates
            edges_dict[kept_cell] = edges_dict[kept_cell].union(edges_dict[removed_cell])
            # Remove self-loop if present
            edges_dict[kept_cell].discard(kept_cell)
            edges_dict[kept_cell].discard(removed_cell)
    
    # Filter all_hard_ids to exclude merged cells
    all_hard_ids_filtered = [cell for cell in all_hard_ids if cell not in merged_cells]

    # Also need to update edges: for any cell that had edges to removed_cell,
    # redirect those edges to kept_cell
    for cell in list(edges_dict.keys()):
        if cell in merged_cells:
            # This cell was removed, skip it
            continue
        
        # Check if this cell has edges to any removed cells
        edges_set = edges_dict[cell].copy()
        for removed_cell, kept_cell in merged_cells.items():
            if removed_cell in edges_set:
                edges_set.remove(removed_cell)
                edges_set.add(kept_cell)
        edges_dict[cell] = edges_set
    
    # Remove entries for merged cells from edges
    for removed_cell in merged_cells.keys():
        if removed_cell in edges_dict:
            del edges_dict[removed_cell]
    
    # Update all_hard_ids to use the filtered list
    all_hard_ids = all_hard_ids_filtered
    edges = edges_dict

    if debug and 'debug_print_base_graph_after_merge' in globals():
        debug_print_base_graph_after_merge(
            all_hard_ids=all_hard_ids,
            edges=edges,
            matches_hard=matches_hard,
            movieInfo1_partial=movieInfo1_partial,
            movieInfo2_partial=movieInfo2_partial,
            merged_cells=merged_cells,
            matches_to_remove=matches_to_remove,
        )

    matched_nodes = set(matches_hard[:, 0]) | set(matches_hard[:, 1])
    unmatched_nodes = [u for u in all_hard_ids if u not in matched_nodes]

    detection_arcs = []
    transition_arcs = []

    node_single = {}
    node_in = {}
    node_out = {}

    # Track control nodes for each match
    control_nodes = {}  # (u, v) -> control_node_id
    expand_to_original = {}
    next_id = 0

    # Unmatched nodes
    for u in unmatched_nodes:
        detection_arcs.append([next_id, threshold, threshold, threshold2])
        node_single[u] = next_id
        expand_to_original[next_id] = u
        next_id += 1

    # First pass: create input and output nodes for all matched cells
    # Ensure each cell has only one input and one output node
    for u, v, *_ in matches_hard:
        uid = u
        vid = v
        
        # Create input node for u if not exists
        if uid not in node_in:
            u_in = next_id
            detection_arcs.append([u_in, threshold, BIG, 0])
            node_in[uid] = u_in
            expand_to_original[u_in] = uid
            next_id += 1
            
            # Create output node for u if not exists
            u_out = next_id
            detection_arcs.append([u_out, BIG, threshold, 0])
            node_out[uid] = u_out
            expand_to_original[u_out] = uid
            next_id += 1
        
        # Create input node for v if not exists
        if vid not in node_in:
            v_in = next_id
            detection_arcs.append([v_in, threshold, BIG, 0])
            node_in[vid] = v_in
            expand_to_original[v_in] = vid
            next_id += 1
            
            # Create output node for v if not exists
            v_out = next_id
            detection_arcs.append([v_out, BIG, threshold, 0])
            node_out[vid] = v_out
            expand_to_original[v_out] = vid
            next_id += 1

   # Second pass: create control nodes for each match and connect them
    for u, v, *_ in matches_hard:
        uid = u
        vid = v
        
        # Create control node for this match
        ctrl = next_id
        detection_arcs.append([ctrl, BIG, BIG, threshold2])
        control_nodes[(uid, vid)] = ctrl
        # expand_to_original[ctrl] = (uid, vid)  # or None since it's a virtual node
        next_id += 1
        
        # Connect input nodes to control node
        transition_arcs.append([node_in[uid], ctrl, 0])
        transition_arcs.append([node_in[vid], ctrl, 0])
        
        # Connect control node to output nodes
        transition_arcs.append([ctrl, node_out[uid], 0])
        transition_arcs.append([ctrl, node_out[vid], 0])


    # Inter-cell edges
    for u in all_hard_ids:
        for v in edges[u]:
            if v not in all_hard_ids:
                continue

            src = node_out[u] if u in node_out else node_single[u]
            dst = node_in[v] if v in node_in else node_single[v]

            if u < n1 and v < n1:
                cost = ovDistanceRegion(movieInfo1_partial["vox"][u],
                                        movieInfo1_partial["vox"][v])[0]
            elif u >= n1 and v >= n1:
                cost = ovDistanceRegion(movieInfo2_partial["vox"][u - n1],
                                        movieInfo2_partial["vox"][v - n1])[0]
            elif u < n1 and v >= n1:
                cost = ovDistanceRegion(movieInfo1_partial["vox"][u],
                                        movieInfo2_partial["vox"][v - n1])[0]
            else:
                cost = ovDistanceRegion(movieInfo2_partial["vox"][u - n1],
                                        movieInfo1_partial["vox"][v])[0]

            transition_arcs.append([src, dst, cost])

    transition_arcs = np.array(transition_arcs)
    detection_arcs = np.array(detection_arcs)

    return detection_arcs, transition_arcs, expand_to_original, node_in, node_out, node_single, matches_hard, threshold

def solve_flow_node(node, matches_hard, node_in_id=None, node_out_id=None):
    try:
        trajectories, cost = mcc4mot(node["detection"], node["transition"])
    except Exception:
        return None, float("inf"), None

    if trajectories is None:
        return None, float("inf"), None

    violations = find_violations(
        trajectories,
        matches_hard,
        node_in_id,
        node_out_id,
    )

    return trajectories, float(np.sum(cost)), violations

def initialize_correction_data_fast(transition_arcs, matches_hard, node_in, node_out, threshold):
    """
    Faster version of initialize_correction_data.

    Original complexity:
        O(num_matches * num_transition_arcs)

    New complexity:
        O(num_transition_arcs + num_matches)
    """
    min_switch_cost = {}
    cost_diff = {}
    u_switched_cost = {}
    v_switched_cost = {}

    # node_id -> min incoming / outgoing transition cost
    incoming_min = defaultdict(lambda: threshold)
    outgoing_min = defaultdict(lambda: threshold)

    # One pass over all transition arcs
    for s, t, cost in transition_arcs:
        s = int(s)
        t = int(t)
        cost = float(cost)

        if cost < outgoing_min[s]:
            outgoing_min[s] = cost
        if cost < incoming_min[t]:
            incoming_min[t] = cost

    for match_id, (u, v, *_) in enumerate(matches_hard):
        u = int(u)
        v = int(v)

        u_in = node_in.get(u, -1)
        u_out = node_out.get(u, -1)
        v_in = node_in.get(v, -1)
        v_out = node_out.get(v, -1)

        u_in_min = incoming_min[u_in]
        u_out_min = outgoing_min[u_out]
        v_in_min = incoming_min[v_in]
        v_out_min = outgoing_min[v_out]

        min_violation_cost = max(
            u_in_min + v_out_min,
            v_in_min + u_out_min
        )

        u_min_cost = u_in_min + u_out_min
        v_min_cost = v_in_min + v_out_min
        min_feasible_cost = min(u_min_cost, v_min_cost)

        raw_switch = min_feasible_cost - min_violation_cost

        min_switch_cost[match_id] = raw_switch
        cost_diff[match_id] = u_min_cost - v_min_cost
        u_switched_cost[match_id] = max(u_min_cost - min_violation_cost, 0.0)
        v_switched_cost[match_id] = max(v_min_cost - min_violation_cost, 0.0)

    correction_data = {
        key: max(value, 0.0)
        for key, value in min_switch_cost.items()
    }

    return min_switch_cost, correction_data, cost_diff, u_switched_cost, v_switched_cost

def compute_corrected_lower_bound(current_cost, violations, correction_data, best_cost):
    """
    Compute corrected lower bound by adding minimum cost to fix violations.
    """
    if not violations:
        return current_cost
    
    # Estimate minimum additional cost to fix all violations
    min_fix_cost = 0
    
    # For each violation, add the minimum switch cost
    for match_id in violations:
        if match_id in correction_data:
            min_fix_cost += correction_data[match_id]
    
    corrected_lb = current_cost + min_fix_cost
    
    # Early pruning: if corrected lower bound already >= best_cost, prune
    if corrected_lb >= best_cost:
        return float('inf')
    
    return corrected_lb

def apply_forbid_many(detection_arcs, transition_arcs, forbid_ids, node_in, node_out):
    """
    Forbid multiple original cells at once.

    This is equivalent to repeated apply_forbid calls, but it filters the
    transition array only once.  It is especially useful for conflict-block
    branching, where choosing one side may forbid several original cells.
    """
    BIG = 1e3

    forbid_ids = [int(x) for x in forbid_ids]
    if len(forbid_ids) == 0:
        return detection_arcs, transition_arcs

    in_out_ids = []
    for fid in dict.fromkeys(forbid_ids):
        if fid not in node_in or fid not in node_out:
            return None, None
        in_out_ids.append(int(node_in[fid]))
        in_out_ids.append(int(node_out[fid]))

    in_out_ids = np.asarray(sorted(set(in_out_ids)), dtype=np.int64)

    ids = detection_arcs[:, 0].astype(np.int64)
    det_mask = np.isin(ids, in_out_ids)
    detection_arcs[det_mask, 1:3] = BIG

    if transition_arcs is None or len(transition_arcs) == 0:
        return detection_arcs, transition_arcs

    src = transition_arcs[:, 0].astype(np.int64)
    dst = transition_arcs[:, 1].astype(np.int64)
    bad_src = np.isin(src, in_out_ids)
    bad_dst = np.isin(dst, in_out_ids)
    keep = ~(bad_src | bad_dst)

    return detection_arcs, transition_arcs[keep]

def apply_forbid(detection_arcs, transition_arcs, forbid_id, node_in, node_out):
    """
    Forbid one original cell by disabling its in/out detection nodes and
    removing all incident transition arcs.
    """
    return apply_forbid_many(
        detection_arcs,
        transition_arcs,
        [int(forbid_id)],
        node_in,
        node_out,
    )

def find_violations(trajectories, matches_hard, node_in_id, node_out_id):
    """
    Find which hard matches violate the exclusivity constraint.
    New definition: A match (u, v) is violated if both u and v appear in trajectories.
    
    Parameters:
    -----------
    trajectories : list of lists
        List of trajectories, each trajectory is a list of node IDs in the constructed graph
    matches_hard : array
        Array of hard matches, each row is [u, v, ...]
    node_in_id : dict
        Mapping from original cell ID to its input node ID in the graph
    node_out_id : dict
        Mapping from original cell ID to its output node ID in the graph
        
    Returns:
    --------
    violations : list
        Indices of matches_hard that are violated
    """
    
    # Create reverse mapping: from graph node ID to original cell ID
    # For cells that appear in matches_hard, they have both input and output nodes
    node_to_cell = {}
    for cell_id, in_node in node_in_id.items():
        node_to_cell[in_node] = cell_id
    for cell_id, out_node in node_out_id.items():
        node_to_cell[out_node] = cell_id
    
    # Track which cells appear in any trajectory
    appeared_cells = set()
    
    # Process each trajectory
    for path in trajectories:
        for node_id in path:
            # Check if this node maps to an original cell
            if node_id in node_to_cell:
                cell_id = node_to_cell[node_id]
                appeared_cells.add(cell_id)
    
    # Find matches where both cells appear in trajectories
    violations = []
    for i, (u, v, *_) in enumerate(matches_hard):
        if u in appeared_cells and v in appeared_cells:
            violations.append(i)
    
    return violations

def mcc4mot(detection_arcs, transition_arcs):
    _cinda = get_cinda_solver()

    mtail, mhead, mlow, macap, mcost, msz = cinda_data_process(
        detection_arcs,
        transition_arcs,
    )

    scale = 10 ** 7
    mcost = np.rint(mcost * scale).astype(np.float64, copy=False)

    inf_type = ctypes.c_long * len(msz)
    double_ptr = ctypes.POINTER(ctypes.c_double)

    track_vec = _cinda.pyCS2(
        inf_type(*msz),
        mtail.ctypes.data_as(double_ptr),
        mhead.ctypes.data_as(double_ptr),
        mlow.ctypes.data_as(double_ptr),
        macap.ctypes.data_as(double_ptr),
        mcost.ctypes.data_as(double_ptr),
    )

    cost = []
    traj = []
    sub_traj = []

    for i in range(1, track_vec[0] + 1):
        if track_vec[i] > 0:
            sub_traj.append(track_vec[i])
        else:
            cost.append(float(track_vec[i]) / scale)
            new = [int(x / 2) for x in sub_traj[::2]]
            traj.append(np.array(new, dtype=int) - 1)
            sub_traj = []

    return traj, cost

def cinda_data_process(detection_arcs, transition_arcs):
    detection_arcs = np.asarray(detection_arcs, dtype=np.float64)
    transition_arcs = np.asarray(transition_arcs, dtype=np.float64)

    n_detection = detection_arcs.shape[0]
    n_transition = transition_arcs.shape[0]
    n_arcs = n_detection * 3 + n_transition

    mtail = np.empty(n_arcs, dtype=np.float64)
    mhead = np.empty(n_arcs, dtype=np.float64)
    mlow = np.zeros(n_arcs, dtype=np.float64)
    macap = np.ones(n_arcs, dtype=np.float64)
    mcost = np.empty(n_arcs, dtype=np.float64)

    # 这里直接转成 C solver 需要的 1-based detection id
    det_id = detection_arcs[:, 0] + 1

    a = 0
    b = n_detection
    mtail[a:b] = 1
    mhead[a:b] = det_id * 2
    mcost[a:b] = detection_arcs[:, 1]

    a = b
    b = a + n_detection
    mtail[a:b] = det_id * 2 + 1
    mhead[a:b] = 1
    mcost[a:b] = detection_arcs[:, 2]

    a = b
    b = a + n_detection
    mtail[a:b] = det_id * 2
    mhead[a:b] = det_id * 2 + 1
    mcost[a:b] = detection_arcs[:, 3]

    a = b
    b = a + n_transition
    if n_transition > 0:
        src_id = transition_arcs[:, 0] + 1
        dst_id = transition_arcs[:, 1] + 1
        mtail[a:b] = src_id * 2 + 1
        mhead[a:b] = dst_id * 2
        mcost[a:b] = transition_arcs[:, 2]

    msz = [12, int(2 * n_detection + 1), int(n_arcs)]
    return mtail, mhead, mlow, macap, mcost, msz

def ovDistanceRegion(curRegVox: np.ndarray,
                     nextRegVox: np.ndarray,
                     frame_shift: Optional[np.ndarray] = None,
                     ovFlag: bool = False) -> Tuple[float, float, np.ndarray, float]:
    """
    Calculate region-to-region distance.

    Default behavior:
      - 3D vox: use EDT distance, same as before.
      - 2D vox: lift [y, x] to [z=0, y, x], then use the same EDT distance.
    
    If ovFlag=True, explicitly use old overlap/Jaccard cost.
    """

    curRegVox = np.asarray(curRegVox)
    nextRegVox = np.asarray(nextRegVox)

    if curRegVox.ndim != 2 or nextRegVox.ndim != 2:
        raise ValueError(
            f"vox must be 2D arrays, got {curRegVox.shape}, {nextRegVox.shape}"
        )

    if curRegVox.shape[1] != nextRegVox.shape[1]:
        raise ValueError(
            f"vox dimension mismatch: {curRegVox.shape} vs {nextRegVox.shape}"
        )

    dim = curRegVox.shape[1]
    re_ratio = 0.0

    if frame_shift is None:
        frame_shift = np.zeros(dim, dtype=float)
    else:
        frame_shift = np.asarray(frame_shift, dtype=float)

    # ------------------------------------------------------------
    # Optional old overlap cost.
    # Only used when caller explicitly passes ovFlag=True.
    # ------------------------------------------------------------
    if ovFlag:
        cur_set = set(tuple(pt) for pt in curRegVox)
        next_set = set(tuple(pt) for pt in nextRegVox)
        intersection = cur_set.intersection(next_set)

        overlap_ratio = len(intersection) / (
            len(cur_set) + len(next_set) - len(intersection)
        )

        distances = -np.log(overlap_ratio) if overlap_ratio > 0 else 1e4
        distances = np.array([distances, distances], dtype=float)
        re_ratio = overlap_ratio

        maxDistance = float(np.max(distances))
        minDistance = float(np.min(distances))
        return maxDistance, minDistance, distances, re_ratio

    # ------------------------------------------------------------
    # Key change:
    # 2D [y, x] -> pseudo-3D [z=0, y, x]
    # Then it goes through exactly the same EDT branch as 3D.
    # ------------------------------------------------------------
    if dim == 2:
        z_cur = np.zeros((curRegVox.shape[0], 1), dtype=curRegVox.dtype)
        z_next = np.zeros((nextRegVox.shape[0], 1), dtype=nextRegVox.dtype)

        curRegVox = np.concatenate([z_cur, curRegVox], axis=1)
        nextRegVox = np.concatenate([z_next, nextRegVox], axis=1)

        # Original 2D shift is [y, x].
        # After lifting to [z, y, x], shift becomes [0, y, x].
        frame_shift = np.array([0.0, frame_shift[0], frame_shift[1]], dtype=float)
        dim = 3

    if dim != 3:
        raise ValueError(f"Unsupported vox dimension: {dim}")

    # ------------------------------------------------------------
    # Same EDT logic as your original 3D branch.
    # ------------------------------------------------------------
    if nextRegVox.shape[0] < 2 or curRegVox.shape[0] < 2:
        distances = np.array([100.0, 100.0], dtype=float)
        maxDistance = float(np.max(distances))
        minDistance = float(np.min(distances))
        return maxDistance, minDistance, distances, re_ratio

    st_pt1 = np.min(curRegVox, axis=0)
    end_pt1 = np.max(curRegVox, axis=0)
    bw_sz1 = np.ceil(end_pt1 - st_pt1 + 1).astype(int)

    mask1 = np.zeros(bw_sz1, dtype=bool)
    cell1_sub = (curRegVox - st_pt1).astype(int)
    mask1[tuple(cell1_sub.T)] = True

    st_pt2 = np.min(nextRegVox, axis=0)
    end_pt2 = np.max(nextRegVox, axis=0)
    bw_sz2 = np.ceil(end_pt2 - st_pt2 + 1).astype(int)

    mask2 = np.zeros(bw_sz2, dtype=bool)
    cell2_sub = (nextRegVox - st_pt2).astype(int)
    mask2[tuple(cell2_sub.T)] = True

    mov_shift = (st_pt2 - st_pt1 - frame_shift).astype(float)

    dist2cell1 = edt_3d(mask1, mask2, mov_shift)
    dist2cell2 = edt_3d(mask2, mask1, -mov_shift)

    distances_n2c_way5 = dist2cell1[tuple(cell2_sub.T)]
    distances_c2n_way5 = dist2cell2[tuple(cell1_sub.T)]

    distances_n2c_way5 = np.sqrt(np.asarray(distances_n2c_way5))
    distances_c2n_way5 = np.sqrt(np.asarray(distances_c2n_way5))

    distances = np.array([
        np.mean(distances_c2n_way5),
        np.mean(distances_n2c_way5),
    ], dtype=float)

    boxSize = np.prod(bw_sz1) + np.prod(bw_sz2)
    redundant_sz = boxSize - len(cell2_sub)
    re_ratio = redundant_sz / boxSize

    maxDistance = float(np.max(distances))
    minDistance = float(np.min(distances))

    return maxDistance, minDistance, distances, re_ratio

def edt_3d(ref_cell, mov_cell, shift):
    """
    Compute EDT using the existing C++ edt_3d.

    Supports:
      - 3D mask: used directly.
      - 2D mask: lifted to one-slice 3D mask.
    """

    ref_cell = np.asarray(ref_cell, dtype=np.bool_).astype(np.uint8)
    mov_cell = np.asarray(mov_cell, dtype=np.bool_).astype(np.uint8)
    shift = np.asarray(shift, dtype=np.float32)

    squeeze_2d = False

    if ref_cell.ndim == 2:
        if mov_cell.ndim != 2:
            raise ValueError(
                f"ref_cell is 2D but mov_cell is {mov_cell.ndim}D"
            )

        # Current Python-side convention:
        # 2D mask [y, x] -> pseudo-3D mask [z=0, y, x]
        ref_cell = ref_cell[None, :, :]
        mov_cell = mov_cell[None, :, :]

        # shift [y, x] -> [z, y, x]
        if shift.size == 2:
            shift = np.array([0.0, shift[0], shift[1]], dtype=np.float32)
        elif shift.size != 3:
            raise ValueError(f"2D shift must have size 2 or 3, got {shift}")

        squeeze_2d = True

    elif ref_cell.ndim == 3:
        if mov_cell.ndim != 3:
            raise ValueError(
                f"ref_cell is 3D but mov_cell is {mov_cell.ndim}D"
            )

        if shift.size != 3:
            raise ValueError(f"3D shift must have size 3, got {shift}")

    else:
        raise ValueError(f"Unsupported ref_cell.ndim={ref_cell.ndim}")

    ref_cell = np.ascontiguousarray(ref_cell)
    mov_cell = np.ascontiguousarray(mov_cell)
    shift = np.ascontiguousarray(shift.astype(np.float32))

    ref_dims = np.array(ref_cell.shape, dtype=np.int32)
    mov_dims = np.array(mov_cell.shape, dtype=np.int32)

    lib = get_edt3d_lib()
    output = np.zeros(mov_cell.shape, dtype=np.float32)

    lib.edt_3d(
        ref_cell,
        ref_dims,
        3,
        mov_cell,
        mov_dims,
        3,
        shift,
        output,
    )

    if squeeze_2d:
        return output[0]

    return output

def infer_vox_dim(movieInfo1_partial, movieInfo2_partial):
    for mi in (movieInfo1_partial, movieInfo2_partial):
        for vox in mi.get("vox", []):
            vox = np.asarray(vox)
            if vox.ndim == 2 and vox.shape[0] > 0:
                return int(vox.shape[1])
    return 3

def get_global_vox(global_id, n1, movieInfo1_partial, movieInfo2_partial):
    global_id = int(global_id)

    if global_id < n1:
        return movieInfo1_partial["vox"][global_id]
    else:
        return movieInfo2_partial["vox"][global_id - n1]

def recompute_2d_transition_costs_by_edt(
    transition_arcs,
    movieInfo1_partial,
    movieInfo2_partial,
    verbose=True,
):
    """
    Recompute real-node transition costs for 2D data using EDT principle.
    Pseudo/control-node arcs are not changed.
    """

    vox_dim = infer_vox_dim(movieInfo1_partial, movieInfo2_partial)

    if vox_dim != 2:
        return transition_arcs

    n1 = len(movieInfo1_partial["frames"])
    n2 = len(movieInfo2_partial["frames"])
    N = n1 + n2

    transition_arcs = np.ascontiguousarray(
        transition_arcs,
        dtype=np.float64,
    ).copy()

    changed = 0
    real_arcs = 0

    for k in range(len(transition_arcs)):
        u = int(transition_arcs[k, 0])
        v = int(transition_arcs[k, 1])

        # Skip pseudo/control nodes.
        if u >= N or v >= N:
            continue

        vox_u = get_global_vox(
            u,
            n1,
            movieInfo1_partial,
            movieInfo2_partial,
        )
        vox_v = get_global_vox(
            v,
            n1,
            movieInfo1_partial,
            movieInfo2_partial,
        )

        old_cost = float(transition_arcs[k, 2])
        new_cost = float(ovDistanceRegion(vox_u, vox_v, ovFlag=False)[0])

        transition_arcs[k, 2] = new_cost

        if not np.isclose(old_cost, new_cost):
            changed += 1

        real_arcs += 1

    if verbose:
        print(
            f"[2D-EDT-COST] recomputed transition costs: "
            f"real_arcs={real_arcs}, changed={changed}"
        )

    return transition_arcs

def create_fused_movieInfo(tracks, movieInfo1, movieInfo2, movieInfo1_partial, movieInfo2_partial, idmap1=None, idmap2=None, keep_parent_idx=None, crop_area=[], tm_shift = 0, img_shape = None, seg_indice = 'python'):
    """
    Create a fused movieInfo by combining information from movieInfo1 and movieInfo2 based on tracking results.
    
    Args:
        tracks (list): List of trajectories, each trajectory is an array of cell IDs
        movieInfo1 (dict): Full movieInfo for first dataset with keys: xCoord, yCoord, zCoord, frames, vox, voxIdx, parents, perframe
        movieInfo2 (dict): Full movieInfo for second dataset with same structure as movieInfo1
        movieInfo1_partial (dict): Partial movieInfo for first dataset corresponding to tracks
        movieInfo2_partial (dict): Partial movieInfo for second dataset corresponding to tracks
        idmap1 (dict, optional): Mapping from full ID in movieInfo1 to partial ID in movieInfo1_partial
        idmap2 (dict, optional): Mapping from full ID in movieInfo2 to partial ID in movieInfo2_partial
        keep_parent_idx (list, optional): List of indices of parent cells to keep from both datasets
    Returns:
        dict: Fused movieInfo_new with updated parents and sorted by frames
    """
    
    # Initialize result structure
    # ============================================================
    # Recover coordinates for cropped partial movieInfo
    # ============================================================
    if crop_area is not None and len(crop_area) > 0:

        crop_dim = len(crop_area)

        # -------------------------
        # 2D case
        # -------------------------
        if crop_dim == 2:
            y0 = crop_area[0][0]
            x0 = crop_area[1][0]

            if 'xCoord' in movieInfo1_partial:
                movieInfo1_partial['xCoord'] = movieInfo1_partial['xCoord'] + x0
            if 'yCoord' in movieInfo1_partial:
                movieInfo1_partial['yCoord'] = movieInfo1_partial['yCoord'] + y0

            if 'xCoord' in movieInfo2_partial:
                movieInfo2_partial['xCoord'] = movieInfo2_partial['xCoord'] + x0
            if 'yCoord' in movieInfo2_partial:
                movieInfo2_partial['yCoord'] = movieInfo2_partial['yCoord'] + y0

            offset_vox = np.array([y0, x0], dtype=np.asarray(movieInfo1_partial['vox'][0]).dtype)

            movieInfo1_partial['vox'] = [
                np.asarray(vox) - offset_vox
                for vox in movieInfo1_partial['vox']
            ]

            movieInfo2_partial['vox'] = [
                np.asarray(vox) + offset_vox
                for vox in movieInfo2_partial['vox']
            ]

        # -------------------------
        # 3D case
        # -------------------------
        elif crop_dim == 3:
            z0 = crop_area[0][0]
            y0 = crop_area[1][0]
            x0 = crop_area[2][0]

            if 'xCoord' in movieInfo1_partial:
                movieInfo1_partial['xCoord'] = movieInfo1_partial['xCoord'] + x0
            if 'yCoord' in movieInfo1_partial:
                movieInfo1_partial['yCoord'] = movieInfo1_partial['yCoord'] + y0
            if 'zCoord' in movieInfo1_partial:
                movieInfo1_partial['zCoord'] = movieInfo1_partial['zCoord'] + z0

            if 'xCoord' in movieInfo2_partial:
                movieInfo2_partial['xCoord'] = movieInfo2_partial['xCoord'] + x0
            if 'yCoord' in movieInfo2_partial:
                movieInfo2_partial['yCoord'] = movieInfo2_partial['yCoord'] + y0
            if 'zCoord' in movieInfo2_partial:
                movieInfo2_partial['zCoord'] = movieInfo2_partial['zCoord'] + z0

            # vox order is [z, y, x]
            offset_vox = np.array([z0, y0, x0], dtype=np.asarray(movieInfo1_partial['vox'][0]).dtype)

            movieInfo1_partial['vox'] = [
                np.asarray(vox) - offset_vox
                for vox in movieInfo1_partial['vox']
            ]

            movieInfo2_partial['vox'] = [
                np.asarray(vox) + offset_vox
                for vox in movieInfo2_partial['vox']
            ]

        else:
            raise ValueError(
                f"Unsupported crop_area dimension: len(crop_area)={crop_dim}, crop_area={crop_area}"
            )
            
    movieInfo1['frames'] = movieInfo1['frames'] + tm_shift
    movieInfo2['frames'] = movieInfo2['frames'] + tm_shift
    movieInfo1_partial['frames'] = movieInfo1_partial['frames'] + tm_shift
    movieInfo2_partial['frames'] = movieInfo2_partial['frames'] + tm_shift

    movieInfo_new = {
        'xCoord': [],
        'yCoord': [],
        'zCoord': [],
        'frames': [],
        'vox': [],
        'voxIdx': [],
        'parents': [],
        'perframe': []
    }
    
    # Track the mapping from old IDs to new IDs
    old_to_new_id = {}
    
    # Counter for new IDs
    new_id_counter = 0
    
    # Helper function to add a cell to movieInfo_new
    def add_cell(cell_data, parent_id=-1):
        nonlocal new_id_counter
        for key in ['xCoord', 'yCoord', 'zCoord', 'frames', 'vox', 'voxIdx']:
            movieInfo_new[key].append(cell_data[key])
        movieInfo_new['parents'].append(parent_id)
        # Handle perframe if exists
        if 'perframe' in cell_data and cell_data['perframe']:
            movieInfo_new['perframe'].append(cell_data['perframe'])
        else:
            movieInfo_new['perframe'].append([])
        return new_id_counter
    
    if idmap1 is None or idmap2 is None:
        # Process tracks and extract cells from partial movieInfos
        n1 = len(movieInfo1_partial['frames'])
        
        for track in tracks:
            prev_new_id = -1
            for i, cell_id in enumerate(track):
                if cell_id < n1:
                    # Cell from movieInfo1_partial
                    cell_data = {key: movieInfo1_partial[key][cell_id] for key in 
                                ['xCoord', 'yCoord', 'zCoord', 'frames', 'vox', 'voxIdx']}
                else:
                    # Cell from movieInfo2_partial (adjust index)
                    adjusted_id = cell_id - n1
                    cell_data = {key: movieInfo2_partial[key][adjusted_id] for key in 
                                ['xCoord', 'yCoord', 'zCoord', 'frames', 'vox', 'voxIdx']}
                
                # Add to new movieInfo
                parent_id = -1 if i == 0 else prev_new_id
                new_id = add_cell(cell_data, parent_id)
                
                # Store mapping and update for next cell in track
                old_to_new_id[cell_id] = new_id
                prev_new_id = new_id
                new_id_counter += 1
                
    else:
        # =================== temporal or spatial mode ===================
        # Create reverse mappings from partial ID to full ID
        rev_idmap1 = {v: k for k, v in idmap1.items()}  # partial_id -> full_id
        rev_idmap2 = {v: k for k, v in idmap2.items()}  # partial_id -> full_id
        
        # First, collect all cells that appear in tracks (through partial movieInfos)
        cells_in_tracks = set()
        n1 = len(movieInfo1_partial['frames'])
        n1_all = len(movieInfo1['frames'])
        # Map from full ID to its track parent (full ID)
        full_id_parent_map = {}


        for track in tracks:
            prev_full_id = None
            for i, cell_id in enumerate(track):
                if cell_id < n1:
                    # Cell from movieInfo1_partial
                    full_id = rev_idmap1[cell_id]
                    cells_in_tracks.add((full_id, 'm1'))
                else:
                    # Cell from movieInfo2_partial
                    full_id = rev_idmap2[cell_id - n1] + n1_all
                    cells_in_tracks.add((full_id, 'm2'))
                               
                # Map parent relationship
                parent_full_id = -1
                if i > 0 and prev_full_id is not None:
                    parent_full_id = prev_full_id
                full_id_parent_map[full_id] = parent_full_id
                
                prev_full_id = full_id
        
        # Add cells from movieInfo1 that are NOT in the partial set
        for full_id in range(len(movieInfo1['frames'])):
            if full_id not in idmap1:  # Not in partial, so keep as-is
                cell_data = {key: movieInfo1[key][full_id] for key in 
                            ['xCoord', 'yCoord', 'zCoord', 'frames', 'vox', 'voxIdx']}
                # Copy original parent (adjust later if needed)
                parent_id = movieInfo1['parents'][full_id]
                new_id = add_cell(cell_data, parent_id)
                old_to_new_id[full_id] = new_id
                new_id_counter += 1
        
        # Add cells from movieInfo2 that are NOT in the partial set
        for full_id in range(len(movieInfo2['frames'])):
            if full_id not in idmap2:  # Not in partial, so keep as-is
                cell_data = {key: movieInfo2[key][full_id] for key in 
                            ['xCoord', 'yCoord', 'zCoord', 'frames', 'vox', 'voxIdx']}
                # Copy original parent (adjust later if needed)
                parent_id = -1 if movieInfo2['parents'][full_id] == -1 else movieInfo2['parents'][full_id] + n1_all
                new_id = add_cell(cell_data, parent_id)
                old_to_new_id[full_id + n1_all] = new_id
                new_id_counter += 1
        
        # Add cells that appear in tracks (from partial movieInfos)
        for full_id, marker in cells_in_tracks:
            if marker == 'm1':
                cell_data = {key: movieInfo1[key][full_id] for key in 
                            ['xCoord', 'yCoord', 'zCoord', 'frames', 'vox', 'voxIdx']}
                parent_id = full_id_parent_map[full_id]
                new_id = add_cell(cell_data, parent_id)
                old_to_new_id[full_id] = new_id
            else:
                cell_data = {key: movieInfo2[key][full_id-n1_all] for key in 
                            ['xCoord', 'yCoord', 'zCoord', 'frames', 'vox', 'voxIdx']}
                parent_id = full_id_parent_map[full_id]
                new_id = add_cell(cell_data, parent_id)
                old_to_new_id[full_id] = new_id

            new_id_counter += 1
        for idx in keep_parent_idx:
            if idx < n1_all:
                movieInfo_new['parents'][old_to_new_id[idx]] = movieInfo1['parents'][idx]
            else:
                movieInfo_new['parents'][old_to_new_id[idx]] = movieInfo2['parents'][idx - n1_all] + n1_all
        movieInfo_new['parents'] = [old_to_new_id.get(p, -1) if p != -1 else -1 for p in movieInfo_new['parents']]

    # Now we need to sort by frames while keeping all information synchronized
    # Get indices sorted by frame value
    sorted_indices = np.argsort(movieInfo_new['frames'])
    old_to_new_id_map = np.empty_like(sorted_indices)
    old_to_new_id_map[sorted_indices] = np.arange(len(sorted_indices))

    for key in ['xCoord', 'yCoord', 'zCoord', 'frames','parents']:
        if key in movieInfo_new:
            arr = movieInfo_new[key]
            movieInfo_new[key] = np.array([arr[i] for i in sorted_indices])

    for key in ['vox']:
        if key in movieInfo_new:
            arr = movieInfo_new[key]
            movieInfo_new[key] = [arr[i] for i in sorted_indices]
    for key in ['voxIdx']:
        arr = movieInfo_new[key]
        
        if len(crop_area) > 0 and key == 'voxIdx':
            # When crop_area exists, reconstruct voxIdx from vox coordinates
            # Get the vox coordinates (which are correct)
            vox_coords_list = movieInfo_new['vox']
            
            converted_indices = []
            for vox_coords in vox_coords_list:
                if len(img_shape) == 3:
                    # 3D case: vox coordinate is [z, y, x] (Python format)
                    z, y, x = vox_coords[:,0], vox_coords[:,1], vox_coords[:,2]
                    if z.max() >= img_shape[0] or y.max() >= img_shape[1] or x.max() >= img_shape[2]:
                        print(f"Warning: Coordinates out of bounds detected!")
                        print(vox_coords)
                    if seg_indice == 'matlab':
                        flat_idx = np.ravel_multi_index((z, x, y), (img_shape[0], img_shape[2], img_shape[1])) + 1  # +1 for 1-based indexing
                    elif seg_indice == 'python':
                        flat_idx = np.ravel_multi_index((z, y, x), img_shape)
                    converted_indices.append(flat_idx)
                        
                elif len(img_shape) == 2:
                    # 2D case: vox coordinate is [y, x] (Python format)
                    y, x = vox_coords[:,0], vox_coords[:,1]
                    if seg_indice == 'matlab':
                        flat_idx = np.ravel_multi_index((x, y), (img_shape[1], img_shape[0]), order='F') + 1  # +1 for 1-based indexing
                    elif seg_indice == 'python':
                        flat_idx = np.ravel_multi_index((y, x), img_shape)
                    converted_indices.append(flat_idx)
                    
                else:
                    # Keep original if dimensions not recognized
                    converted_indices.append(arr[0])  # Fallback
            
            # Reorder the converted indices according to sorted_indices
            movieInfo_new[key] = [converted_indices[i] for i in sorted_indices]
        else:
            # Default behavior: simply reorder existing values
            movieInfo_new[key] = [arr[i] for i in sorted_indices]
    new_parents = []
    for p in movieInfo_new['parents']:
        if p >= 0:
            new_parents.append(old_to_new_id_map[p])
        else:
            new_parents.append(p)
    movieInfo_new['parents'] = new_parents



    return movieInfo_new