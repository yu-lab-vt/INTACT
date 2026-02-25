import numpy as np
import pandas as pd
import copy
import os
from typing import Dict, List, Tuple, Any, Set, Optional
import json
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'C_package'))
import ctypes
from scipy import stats
import warnings
from collections import deque, defaultdict
import heapq
import ctypes

def solve_fusion(movieInfo1_partial, movieInfo2_partial, matches):
    """
    Efficient pre-pruning for very large graphs.
    Build joint graph, extract connected components,
    and classify easy vs hard subgraphs.

    This function also builds detection_arcs and transition_arcs for easy subgraphs
    and uses the mcc4mot function to solve the problem.

    Returns
    -------
    selected_cells : list of tuples
        Each tuple contains (cell_id, parent_id) for selected cells.
    """
    print("\nStep 4-1: Building subgraphs...")
    easy_subgraphs, hard_subgraphs, edges = build_subgraphs(movieInfo1_partial, movieInfo2_partial, matches)
    
    print("\nStep 4-2: Solving easy subgraphs...")
    # pruning and solve easy subgraphs
    tracks_easy,_ = pruning(easy_subgraphs, movieInfo1_partial, movieInfo2_partial, edges)
    matches_arr = np.array(matches)
    matches_arr[:,1] = matches_arr[:,1] + len(movieInfo1_partial['frames'])
    matching_rows = [i for i, row in enumerate(matches_arr) 
                if all(idx in np.concatenate(tracks_easy) for idx in row)]
    if len(matching_rows) > 0:
        warnings.warn(f"Warning: Some matches are included in easy subgraphs. Rows: {matching_rows}")
    
    print("\nStep 4-3: Solving hard subgraphs...")
    tracks_hard = []

    for subgraph_nodes in hard_subgraphs:
        current_ids = set(subgraph_nodes)
        mask = np.isin(matches_arr[:, 0], list(current_ids))
        matches_hard = matches_arr[mask]
        subtracks_hard = solve_hard_fusion(current_ids, movieInfo1_partial, movieInfo2_partial, matches_hard, edges)
        tracks_hard.extend(subtracks_hard)
    final_tracks = tracks_easy + tracks_hard

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

def solve_hard_fusion(hard_ids,
                      movieInfo1_partial,
                      movieInfo2_partial,
                      matches_hard,
                      edges,
                      max_bb_nodes=1000000):
    """
    Branch-and-Bound + Min-Cost Flow for hard fusion.

    Parameters
    ----------
    all_hard_ids : list[int]
        Node IDs in merged index space
    movieInfo1_partial, movieInfo2_partial : dict
        Partial movieInfo structures
    matches_hard : array-like, shape (K, 2 or 3)
        Matched node pairs (merged IDs)
    edges : list[list[int]]
        Adjacency list in merged graph
    max_bb_nodes : int
        Maximum number of B&B nodes to explore

    Returns
    -------
    best_trajectories_original : list[list[int]]
        Final trajectories in original ID space
    """

    # Step 0: build base flow graph
    base_detection_arcs,base_transition_arcs,expand_to_original, \
        node_in, node_out, node_single, matches_hard, threshold = build_base_graph(
        hard_ids, movieInfo1_partial, movieInfo2_partial, matches_hard, edges)

    # Branch-and-bound structures, Priority queue: (lower_bound, counter, node)
    best_cost = float("inf")
    best_solution = None
    flow_cache = {}

    pq = []
    counter = 0

    root = {"fixed": {},  # match_id -> 0 or 1
        "detection": base_detection_arcs,
        "transition": base_transition_arcs}

    root_sol, root_cost, root_violations = solve_flow_node(root, matches_hard, node_in, node_out)

    # Step 1: Generate initial feasible solution by fixing violations in root solution
    if root_sol is not None and len(root_violations) > 0:
        initial_feasible_sol, initial_cost = repair_to_feasible(root_sol, root_violations, matches_hard, 
                                                  node_in, node_out, base_detection_arcs, 
                                                  base_transition_arcs)
        if initial_feasible_sol is not None and initial_cost < best_cost:
            best_cost = initial_cost
            best_solution = initial_feasible_sol
    
    # Initialize lower bound correction data structure
    min_switch_cost, correction_data, cost_diff, u_switch_cost, v_switch_cost = initialize_correction_data(base_transition_arcs, matches_hard, 
                                                node_in, node_out, threshold)
    
    # Compute initial lower bound with correction
    corrected_root_lb = compute_corrected_lower_bound(root_cost, root_violations, 
                                                     correction_data, best_cost)
    
    heapq.heappush(pq, (corrected_root_lb, counter, root, root_violations))
    counter += 1

    explored = 0

    # Main B&B loop
    while pq and explored < max_bb_nodes:
        lb, _, node,_ = heapq.heappop(pq)
        explored += 1

        # Bounding
        if lb >= best_cost:
            continue
        key = tuple(sorted(node["fixed"].items()))
        if key in flow_cache:
            sol, cost, violations = flow_cache[key]
        else:
            sol, cost, violations = solve_flow_node(node, matches_hard, node_in, node_out)
            flow_cache[key] = (sol, cost, violations)

        if sol is None:
            continue

        # If no violation, we have a feasible solution
        if len(violations) == 0:
            if lb < best_cost:
                best_cost = lb
                best_solution = sol
            continue

        # Branch on the first violated match
        # match_id = violations[0]
        match_id = max(violations, key=lambda m: min_switch_cost.get(m, 0.0))
        u, v = matches_hard[match_id][:2]

        u_cost = u_switch_cost[match_id] + cost
        v_cost = v_switch_cost[match_id] + cost
        best = best_cost
        
        if u_cost >= best and v_cost >= best:
            continue
        elif u_cost >= best:
            choices = [0]
        elif v_cost >= best:
            choices = [1]
        else:
            choices = [0, 1] if cost_diff[match_id] > 0 else [1, 0]

        # Two branches: forbid u OR forbid v
        for choice in choices:
            forbid_node = v if choice == 0 else u

            child = {
                "fixed": dict(node["fixed"]),
                "detection": node["detection"].copy(),
                "transition": node["transition"].copy()
            }
            child["fixed"][match_id] = choice

            child['detection'], child['transition'] = apply_forbid(child['detection'], child['transition'], forbid_node, node_in, node_out)
            child_sol, child_cost, child_violations= solve_flow_node(child, matches_hard, node_in, node_out)
            key = tuple(sorted(child["fixed"].items()))
            flow_cache[key] = (child_sol, child_cost, child_violations)
            
            # Compute corrected lower bound for child
            child_lb = compute_corrected_lower_bound(child_cost, child_violations, 
                                                    correction_data, best_cost)
            
            heapq.heappush(pq, (child_lb, counter, child, child_violations))
            counter += 1

    # Recover original trajectories
    if best_solution is None or best_cost == float("inf"):
        warnings.warn('No feasible solution found.')
        return []

    trajectories_original = []
    for path in best_solution:
        cells = []
        for nid in path:
            if nid in expand_to_original:
                cells.append(expand_to_original[nid])
        cells_unique = list(dict.fromkeys(cells))
        trajectories_original.append(np.array(cells_unique, dtype=int))

    return trajectories_original

def repair_to_feasible(sol, violations, matches_hard, node_in, node_out, 
                       detection_arcs, transition_arcs):
    """
    Repair a solution with violations to a feasible solution.
    
    Strategy: Keep all nodes that already satisfy exclusivity constraint,
    and force fix the violated matches by selecting one node from each pair.
    """
    # Identify nodes that already satisfy exclusivity
    satisfied_matches = set()
    violated_matches = set(violations)
    forced_choices = {}

    # For all matches that are not violated, both nodes are already exclusive
    # First, record the current choices for all matches that are not violated
    for i in range(len(matches_hard)):
        if i not in violations:
            u, v = matches_hard[i][:2]
            # Check which node is currently selected in the solution
            u_selected = False
            v_selected = False
            
            # Check if u is selected (in or out node appears in trajectories)
            for path in sol:
                if node_in[u] in path or node_out[u] in path:
                    u_selected = True
                if node_in[v] in path or node_out[v] in path:
                    v_selected = True
            
            # Record the current choice
            if u_selected and not v_selected:
                forced_choices[i] = 0  # Choose u (forbid v)
            elif not u_selected and v_selected:
                forced_choices[i] = 1  # Choose v (forbid u)
            else:
                # This should not happen for non-violated matches
                continue
                
            satisfied_matches.add(i)
    
    # Reconstruct the flow graph with forced choices
    repaired_detection = detection_arcs.copy()
    repaired_transition = transition_arcs.copy()
    
    # First apply fixes for already satisfied matches
    for match_id in satisfied_matches:
        u, v = matches_hard[match_id][:2]
        if forced_choices[match_id] == 0:
            # Choose u, forbid v
            repaired_detection, repaired_transition = apply_forbid(repaired_detection, repaired_transition, v,
                        node_in, node_out)
        else:
            # Choose v, forbid u
            repaired_detection, repaired_transition = apply_forbid(repaired_detection, repaired_transition, u,
                        node_in, node_out)
            
    for match_id in violations:
        u, v = matches_hard[match_id][:2]
        # forbid u
        repaired_detection, repaired_transition = apply_forbid(repaired_detection, repaired_transition, v,
                        node_in, node_out)
        forced_choices[match_id] = 0  # 0 means choose u (forbid v)
    
    # Create repaired node
    repaired_node = {
        "fixed": forced_choices,
        "detection": repaired_detection,
        "transition": repaired_transition
    }
    
    # Solve the repaired flow problem
    repaired_sol, repaired_cost, repaired_violations = solve_flow_node(
        repaired_node, matches_hard, node_in, node_out)
    
    if repaired_sol is not None and len(repaired_violations) == 0:
        return repaired_sol, repaired_cost
    else:
        return None, None
    
def build_base_graph(all_hard_ids,
                     movieInfo1_partial,
                     movieInfo2_partial,
                     matches_hard,
                     edges):
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
    """
    Solve min-cost flow and detect violations.
    """

    try:
        trajectories, cost = mcc4mot(node["detection"], node["transition"])
    except Exception:
        return float("inf"), None, float("inf"), []

    if trajectories is None:
        return float("inf"), None, float("inf"), []

    violations = find_violations(trajectories, matches_hard, node_in_id, node_out_id)
    
    return trajectories, np.sum(cost), violations

def initialize_correction_data(transition_arcs, matches_hard, node_in, node_out, threshold):
    """
    Initialize data structure for lower bound correction.
    
    For each match, compute the minimum additional cost required to switch
    from having both nodes selected to having only one selected.
    """
    min_switch_cost = {}
    cost_diff = {}
    u_switched_cost = {}
    v_switched_cost = {}

    for match_id, (u, v, *_) in enumerate(matches_hard):
        # Find all incoming/outgoing edges for u and v
        u_incoming_cost = []
        u_outgoing_cost = []
        v_incoming_cost = []
        v_outgoing_cost = []
        
        # Collect transition costs
        for s, t, cost in transition_arcs:
            if t == node_in.get(u, -1):
                u_incoming_cost.append(cost)
            if s == node_out.get(u, -1):
                u_outgoing_cost.append(cost)
            if t == node_in.get(v, -1):
                v_incoming_cost.append(cost)
            if s == node_out.get(v, -1):
                v_outgoing_cost.append(cost)
        
        # Compute minimum costs
        u_in_min = min(u_incoming_cost) if u_incoming_cost else threshold
        u_out_min = min(u_outgoing_cost) if u_outgoing_cost else threshold
        v_in_min = min(v_incoming_cost) if v_incoming_cost else threshold
        v_out_min = min(v_outgoing_cost) if v_outgoing_cost else threshold
        
        min_violation_cost = max(u_in_min + v_out_min, v_in_min + u_out_min)
        u_min_cost = u_in_min + u_out_min
        v_min_cost = v_in_min + v_out_min
        min_feasible_cost = min(u_min_cost, v_min_cost)
        # Minimum cost to switch from selecting both to selecting only one
        # This is problem-specific heuristic
        min_switch_cost[match_id] = min_feasible_cost - min_violation_cost
        cost_diff[match_id] = u_min_cost - v_min_cost
        u_switched_cost[match_id] = u_min_cost - min_violation_cost
        v_switched_cost[match_id] = v_min_cost - min_violation_cost
        
    min_switch_cost_modified = {key: max(value, 0) for key, value in min_switch_cost.items()}
    u_switched_cost = {key: max(value, 0) for key, value in u_switched_cost.items()}
    v_switched_cost = {key: max(value, 0) for key, value in v_switched_cost.items()}
    return min_switch_cost, min_switch_cost_modified, cost_diff, u_switched_cost, v_switched_cost

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

def compute_solution_cost(sol, detection_arcs, transition_arcs, matches_hard, node_in, node_out):
    """
    Compute total cost of a solution.
    """
    total_cost = 0
    
    # Map solution to node selections
    selected_nodes = set()
    for path in sol:
        for node in path:
            selected_nodes.add(node)
    
    # Add detection costs
    for arc in detection_arcs:
        node_id = arc[0]
        # Check if node is selected (assuming detection cost is in arc[3])
        if node_id in selected_nodes:
            total_cost += arc[3]
    
    # Add transition costs
    for path in sol:
        for i in range(len(path)-1):
            from_node = path[i]
            to_node = path[i+1]
            # Find transition cost
            for s, t, cost in transition_arcs:
                if s == from_node and t == to_node:
                    total_cost += cost
                    break
    
    return total_cost

def apply_forbid(detection_arcs, transition_arcs, forbid_id, node_in, node_out): 
    """ Forbid a node by disabling detection and removing all incident transitions. """ 
    BIG = 1e3 
    # Disable detection 
    detection_arcs[np.where(detection_arcs[:, 0] == node_in[forbid_id])[0], 1:3] = BIG 
    detection_arcs[np.where(detection_arcs[:, 0] == node_out[forbid_id])[0], 1:3] = BIG 

    # Remove transition
    keep = [] 
    for s, t, _ in transition_arcs: 
        keep.append((s != node_in[forbid_id]) and (t != node_in[forbid_id]) and (s != node_out[forbid_id]) and (t != node_out[forbid_id])) 
    return detection_arcs, transition_arcs[np.array(keep)]

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
    # The min-cost circulation formulation of MAP solver for multi-object
    # tracking.
    # INPUT: 
    # Assuming we have n detections in the video, then
    # detection_arcs: a n x 4 matrix, each row corresponds to a detection in the
    # form of [detection_id, C_i, C_i^en, C_i^ex];
    # transition_arcs: a m x 3 matrix, each row corresponds to a transition arc
    # in the form of [detection_id_i, detection_id_j, C_i,j]
    # NOTE that the id should be unique and in the range of 1 to n. Detailed 
    # defintion can be found in section 3 of the reference:

    # OUTPUT:
    # traj: cells containing the linking results; each cell contains a
    # set of ordered detection ids, which indicate a trajectory
    # cost: costs of these trajectories
    detection_arcs_tem = detection_arcs.copy()
    transition_arcs_tem = transition_arcs.copy()
    detection_arcs_tem[:,0] = detection_arcs_tem[:,0] + 1
    transition_arcs_tem[:,:2] = transition_arcs_tem[:,:2] + 1
    if sys.platform == 'win32':
        _cinda = ctypes.CDLL(os.path.join('C_package', 'lib_cinda_funcs.dll'))
    else:
        _cinda = ctypes.CDLL(os.path.join('C_package', 'lib_cinda_funcs.so'))
    _cinda.pyCS2.argtypes = (ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
    _cinda.pyCS2.restype = ctypes.POINTER(ctypes.c_longlong)

    mtail, mhead, mlow, macap, mcost, msz = cinda_data_process(detection_arcs_tem, transition_arcs_tem)
    it_flag = False
    if isinstance(mcost[0], float):
        mcost = [int(n * 10**7) for n in mcost]
        it_flag = True
    
    inf_type = ctypes.c_long * msz[0]
    a_type = ctypes.c_double * msz[2]

    track_vec = _cinda.pyCS2(inf_type(*msz), a_type(*mtail), a_type(*mhead), a_type(*mlow), a_type(*macap), a_type(*mcost))

    cost = []
    traj = []
    sub_traj = []
    #print trac_vec[0]
    # print(track_vec[10])
    for i in range(1, track_vec[0]+1):
        # print(i, track_vec[i])
        if  track_vec[i] > 0:
            sub_traj.append(track_vec[i])
        else:
            cost.append(track_vec[i])
            # print(cost)
            new = [int(x/2) for x in sub_traj[::2]]
            traj.append(np.array(new)-1)
            sub_traj = []
    
    if it_flag:
        cost = [(float(n) / 10**7) for n in cost]
    return traj, cost

def cinda_data_process(detection_arcs, transition_arcs):
    mtail = []
    mhead = []
    mlow = []
    macap = []
    mcost = []

    n_detection = len(detection_arcs)
    n_transition = len(transition_arcs)
    n_traj = n_detection * 3 + n_transition

    mlow = [0] * n_traj
    macap = [1] * n_traj

    # construct entry arc in detection_arcs
    mtail.extend([1] * n_detection)
    mhead.extend(detection_arcs[:, 0] * 2)
    mcost.extend(detection_arcs[:, 1])

    # construct existing arc in detection_arcs
    mtail.extend(detection_arcs[:, 0] * 2 + 1)
    mhead.extend([1] * n_detection)
    mcost.extend(detection_arcs[:, 2])

    # construct detection arc in detection_arcs
    mtail.extend(detection_arcs[:, 0] * 2)
    mhead.extend(detection_arcs[:, 0] * 2 + 1)
    mcost.extend(detection_arcs[:, 3])

    # construct transition arc
    mtail.extend(transition_arcs[:, 0] * 2 + 1)
    mhead.extend(transition_arcs[:, 1] * 2)
    mcost.extend(transition_arcs[:, 2])
    
    msz = [12, 2 * n_detection + 1, len(mtail)]
    return mtail, mhead, mlow, macap, mcost, msz

def ovDistanceRegion(curRegVox: np.ndarray, nextRegVox: np.ndarray, 
                    frame_shift: Optional[np.ndarray] = None, 
                    ovFlag: bool = False) -> Tuple[float, float, np.ndarray, float]:
    """
    Calculate the overlapping distance between two regions
    
    Parameters:
    -----------
    curRegVox : np.ndarray
        coordinates of the current region z,y,x
    nextRegVox : np.ndarray
        coordinates of the neighboring region with order the same as curRegVox
    frame_shift : np.ndarray, optional
        shift vector (same order as curRegVox)
    ovFlag : bool, optional
        whether using overlapping ratio as distance
        
    Returns:
    --------
    maxDistance : float
        maximum distance
    minDistance : float
        minimum distance
    distances : np.ndarray
        overlapping distances between two regions
    re_ratio : float
        redundancy ratio
    """
    
    re_ratio = 0.0
    if frame_shift is None:
        if curRegVox.shape[1] == 3:
            frame_shift = np.array([0, 0, 0])
        elif curRegVox.shape[1] == 2:
            frame_shift = np.array([0, 0])
        else:
            frame_shift = np.array([])

    if not ovFlag:
        # After downsampling, we can use this method to calculate distance
        if nextRegVox.shape[0] < 2 or curRegVox.shape[0] < 2:
            # Less than 2 pixels
            distances = np.array([100.0, 100.0])
        else:
            # Calculate bounding boxes
            st_pt1 = np.min(curRegVox, axis=0)
            end_pt1 = np.max(curRegVox, axis=0)
            bw_sz1 = np.ceil(end_pt1 - st_pt1 + 1).astype(int)
            
            # Create binary mask for first region
            mask1 = np.zeros(bw_sz1, dtype=bool)
            cell1_sub = (curRegVox - st_pt1).astype(int)
            mask1[tuple(cell1_sub.T)] = True
            cell1_idx = np.ravel_multi_index(
                tuple(cell1_sub.T),dims=bw_sz1)

            st_pt2 = np.min(nextRegVox, axis=0)
            end_pt2 = np.max(nextRegVox, axis=0)
            bw_sz2 = np.ceil(end_pt2 - st_pt2 + 1).astype(int)
            
            # Create binary mask for second region
            mask2 = np.zeros(bw_sz2, dtype=bool)
            cell2_sub = (nextRegVox - st_pt2).astype(int)
            cell2_idx = np.ravel_multi_index(tuple(cell2_sub.T), bw_sz2)
            mask2[tuple(cell2_sub.T)] = True
            
            # Calculate distances using Euclidean distance transform
            
            # Calculate shift
            mov_shift = (st_pt2 - st_pt1 - frame_shift).astype(float)

            dist2cell1 = edt_3d(mask1, mask2, mov_shift)
            dist2cell2 = edt_3d(mask2, mask1, -mov_shift)
                
            distances_n2c_way5 = dist2cell1[tuple(cell2_sub.T)]
            distances_c2n_way5 = dist2cell2[tuple(cell1_sub.T)]
            
            distances_n2c_way5 = np.sqrt(np.array(distances_n2c_way5))
            distances_c2n_way5 = np.sqrt(np.array(distances_c2n_way5))
            
            distances = np.array([
                np.mean(distances_c2n_way5),
                np.mean(distances_n2c_way5)
            ])  # i2j and j2i
            
            # Calculate redundancy ratio
            boxSize = np.prod(bw_sz1) + np.prod(bw_sz2)
            redundant_sz = boxSize - len(cell2_sub)
            re_ratio = redundant_sz / boxSize
    else:
        # Purely based on overlapping ratio
        # Find intersection of points
        cur_set = set(tuple(pt) for pt in curRegVox)
        next_set = set(tuple(pt) for pt in nextRegVox)
        intersection = cur_set.intersection(next_set)
        
        overlap_ratio = len(intersection) / (len(cur_set) + len(next_set) - len(intersection))
        distances = -np.log(overlap_ratio) if overlap_ratio > 0 else np.inf
        distances = [distances,distances]
        re_ratio = overlap_ratio
    maxDistance = np.max(distances)
    minDistance = np.min(distances)
    
    return maxDistance, minDistance, distances, re_ratio

def edt_3d(ref_cell, mov_cell, shift):



    """
    Compute 3D Euclidean Distance Transform
    
    Parameters:
    -----------
    ref_cell : numpy.ndarray (bool, 3D)
        Reference cell mask
    mov_cell : numpy.ndarray (bool, 3D)
        Moving cell mask
    shift : list or numpy.ndarray (float, 3)
        [y, x, z] shifts
        
    Returns:
    --------
    output : numpy.ndarray (float, 3D)
        Distance transform result
    """
    ref_cell = np.asarray(ref_cell, dtype=np.bool_).astype(np.uint8)
    mov_cell = np.asarray(mov_cell, dtype=np.bool_).astype(np.uint8)
    shift = np.asarray(shift, dtype=np.float32)
    
    ref_dims = np.array(ref_cell.shape, dtype=np.int32)
    mov_dims = np.array(mov_cell.shape, dtype=np.int32)
    
    
    if sys.platform == 'win32':
        lib = ctypes.CDLL(os.path.join('C_package', 'edt_3d.dll'))
    else:
        lib = ctypes.CDLL(os.path.join('C_package', 'libedt3d.so'))

    lib.edt_3d.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),  # ref_cell
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),  # ref_dims
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),  # mov_cell
        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),  # mov_dims
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # shift
        np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')  # output
    ]
    lib.edt_3d.restype = None
    # ref_cell = ref_cell.transpose(1, 2, 0)  # zyx -> yxz
    # mov_cell = mov_cell.transpose(1, 2, 0)  # zyx -> yxz
    # ref_cell = np.ascontiguousarray(ref_cell)
    # mov_cell = np.ascontiguousarray(mov_cell)
    # shift = np.array([shift[1], shift[2], shift[0]], dtype=np.float32)
    output = np.zeros(mov_cell.shape, dtype=np.float32)
    lib.edt_3d(
        ref_cell,
        ref_dims,2,
        mov_cell,
        mov_dims,2,
        shift,
        output
    )
    # output = output.transpose(2, 0, 1)  # yxz -> zyx
    return output

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
    if len(crop_area) > 0:
        if len(crop_area[0]) == 2:
            movieInfo1_partial['xCoord'] = movieInfo1_partial['xCoord'] + crop_area[1][0]
            movieInfo1_partial['yCoord'] = movieInfo1_partial['yCoord'] + crop_area[0][0]
            movieInfo2_partial['xCoord'] = movieInfo2_partial['xCoord'] + crop_area[1][0]
            movieInfo2_partial['yCoord'] = movieInfo2_partial['yCoord'] + crop_area[0][0]
            movieInfo1_partial['vox'] = [vox - np.array([crop_area[0][0], crop_area[1][0]]) 
                for vox in movieInfo1_partial['vox']]
            movieInfo2_partial['vox'] = [vox + np.array([crop_area[0][0], crop_area[1][0]]) 
                for vox in movieInfo2_partial['vox']]
        if len(crop_area[0]) == 3:
            movieInfo1_partial['xCoord'] = movieInfo1_partial['xCoord'] + crop_area[2][0]
            movieInfo1_partial['yCoord'] = movieInfo1_partial['yCoord'] + crop_area[1][0]
            movieInfo1_partial['zCoord'] = movieInfo1_partial['zCoord'] + crop_area[0][0]
            movieInfo2_partial['xCoord'] = movieInfo2_partial['xCoord'] + crop_area[2][0]
            movieInfo2_partial['yCoord'] = movieInfo2_partial['yCoord'] + crop_area[1][0]
            movieInfo2_partial['zCoord'] = movieInfo2_partial['zCoord'] + crop_area[0][0]
            movieInfo1_partial['vox'] = [vox - np.array([crop_area[0][0], crop_area[1][0], crop_area[2][0]]) 
                for vox in movieInfo1_partial['vox']]
            movieInfo2_partial['vox'] = [vox + np.array([crop_area[0][0], crop_area[1][0], crop_area[2][0]]) 
                for vox in movieInfo2_partial['vox']]
            
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
                parent_id = movieInfo2['parents'][full_id] + n1_all
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