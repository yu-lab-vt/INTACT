import numpy as np
import pandas as pd
import branchbound as bb
import ilp
import copy
import os
from typing import Dict, List, Tuple, Any, Set, Optional
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'C_package'))
import csv
import tifffile
import time
import warnings
import pickle
import pulp

def tracklets_integration(
    csv_path1: str,
    csv_path2: str,
    seg_indice: str = 'python',
    output_path: str = '',
    resolution: List[float] = [1.0, 1.0, 1.0],
    img_shape: List[int] = None,
    img_shape1: List[int] = None,
    img_shape2: List[int] = None,
    temporal_shift: List[int] = [0, 0],
    spatial_shift: List[int] = [[0,0,0],[0,0,0]],
    mode: str = 'simple',
    solver: str = 'gurobi',
    saveSeg: bool = False
) -> Dict[str, Any]:
    """
    Complete tracklet fusion pipeline.
    
    Parameters:
    -----------
    csv_path1, csv_path2 : str
        Paths to input CSV files
    output_path : str
        Path for output CSV file
    resolution : List[float]
        Resolution scaling factors [z, y, x]

    Returns:
    --------
    result_info : Dict[str, Any]
        Dictionary containing fusion results and statistics
    """
    print("=" * 60)
    print("Start Tracklet Integration!")
    print("=" * 60)

    # Step 1: Read tracking data
    print("\nStep 1: Reading tracking data...W")
    start_time = time.time()
    if mode == 'temporal':
        movieInfo1 = csv2movieInfo(csv_path1,img_shape,seg_indice)
        print(f"  Tracklet 1: {len(movieInfo1['xCoord'])} detections")

        movieInfo2 = csv2movieInfo(csv_path2,img_shape,seg_indice)
        print(f"  Tracklet 2: {len(movieInfo2['xCoord'])} detections")

    elif mode =='spatial':
        movieInfo1 = csv2movieInfo(csv_path1,img_shape1,seg_indice)
        print(f"  Tracklet 1: {len(movieInfo1['xCoord'])} detections")

        movieInfo2 = csv2movieInfo(csv_path2,img_shape2,seg_indice)
        print(f"  Tracklet 2: {len(movieInfo2['xCoord'])} detections")

    end_time = time.time()
    print(f"  Reading time: {end_time - start_time:.2f} seconds")

    # Step2: Select common cells based on temporal or spatial overlap
    print("\nStep 2: Selecting common cells...")
    start_time = time.time()
    crop_size = img_shape

    if mode == 'temporal':
        movieInfo1_partial, movieInfo2_partial, idmap1, idmap2, keep_parent_idx, crop_area, tm_shift, movieInfo1, movieInfo2 = select_common(movieInfo1, movieInfo2, mode, temporal_shift, img_shape)
    elif mode == 'spatial':
        movieInfo1_partial, movieInfo2_partial, idmap1, idmap2, keep_parent_idx, crop_area, tm_shift, movieInfo1, movieInfo2 = select_common(movieInfo1, movieInfo2, mode, spatial_shift, img_shape, img_shape1, img_shape2)
        if len(crop_area) == 2:
            crop_size = [crop_area[0][1] - crop_area[0][0]+1, crop_area[1][1]-crop_area[1][0]+1]
        elif len(crop_area) == 3:
            crop_size = [crop_area[0][1] - crop_area[0][0]+1, crop_area[1][1]-crop_area[1][0]+1, crop_area[2][1]-crop_area[2][0]+1]
    else:
        movieInfo1_partial = movieInfo1
        movieInfo2_partial = movieInfo2

    end_time = time.time()
    print(f"  Selection time: {end_time - start_time:.2f} seconds")

    # Step 3: Match nodes
    print("\nStep 3: Find overlapping nodes...")
    start_time = time.time()
    refine_res1, movieInfo1_partial = movieInfo2refine_res(movieInfo1_partial,crop_size)
    refine_res2, movieInfo2_partial = movieInfo2refine_res(movieInfo2_partial,crop_size)

    matches = match_nodes(movieInfo1_partial, movieInfo2_partial, refine_res1, refine_res2)
    end_time = time.time()

    print(f"  Found {len(matches)} matching pairs")
    print(f"  Matching time: {end_time - start_time:.2f} seconds")
    
    if solver == 'gurobi' or solver == 'pulp' or solver == 'cplex':
        # Step 4: Build candidate fusion graph
        print("\nStep 4: Building candidate fusion graph...")
        start_time = time.time()
        graph = ilp.CandidateFusionGraph(
            movieInfo1_partial, movieInfo2_partial, matches,
            resolution=resolution)
        print(f"  Graph has {graph.n_nodes} nodes and {len(graph.edges)} edges")
        # print(f"  Edge types: {np.unique(graph.edge_types, return_counts=True)}")
        end_time = time.time()
        print(f"  Building time: {end_time - start_time:.2f} seconds")

        # Step 5: Solve ILP
        print("\nStep 5: Solving Integer Linear Program...")
        start_time = time.time()
        if solver == 'gurobi':
            solution, solver_info, selected_nodes, selected_edges = ilp.solve_ilp_fusion(graph)
        elif solver == 'pulp':
            solution, solver_info, selected_nodes, selected_edges = ilp.solve_ilp_fusion_pulp(graph)
        elif solver == 'cplex':
            solution, solver_info, selected_nodes, selected_edges = ilp.solve_ilp_fusion_ortools(graph)

        print(f"  Solver status: {solver_info['status_str']}")
        print(f"  Objective value: {solution['objective_value']:.2f}")
        print(f"  Selected nodes: {np.sum(solution['node_selection'])}")
        print(f"  Selected edges: {np.sum(solution['edge_selection'])}")
        end_time = time.time()
        print(f"  Solving time: {end_time - start_time:.2f} seconds")

        # Step 6: Create fused movieInfo
        print("\nStep 6: Creating fused tracklet...")
        start_time = time.time()

        if mode == 'temporal' or mode == 'spatial':
            movieInfo_new = ilp.create_fused_movieInfo(selected_nodes, selected_edges, movieInfo1, movieInfo2, movieInfo1_partial, movieInfo2_partial, idmap1, idmap2, keep_parent_idx, crop_area, tm_shift, img_shape, seg_indice = seg_indice)
        else:
            movieInfo_new = ilp.create_fused_movieInfo(selected_nodes, selected_edges, movieInfo1, movieInfo2, movieInfo1_partial, movieInfo2_partial, seg_indice = seg_indice)
        print(f"  Fused tracklet has {len(movieInfo_new['xCoord'])} detections")

        end_time = time.time()
        print(f"  Fusion time: {end_time - start_time:.2f} seconds")
    elif solver == 'bb':
        # Step 4: Solve fusion
        print("\nStep 4 & 5: Solving fusion...")
        start_time = time.time()
        tracks = bb.solve_fusion(movieInfo1_partial, movieInfo2_partial, matches)
        end_time = time.time()
        print(f"  Fusion time: {end_time - start_time:.2f} seconds")

        # Step 6: Create fused movieInfo
        print("\nStep 6: Creating fused tracklet...")
        start_time = time.time()
        if mode == 'temporal' or mode == 'spatial':
            movieInfo_new = bb.create_fused_movieInfo(tracks, movieInfo1, movieInfo2, movieInfo1_partial, movieInfo2_partial, idmap1, idmap2, keep_parent_idx, crop_area, tm_shift, img_shape, seg_indice = seg_indice)
        else:
            movieInfo_new = bb.create_fused_movieInfo(tracks, movieInfo1, movieInfo2, movieInfo1_partial, movieInfo2_partial, seg_indice = seg_indice)
        print(f"  Fused tracklet has {len(movieInfo_new['xCoord'])} detections")
        end_time = time.time()
        print(f" Fusion time: {end_time - start_time:.2f} seconds")

    # Step 7: Save results to csv
    print("\nStep 7: Start saving...")
    start_time = time.time()
    os.makedirs(output_path, exist_ok=True)
    
    movieInfo2csv(movieInfo_new, os.path.join(output_path,'tracking_result_merged.csv'), saveSeg = saveSeg)
    save_file_path = os.path.join(output_path, 'movieInfo.pkl')
    with open(save_file_path, 'wb') as f:
        pickle.dump(movieInfo_new, f)
    end_time = time.time()
    print(f"  Saving time: {end_time - start_time:.2f} seconds")
    
    print("\n" + "=" * 60)
    print("Fusion Complete!")
    print("=" * 60)
    
    return movieInfo_new

def select_common(
    movieInfo1: Dict, 
    movieInfo2: Dict, 
    mode: str, 
    shift: Optional[int] = None,
    img_shape: List[int] = None,
    img_shape1: Optional[List[int]] = None,
    img_shape2: Optional[List[int]] = None,
) -> Tuple[Dict, Dict]:
    """
    Select common cells from two movieInfo dictionaries based on temporal or spatial overlap.
    
    Args:
        movieInfo1: First movieInfo dictionary with keys ['xCoord','yCoord','zCoord','frames','parents','vox']
        movieInfo2: Second movieInfo dictionary with same structure
        mode: 'temporal' or 'spatial' selection mode
        shift: Optional parameter for temporal mode (not used in current implementation)
    
    Returns:
        Tuple of (movieInfo1_partial, movieInfo2_partial, id_mapping1, id_mapping2)
        where id_mapping are dictionaries mapping original ID to new ID
    """
    if mode == 'temporal':
        movieInfo1['frames'] = movieInfo1['frames'] + shift[0]
        movieInfo2['frames'] = movieInfo2['frames'] + shift[1]
        min_frame = min(min(movieInfo1['frames']),min(movieInfo2['frames']))
        movieInfo1['frames'] = movieInfo1['frames'] - min_frame
        movieInfo2['frames'] = movieInfo2['frames'] - min_frame
        movieInfo1_partial, movieInfo2_partial, id_mapping1, id_mapping2, keep_parent_idx, crop_area = _temporal_selection(movieInfo1, movieInfo2)

    elif mode == 'spatial':
        if len(img_shape) == 3:
            movieInfo1['xCoord'] = movieInfo1['xCoord'] + shift[0][2]
            movieInfo1['yCoord'] = movieInfo1['yCoord'] + shift[0][1]
            movieInfo1['zCoord'] = movieInfo1['zCoord'] + shift[0][0]
        else:
            movieInfo1['xCoord'] = movieInfo1['xCoord'] + shift[0][1]
            movieInfo1['yCoord'] = movieInfo1['yCoord'] + shift[0][0]
        if len(img_shape) == 3:
            movieInfo2['xCoord'] = movieInfo2['xCoord'] + shift[1][2]
            movieInfo2['yCoord'] = movieInfo2['yCoord'] + shift[1][1]
            movieInfo2['zCoord'] = movieInfo2['zCoord'] + shift[1][0]
        else:
            movieInfo2['xCoord'] = movieInfo2['xCoord'] + shift[1][1]
            movieInfo2['yCoord'] = movieInfo2['yCoord'] + shift[1][0]

        min_frame = min(min(movieInfo1['frames']),min(movieInfo2['frames']))
        movieInfo1['frames'] = movieInfo1['frames'] - min_frame
        movieInfo2['frames'] = movieInfo2['frames'] - min_frame
        movieInfo1['vox'] = [vox + shift[0] for vox in movieInfo1['vox']]
        movieInfo2['vox'] = [vox + shift[1] for vox in movieInfo2['vox']]
        movieInfo1, movieInfo2, movieInfo1_partial, movieInfo2_partial, id_mapping1, id_mapping2, keep_parent_idx, crop_area = _spatial_selection(movieInfo1, movieInfo2, img_shape, img_shape1, img_shape2, shift)
        
    else:
        raise ValueError(f"Unsupported mode: {mode}. Must be 'temporal' or 'spatial'")
    
    return movieInfo1_partial, movieInfo2_partial, id_mapping1, id_mapping2, keep_parent_idx, crop_area, min_frame, movieInfo1, movieInfo2

def _temporal_selection(
    movieInfo1: Dict, 
    movieInfo2: Dict
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Perform temporal selection based on overlapping frames.
    
    Strategy:
    1. Find overlapping frames between two datasets
    2. Expand range by ±1 frame
    3. Extract cells within expanded frame range
    4. Set parents of cells in the first frame of range to empty
    5. Record ID mapping for potential restoration
    """
    
    # Get frame lists
    frames1 = np.array(movieInfo1['frames'])
    frames2 = np.array(movieInfo2['frames'])
    
    # Find overlapping frames
    common_frames = set(frames1) & set(frames2)
    if not common_frames:
        # No overlap, return empty structures
        empty_info = {key: [] for key in movieInfo1.keys()}
        return empty_info, empty_info, {}, {}
    
    # Find min and max overlapping frames
    min_common_frame = min(common_frames)
    max_common_frame = max(common_frames)
    
    # Expand range by ±1 frame
    start_frame = min_common_frame - 1
    end_frame = max_common_frame + 1
    
    # Extract indices for cells within frame range for both datasets
    idx1 = np.where((frames1 >= start_frame) & (frames1 <= end_frame))[0]
    idx2 = np.where((frames2 >= start_frame) & (frames2 <= end_frame))[0]
    
    # Create new movieInfo structures with extracted cells
    movieInfo1_partial = _extract_cells_by_indices(movieInfo1, idx1)
    movieInfo2_partial = _extract_cells_by_indices(movieInfo2, idx2)
    
    # Set parents of cells in the first frame to empty
    first_frame_mask1 = np.array(movieInfo1_partial['frames']) <= start_frame
    first_frame_mask2 = np.array(movieInfo2_partial['frames']) <= start_frame

    parents_np = np.array(movieInfo1_partial['parents'])
    frames_np = np.array(movieInfo1['frames'])

    mask = (parents_np != -1) & (frames_np[parents_np] <= (start_frame - 1))
    parents_np[mask] = -1

    movieInfo1_partial['parents'] = parents_np.tolist()

    parents_np = np.array(movieInfo2_partial['parents'])
    frames_np = np.array(movieInfo2['frames'])

    mask = (parents_np != -1) & (frames_np[parents_np] <= (start_frame - 1))
    parents_np[mask] = -1

    movieInfo2_partial['parents'] = parents_np.tolist()

    for i in np.where(first_frame_mask1)[0]:
        movieInfo1_partial['parents'][i] = -1
    
    for i in np.where(first_frame_mask2)[0]:
        movieInfo2_partial['parents'][i] = -1
    
    # Create ID mappings
    id_mapping1 = {original_idx: new_idx for new_idx, original_idx in enumerate(idx1)}
    id_mapping2 = {original_idx: new_idx for new_idx, original_idx in enumerate(idx2)}
    
    parents1 = movieInfo1_partial['parents']
    for i in range(len(parents1)):
        if not first_frame_mask1[i] and parents1[i]:
            # Update single parent ID using mapping
            old_parent = parents1[i]
            if old_parent in id_mapping1:
                parents1[i] = id_mapping1[old_parent]
    
    parents2 = movieInfo2_partial['parents']
    for i in range(len(parents2)):
        if not first_frame_mask2[i] and parents2[i]:
            # Update single parent ID using mapping
            old_parent = parents2[i]
            if old_parent in id_mapping2:
                parents2[i] = id_mapping2[old_parent]
    
    keep_parent_idx1 = np.where(frames1 == start_frame)[0]
    keep_parent_idx2 = np.where(frames2 == start_frame)[0] + len(frames1)
    keep_parent_idx = np.concatenate((keep_parent_idx1, keep_parent_idx2))
    crop_area = []
    return movieInfo1_partial, movieInfo2_partial, id_mapping1, id_mapping2, keep_parent_idx, crop_area

def _spatial_selection(
    movieInfo1: Dict, 
    movieInfo2: Dict,
    img_shape: List[int] = None,
    img_shape1: Optional[List[int]] = None,
    img_shape2: Optional[List[int]] = None,
    shift: Optional[List[int]] = None
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Perform spatial selection based on overlapping regions.
    
    Strategy:
    1. Compute bounding boxes from center coordinates
    2. Find overlapping region with margin
    3. Use bounding box pre-filtering for efficiency
    4. Check voxel-level overlap for candidate cells
    5. Extract overlapping cells and their parents/children
    6. Set parents of parent cells to empty
    
    Changes:
    1. Support 2D data (when 'zCoord' is missing)
    2. Optimize voxel set construction for large datasets
    3. Fix type error in _get_extended_cell_set
    """
    
    # Check if data is 2D or 3D
    is_3d = len(img_shape) == 3
    # Convert to numpy arrays for efficient computation
    x1, y1 = (np.array(movieInfo1[key]) for key in ['xCoord', 'yCoord'])
    x2, y2 = (np.array(movieInfo2[key]) for key in ['xCoord', 'yCoord'])
    
    if is_3d:
        z1 = np.array(movieInfo1['zCoord'])
        z2 = np.array(movieInfo2['zCoord'])
    
    # Compute bounding boxes with margins (10% margin for efficiency)

    if is_3d:
        x1_max = int(x1.max())
        x2_max = int(x2.max())
        y1_max = int(y1.max())
        y2_max = int(y2.max())
        z1_max = int(z1.max())
        z2_max = int(z2.max())
        x1_min = int(x1.min())
        x2_min = int(x2.min())
        y1_min = int(y1.min())
        y2_min = int(y2.min())
        z1_min = int(z1.min())
        z2_min = int(z2.min())

        min_diffs = [
            abs(x1.min() - x2.min()),
            abs(y1.min() - y2.min()),
            abs(z1.min() - z2.min())
        ]

        most_diff_dim = ['X', 'Y', 'Z'][:2+is_3d][np.argmax(min_diffs)]

        margin_x = (np.max([x1_max, x2_max]) - np.min([x1_min, x2_min])) * 0.15
        margin_y = (np.max([y1_max, y2_max]) - np.min([y1_min, y2_min])) * 0.15
        margin_z = (np.max([z1_max, z2_max]) - np.min([z1_min, z2_min])) * 0.15

        # Find overlapping region
        overlap_x_min = max(x1_min, x2_min)
        overlap_x_max = min(x1_max, x2_max)
        overlap_y_min = max(y1_min, y2_min)
        overlap_y_max = min(y1_max, y2_max)
        overlap_z_min = max(z1_min, z2_min)
        overlap_z_max = min(z1_max, z2_max)

        # Quick check if there's any potential overlap
        if (overlap_x_min > overlap_x_max or 
            overlap_y_min > overlap_y_max or 
            overlap_z_min > overlap_z_max):
            empty_info = {key: [] for key in movieInfo1.keys()}
            return empty_info, empty_info, {}, {}, [], []
        
        expand_x_min = max(int(overlap_x_min - margin_x),0)
        expand_x_max = min(int(overlap_x_max + margin_x), img_shape[2])
        expand_y_min = max(int(overlap_y_min - margin_y),0)
        expand_y_max = min(int(overlap_y_max + margin_y), img_shape[1])
        expand_z_min = max(int(overlap_z_min - margin_z),0)
        expand_z_max = min(int(overlap_z_max + margin_z), img_shape[0])

    else:
        x1_max = int(x1.max())
        x2_max = int(x2.max())
        y1_max = int(y1.max())
        y2_max = int(y2.max())
        x1_min = int(x1.min())
        x2_min = int(x2.min())
        y1_min = int(y1.min())
        y2_min = int(y2.min())

        min_diffs = [
            abs(x1.min() - x2.min()),
            abs(y1.min() - y2.min()),
        ]

        most_diff_dim = ['X', 'Y'][:2][np.argmax(min_diffs)]

        # 2D case - no z coordinates
        margin_x = (np.max([x1_max, x2_max]) - np.min([x1_min, x2_min])) * 0.3
        margin_y = (np.max([y1_max, y2_max]) - np.min([y1_min, y2_min])) * 0.3
        
        # Find overlapping region
        overlap_x_min = max(x1_min, x2_min)
        overlap_x_max = min(x1_max, x2_max)
        overlap_y_min = max(y1_min, y2_min)
        overlap_y_max = min(y1_max, y2_max)

        # Quick check if there's any potential overlap
        if (overlap_x_min > overlap_x_max or 
            overlap_y_min > overlap_y_max):
            empty_info = {key: [] for key in movieInfo1.keys()}
            return empty_info, empty_info, {}, {}, [], [],[],[]
        
        expand_x_min = max(int(overlap_x_min - margin_x),0)
        expand_x_max = min(int(overlap_x_max + margin_x), img_shape[1])
        expand_y_min = max(int(overlap_y_min - margin_y),0)
        expand_y_max = min(int(overlap_y_max + margin_y), img_shape[0])

        # Set dummy z values for 2D case
        overlap_z_min = 0
        overlap_z_max = 0
        expand_z_min = 0
        expand_z_max = 0
        margin_z = 0

    # Pre-filter cells using bounding box check (fast)
    if is_3d:
        bbox_filter1 = ((x1 >= overlap_x_min) & (x1 <= overlap_x_max) &
                        (y1 >= overlap_y_min) & (y1 <= overlap_y_max) &
                        (z1 >= overlap_z_min) & (z1 <= overlap_z_max))
        
        bbox_filter2 = ((x2 >= overlap_x_min) & (x2 <= overlap_x_max) &
                        (y2 >= overlap_y_min) & (y2 <= overlap_y_max) &
                        (z2 >= overlap_z_min) & (z2 <= overlap_z_max))

        expand_filter1 = ((x1 >= expand_x_min) & (x1 <= expand_x_max) &
                        (y1 >= expand_y_min) & (y1 <= expand_y_max) &
                        (z1 >= expand_z_min) & (z1 <= expand_z_max))
        
        expand_filter2 = ((x2 >= expand_x_min) & (x2 <= expand_x_max) &
                        (y2 >= expand_y_min) & (y2 <= expand_y_max) &
                        (z2 >= expand_z_min) & (z2 <= expand_z_max))

    else:
        # 2D filtering
        bbox_filter1 = ((x1 >= overlap_x_min) & (x1 <= overlap_x_max) &
                        (y1 >= overlap_y_min) & (y1 <= overlap_y_max))
        
        bbox_filter2 = ((x2 >= overlap_x_min) & (x2 <= overlap_x_max) &
                        (y2 >= overlap_y_min) & (y2 <= overlap_y_max))

        expand_filter1 = ((x1 >= expand_x_min) & (x1 <= expand_x_max) &
                        (y1 >= expand_y_min) & (y1 <= expand_y_max))
        
        expand_filter2 = ((x2 >= expand_x_min) & (x2 <= expand_x_max) &
                        (y2 >= expand_y_min) & (y2 <= expand_y_max))
           
    # Get indices for different regions
    bbox_idx1 = np.where(bbox_filter1)[0]
    bbox_idx2 = np.where(bbox_filter2)[0]

    # Get cells in expanded region but not in bbox region (margin cells)
    expand_idx1 = np.where(expand_filter1 & ~bbox_filter1)[0]
    expand_idx2 = np.where(expand_filter2 & ~bbox_filter2)[0]

    # Initialize candidate sets
    candidate_cells1 = set()
    candidate_cells2 = set()

    for idx in expand_idx1:
        vox = movieInfo1['vox'][idx]
        
        if vox.size == 0:
            continue
            
        if is_3d:     # zyx 
            z, y, x = vox[:, 0], vox[:, 1], vox[:, 2]
            in_z = (z >= overlap_z_min) & (z < overlap_z_max)
            in_rect = in_z & (y >= overlap_y_min) & (y < overlap_y_max) & (x >= overlap_x_min) & (x < overlap_x_max)
        else:     # yx 
            y, x = vox[:, 0], vox[:, 1]
            in_rect = (y >= overlap_y_min) & (y < overlap_y_max) & (x >= overlap_x_min) & (x < overlap_x_max)
        
        if np.any(in_rect):
            candidate_cells1.add(idx)

    for idx in expand_idx2:
        vox = movieInfo2['vox'][idx]
        
        if vox.size == 0:
            continue
            
        if is_3d:     # zyx 
            z, y, x = vox[:, 0], vox[:, 1], vox[:, 2]
            in_z = (z >= overlap_z_min) & (z < overlap_z_max)
            in_rect = in_z & (y >= overlap_y_min) & (y < overlap_y_max) & (x >= overlap_x_min) & (x < overlap_x_max)
        else:     # yx 
            y, x = vox[:, 0], vox[:, 1]
            in_rect = (y >= overlap_y_min) & (y < overlap_y_max) & (x >= overlap_x_min) & (x < overlap_x_max)
        
        if np.any(in_rect):
            candidate_cells2.add(idx)

    # Final cell selection: bbox cells + candidate margin cells
    final_idx1 = set(bbox_idx1) | candidate_cells1
    final_idx2 = set(bbox_idx2) | candidate_cells2

    # If no cells selected, return empty structures
    if len(final_idx1) == 0 or len(final_idx2) == 0:
        empty_info = {key: [] for key in movieInfo1.keys()}
        return empty_info, empty_info, {}, {}, [], [],[],[]

    # Collect indices of cells to remove based on boundary touching
    cells_to_remove1 = set()
    cells_to_remove2 = set()

    if most_diff_dim == 'X':
        # For dataset 1, mark cells touching x1_max
        for idx in final_idx1:
            vox = movieInfo1['vox'][idx]
            if vox.size == 0:
                continue
            if is_3d:
                # vox format: (z, y, x)
                if np.any(vox[:, 2] >= x1_max - 1):  # Touching boundary
                    cells_to_remove1.add(idx)
            else:
                # vox format: (y, x)
                if np.any(vox[:, 1] >= x1_max - 1):  # Touching boundary
                    cells_to_remove1.add(idx)
        
        # For dataset 2, mark cells touching x2_min
        for idx in final_idx2:
            vox = movieInfo2['vox'][idx]
            if vox.size == 0:
                continue
            if is_3d:
                if np.any(vox[:, 2] <= x2_min + 1):  # Touching boundary
                    cells_to_remove2.add(idx)
            else:
                if np.any(vox[:, 1] <= x2_min + 1):  # Touching boundary
                    cells_to_remove2.add(idx)

    elif most_diff_dim == 'Y':
                # For dataset 1, mark cells touching x1_max
        for idx in final_idx1:
            vox = movieInfo1['vox'][idx]
            if vox.size == 0:
                continue
            if is_3d:
                # vox format: (z, y, x)
                if np.any(vox[:, 1] >= y1_max - 1):  # Touching boundary
                    cells_to_remove1.add(idx)
            else:
                # vox format: (y, x)
                if np.any(vox[:, 0] >= y1_max - 1):  # Touching boundary
                    cells_to_remove1.add(idx)
        
        # For dataset 2, mark cells touching x2_min
        for idx in final_idx2:
            vox = movieInfo2['vox'][idx]
            if vox.size == 0:
                continue
            if is_3d:
                if np.any(vox[:, 1] <= y2_min + 1):  # Touching boundary
                    cells_to_remove2.add(idx)
            else:
                if np.any(vox[:, 0] <= y2_min + 1):  # Touching boundary
                    cells_to_remove2.add(idx)

    elif is_3d and most_diff_dim == 'Z':
                # For dataset 1, mark cells touching x1_max
        for idx in final_idx1:
            vox = movieInfo1['vox'][idx]
            if vox.size == 0:
                continue
                # vox format: (z, y, x)
            if np.any(vox[:, 0] >= z1_max - 1):  # Touching boundary
                cells_to_remove1.add(idx)
        
        # For dataset 2, mark cells touching x2_min
        for idx in final_idx2:
            vox = movieInfo2['vox'][idx]
            if vox.size == 0:
                continue
            if np.any(vox[:, 0] <= z2_min + 1):  # Touching boundary
                cells_to_remove2.add(idx)

    # Remove the marked cells from movieInfo1 and movieInfo2
    if cells_to_remove1:
        movieInfo1, id_mapping1_update = remove_cells_by_indices(movieInfo1, cells_to_remove1)
        # Update final_idx1 using the new mapping
        final_idx1 = {id_mapping1_update[idx] for idx in final_idx1 if idx not in cells_to_remove1}
    else:
        id_mapping1_update = {}

    if cells_to_remove2:
        movieInfo2, id_mapping2_update = remove_cells_by_indices(movieInfo2, cells_to_remove2)
        # Update final_idx2 using the new mapping
        final_idx2 = {id_mapping2_update[idx] for idx in final_idx2 if idx not in cells_to_remove2}
    else:
        id_mapping2_update = {}

    # FIX: Pass cell indices to _get_extended_cell_set, not voxel sets
    idx_to_extract1, added_parents1 = _get_extended_cell_set(movieInfo1, final_idx1)
    idx_to_extract2, added_parents2 = _get_extended_cell_set(movieInfo2, final_idx2)

    extract_idx1 = sorted(idx_to_extract1)
    extract_idx2 = sorted(idx_to_extract2)
    # Extract cells
    movieInfo1_partial = _extract_cells_by_indices(movieInfo1, extract_idx1, added_parents1)
    movieInfo2_partial = _extract_cells_by_indices(movieInfo2, extract_idx2, added_parents2)
    
    # Adjust coordinates
    if is_3d:
        movieInfo1_partial['xCoord'] = movieInfo1_partial['xCoord'] - expand_x_min
        movieInfo1_partial['yCoord'] = movieInfo1_partial['yCoord'] - expand_y_min
        movieInfo1_partial['zCoord'] = movieInfo1_partial['zCoord'] - expand_z_min
        movieInfo2_partial['xCoord'] = movieInfo2_partial['xCoord'] - expand_x_min
        movieInfo2_partial['yCoord'] = movieInfo2_partial['yCoord'] - expand_y_min
        movieInfo2_partial['zCoord'] = movieInfo2_partial['zCoord'] - expand_z_min
        crop_area = [[expand_z_min, expand_z_max],[expand_y_min, expand_y_max], [expand_x_min, expand_x_max]] 
        # Adjust voxel coordinates
        movieInfo1_partial['vox'] = [vox - np.array([expand_z_min, expand_y_min, expand_x_min]) 
            for vox in movieInfo1_partial['vox']]
        movieInfo2_partial['vox'] = [vox - np.array([expand_z_min, expand_y_min, expand_x_min]) 
            for vox in movieInfo2_partial['vox']]
    else:
        # 2D case: adjust only x and y coordinates
        movieInfo1_partial['xCoord'] = movieInfo1_partial['xCoord'] - expand_x_min
        movieInfo1_partial['yCoord'] = movieInfo1_partial['yCoord'] - expand_y_min
        movieInfo2_partial['xCoord'] = movieInfo2_partial['xCoord'] - expand_x_min
        movieInfo2_partial['yCoord'] = movieInfo2_partial['yCoord'] - expand_y_min
        crop_area = [[expand_y_min, expand_y_max], [expand_x_min, expand_x_max]] 
        # Adjust voxel coordinates (2D)

        movieInfo1_partial['vox'] = [vox - np.array([expand_y_min, expand_x_min]) 
            for vox in movieInfo1_partial['vox']]
        movieInfo2_partial['vox'] = [vox - np.array([expand_y_min, expand_x_min]) 
            for vox in movieInfo2_partial['vox']]

    # Create ID mappings
    id_mapping1 = {old_idx: new_idx for new_idx, old_idx in enumerate(extract_idx1)}
    id_mapping2 = {old_idx: new_idx for new_idx, old_idx in enumerate(extract_idx2)}
    # id_mapping1 = {original_idx: new_idx for new_idx, original_idx in enumerate(list(idx_to_extract1))}
    # id_mapping2 = {original_idx: new_idx for new_idx, original_idx in enumerate(list(idx_to_extract2))}
    
    # Update parent IDs using mappings
    parents1 = movieInfo1_partial['parents']
    for i in range(len(parents1)):
        if parents1[i] is not None:  # Check for None instead of not parents1[i]
            old_parent = parents1[i]
            if old_parent in id_mapping1:
                parents1[i] = id_mapping1[old_parent]
            else:
                # If parent is not in the extracted set, set to None
                parents1[i] = -1
    
    parents2 = movieInfo2_partial['parents']
    for i in range(len(parents2)):
        if parents2[i] is not None:  # Check for None instead of not parents2[i]
            old_parent = parents2[i]
            if old_parent in id_mapping2:
                parents2[i] = id_mapping2[old_parent]
            else:
                # If parent is not in the extracted set, set to None
                parents2[i] = -1

    movieInfo1_partial, old_to_new_id_map1 = sort_movieinfo_by_frames(movieInfo1_partial)
    movieInfo2_partial, old_to_new_id_map2 = sort_movieinfo_by_frames(movieInfo2_partial)

    new_id_mapping1 = {}
    for old_original_id, old_extracted_id in id_mapping1.items():
        new_extracted_id = old_to_new_id_map1[old_extracted_id]
        new_id_mapping1[old_original_id] = new_extracted_id
    id_mapping1 = new_id_mapping1

    new_id_mapping2 = {}
    for old_original_id, old_extracted_id in id_mapping2.items():
        new_extracted_id = old_to_new_id_map2[old_extracted_id]
        new_id_mapping2[old_original_id] = new_extracted_id
    id_mapping2 = new_id_mapping2

    keep_parent_idx = np.concatenate((list(added_parents1), list(np.array(list(added_parents2)) + len(movieInfo1['frames']))))

    return movieInfo1, movieInfo2, movieInfo1_partial, movieInfo2_partial, id_mapping1, id_mapping2, keep_parent_idx, crop_area

def remove_cells_by_indices(movieInfo, indices_to_remove):
    """
    Remove cells with given indices from movieInfo dictionary and update parent IDs.
    
    Parameters:
    -----------
    movieInfo : dict
        Original movieInfo dictionary
    indices_to_remove : set
        Set of indices to remove
    
    Returns:
    --------
    tuple: (updated_movieInfo, id_mapping)
        updated_movieInfo: Updated movieInfo with cells removed
        id_mapping: Dictionary mapping old IDs to new IDs
    """
    if not indices_to_remove:
        return movieInfo, {}
    
    # Create list of indices to keep
    n_cells = len(movieInfo['xCoord'])
    all_indices = set(range(n_cells))
    keep_indices = sorted(all_indices - indices_to_remove)
    
    # Create mapping from old IDs to new IDs
    id_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
    
    # Update arrays
    for key in ['xCoord', 'yCoord', 'zCoord', 'frames', 'parents']:
        if key in movieInfo:
            arr = movieInfo[key]
            if isinstance(arr, np.ndarray):
                movieInfo[key] = arr[keep_indices]
            else:
                movieInfo[key] = np.array([arr[i] for i in keep_indices])
    
    # Update lists
    for key in ['vox', 'voxIdx']:
        if key in movieInfo:
            arr = movieInfo[key]
            movieInfo[key] = [arr[i] for i in keep_indices]
    
    # Update parent IDs using the new mapping
    parents = movieInfo['parents']
    for i in range(len(parents)):
        if parents[i] is not None and parents[i] >= 0:  # Check for valid parent ID
            old_parent = parents[i]
            if old_parent in id_mapping:
                parents[i] = id_mapping[old_parent]
            else:
                # If parent was removed, set to -1 (no parent)
                parents[i] = -1
    
    return movieInfo, id_mapping

def sort_movieinfo_by_frames(movieInfo):
    """
    Sort movieInfo by frames field and update parent references
    
    Args:
        movieInfo: Dictionary containing cell tracking information
        
    Returns:
        sorted_movieInfo: Sorted movieInfo dictionary
        old_to_new_id_map: Array mapping old IDs to new IDs
    """
    # Sort by frames
    sorted_indices = np.argsort(movieInfo['frames'])
    old_to_new_id_map = np.empty_like(sorted_indices)
    old_to_new_id_map[sorted_indices] = np.arange(len(sorted_indices))
    
    # Create a copy of movieInfo to avoid modifying original data
    sorted_movieInfo = {}
    
    # Sort numeric fields
    for key in ['xCoord', 'yCoord', 'zCoord', 'frames', 'parents']:
        if key in movieInfo:
            arr = movieInfo[key]
            sorted_movieInfo[key] = np.array([arr[i] for i in sorted_indices])
    
    # Sort list fields
    for key in ['vox', 'voxIdx']:
        if key in movieInfo:
            arr = movieInfo[key]
            sorted_movieInfo[key] = [arr[i] for i in sorted_indices]
    
    # Update parent references to new IDs
    new_parents = []
    for p in sorted_movieInfo['parents']:
        if p >= 0:
            new_parents.append(old_to_new_id_map[p])
        else:
            new_parents.append(p)
    sorted_movieInfo['parents'] = new_parents
    
    return sorted_movieInfo, old_to_new_id_map

def movieInfo2refine_res(movieInfo, crop_size):
    """
    Convert movieInfo to refine_res format.
    
    Parameters:
    -----------
    movieInfo : dict
        Dictionary containing:
        - 'frames': list of frame indices for each cell
        - 'vox': list of n×3 arrays containing zyx coordinates for each cell
    crop_size : tuple
        Size of the crop array (z, y, x)
    
    Returns:
    --------
    refine_res : list
        List of 3D arrays with cell labels
    movieInfo : dict
        Updated dictionary with 'perframe' key containing cell counts per frame
    """
    
    # Calculate frame range
    min_frame = min(movieInfo['frames'])
    max_frame = max(movieInfo['frames'])
    num_frames = int(max_frame - min_frame + 1)
    
    # Initialize refine_res as a list of zero arrays
    refine_res = [np.zeros(tuple(crop_size), dtype=np.int16) for _ in range(num_frames)]
    
    # start_time = time.time()
    current_frame_label = 1
    prev_frame = None

    for i, frame in enumerate(movieInfo['frames']):
        frame_idx = int(frame - min_frame)
        vox = movieInfo['vox'][i]
        
        # Check if frame has changed, if so, reset the label
        if frame != prev_frame:
            current_frame_label = 1
            prev_frame = frame

        # Get the voxel coordinates (z, y, x)
        vox_arr = np.array(vox)
        valid_mask = np.all((vox_arr >= 0) & (vox_arr < np.array(crop_size)), axis=1)
        if sum(valid_mask) != len(vox):
            print(f"Warning: {sum(valid_mask)} valid voxels for cell {i}, expected {len(vox)}")
        # Convert (z, y, x) to linear index for batch assignment
        indices = np.ravel_multi_index(vox_arr.T, crop_size)  # T for transpose
        
        # Assign the labels in bulk (without looping over voxels individually)
        refine_res[frame_idx].flat[indices] = current_frame_label

        # Increment label for the next voxel
        current_frame_label += 1
    # end_time = time.time()
    # print(f"Time taken for optimized label assignment: {end_time - start_time:.2f} seconds")

    movieInfo['perframe'] = [np.max(cell_id) for cell_id in refine_res]
    
    return refine_res, movieInfo

def _get_extended_cell_set(movieInfo: Dict, base_cells: Set[int]) -> Set[int]:
    """
    Get set of cells to extract including:
    1. Base cells (overlapping cells)
    2. Their parents
    3. Their children
    
    Args:
        movieInfo: Dictionary containing cell information
        base_cells: Set of cell indices to start with
    
    Returns:
        extended_set: Set of all cell indices to extract
        added_parents: Set of parent cell indices that were added
    """
    extended_set = set(base_cells)  # FIX: base_cells should be a set of integers, not voxel sets
    added_parents = set()
    parents_list = movieInfo['parents']
    
    # Add parents of base cells
    for cell_idx in base_cells:
        parents = parents_list[cell_idx]
        if parents is not None and parents not in extended_set:  # If has parents
            extended_set.add(parents)
            added_parents.add(parents)

    # Add children of base cells
    for cell_idx, parents in enumerate(parents_list):
        if parents is not None and parents in base_cells:
            if cell_idx not in extended_set:  # If not already added
                extended_set.add(cell_idx)
    extended_set = {x for x in extended_set if x >= 0}
    added_parents = {x for x in added_parents if x >= 0}

    return extended_set, added_parents

def _extract_cells_by_indices(movieInfo: Dict, indices: List[int], parents_to_remove: Set[int] = None) -> Dict:
    """Extract subset of cells by indices and return new movieInfo dictionary."""
    if parents_to_remove is None:
        parents_to_remove = set()
    
    result = {}
    
    for key in movieInfo.keys():
        val = movieInfo[key]
        if key == 'parents':
            new_parents = []
            for i, orig_idx in enumerate(indices):
                parent_val = val[orig_idx]
                if orig_idx in parents_to_remove:
                    new_parents.append(-1)
                else:
                    if parent_val is not None and parent_val >= 0:
                        new_parents.append(parent_val)
                    else:
                        new_parents.append(-1)
            result[key] = new_parents
        elif isinstance(val, np.ndarray):
            indices_array = np.array(indices).astype(int)
            result[key] = val[indices_array]
        else:
            result[key] = [val[i] for i in indices]
    return result

def csv2movieInfo(csv_path: str, img_shape:tuple, seg_indice:str='python') -> Dict[str, Any]:
    """
    Reads a CSV file containing cell information and organizes it into a movieInfo dictionary.
    
    The CSV file is expected to have the following columns (among others):
    ID, incoming links, outgoing links, frames, x, y, z, parents
    Additionally, all columns from 'voxIdx' onward contain variable-length voxel index data.
    
    Args:
        csv_path (str): Path to the CSV file to be read.
    
    Returns:
        Dict[str, Any]: A dictionary containing cell information with keys:
            - 'xCoord': List of x coordinates
            - 'yCoord': List of y coordinates  
            - 'zCoord': List of z coordinates
            - 'frames': List of frame numbers
            - 'parents': List of parent cell IDs
            - 'voxIdx': List of lists containing voxel indices for each cell
    """
    
    # Initialize the movieInfo dictionary
    movieInfo = {
        'xCoord': [], 'yCoord': [], 'zCoord': [],
        'frames': [], 'parents': [], 'voxIdx': [], 'vox':[]
    }
    
    with open(csv_path, 'r') as csvfile:
        # First, read the header row to find column indices
        reader = csv.reader(csvfile)
        headers = next(reader)  # Get the header row
        
        # Find the index of each required column (case-insensitive)
        header_lower = [h.lower() for h in headers]
        
        vox_idx_index = header_lower.index('voxidx')
        x_index = header_lower.index('x')
        y_index = header_lower.index('y')
        z_index = header_lower.index('z') if 'z' in header_lower else None
        frames_index = header_lower.index('frames')
        parents_index = header_lower.index('parents')
        id_index = header_lower.index('id')
        
        # Read all rows into memory to allow lookup and double-pass processing
        all_rows = list(reader)
        
        # Pass 1: Build ID to row_index mapping
        id_to_row = {}
        for row_idx, row in enumerate(all_rows):
            current_id = row[id_index].strip()
            if current_id:
                try:
                    id_to_row[int(current_id)] = row_idx
                except ValueError:
                    pass
        # Now process each row
        for row in all_rows:
            # Extract fixed columns
            movieInfo['xCoord'].append(float(row[x_index]) if row[x_index] else 0.0)
            movieInfo['yCoord'].append(float(row[y_index]) if row[y_index] else 0.0)
            movieInfo['zCoord'].append(float(row[z_index]) if (z_index is not None and row[z_index]) else 0.0)
            movieInfo['frames'].append(int(float(row[frames_index])) if row[frames_index] else 0)
            
            # Handle parents column (may contain 'nan')
            try:
                parent_val = int(row[parents_index])
            except ValueError:
                parent_val = -1  
            if parent_val >=0:
                movieInfo['parents'].append(id_to_row[parent_val])
            else:
                movieInfo['parents'].append(-1)  # Use -1 for no parent
            
            # Extract all values from voxIdx column to the end of the row
            voxel_values = []
            for i in range(vox_idx_index, len(row)):
                value = row[i].strip()
                if value and value.lower() != 'nan':
                    try:
                        voxel_values.append(int(value))
                    except ValueError:
                        voxel_values.append(value)
            
            movieInfo['voxIdx'].append(voxel_values)
    # Convert lists to numpy arrays for consistency
    # Ensure conversion happens even if some lines were skipped
    for key in ['xCoord', 'yCoord', 'zCoord']:
        if movieInfo[key]:  # Check if list is not empty
            movieInfo[key] = np.array(movieInfo[key], dtype=float)
        else:
            movieInfo[key] = np.array([], dtype=float)
       # Convert flat voxel indices to 3D/2D coordinates
    for key in ['frames']:
        if movieInfo[key]:  # Check if list is not empty
            movieInfo[key] = np.array(movieInfo[key], dtype=int)
            if seg_indice == 'matlab':
                movieInfo[key] = movieInfo[key] - 1
        else:
            movieInfo[key] = np.array([], dtype=int)
    # Convert flat voxel indices to 3D/2D coordinates
    vox_list = []
    if img_shape is not None and len(img_shape) == 3:
        for flat_idx_list in movieInfo['voxIdx']:
            if len(flat_idx_list) > 0:
                idx_arr = np.array(flat_idx_list)
                if seg_indice == 'python':
                    z, y, x = np.unravel_index(idx_arr, img_shape)  # python z,y,x
                elif seg_indice =='matlab':
                    z, x, y = np.unravel_index(idx_arr-1, [img_shape[0], img_shape[2],img_shape[1]])    # matlab y,x,z
                vox_list.append(np.column_stack((z, y, x)))
            else:
                vox_list.append(np.zeros((0, 3)))
    elif img_shape is not None and len(img_shape) == 2:
        for idx, flat_idx_list in enumerate(movieInfo['voxIdx']):
            if len(flat_idx_list) > 0:
                idx_arr = np.array(flat_idx_list)
                if seg_indice == 'python':
                    y, x = np.unravel_index(idx_arr, img_shape)   # python y,x
                elif seg_indice == 'matlab':
                    x, y = np.unravel_index(idx_arr-1, [img_shape[1],img_shape[0]])   # matlab x,y
                vox_list.append(np.column_stack((y, x)))
            else:
                vox_list.append(np.zeros((0, 2)))
    else:
        vox_list = [np.array([]) for _ in movieInfo['voxIdx']]
        
    movieInfo['vox'] = vox_list

    return movieInfo

def match_nodes(movieInfo1_partial, movieInfo2_partial, refine_res1, refine_res2):
    """
    Match cells between two partial movieInfo datasets based on spatial overlap.

    Parameters:
    -----------
    movieInfo1_partial : dict
        Dictionary containing 'vox' and 'perframe' for dataset 1.
    movieInfo2_partial : dict
        Dictionary containing 'vox' and 'perframe' for dataset 2.
    refine_res1 : list
        List of 3D arrays with labeled cells for dataset 1.
    refine_res2 : list
        List of 3D arrays with labeled cells for dataset 2.

    Returns:
    --------
    matches : list
        List of tuples (global_idx_1, global_idx_2).
    """

    num_frames = len(refine_res1)
    
    # Calculate total cells to initialize matrix
    total_cells_1 = np.sum(movieInfo1_partial['perframe'])
    total_cells_2 = np.sum(movieInfo2_partial['perframe'])
    
    # Initialize matches list
    matches = []

    # Precompute global offsets for both datasets
    global_offset1 = np.cumsum(movieInfo1_partial['perframe']) - movieInfo1_partial['perframe']
    global_offset2 = np.cumsum(movieInfo2_partial['perframe']) - movieInfo2_partial['perframe']

    # Calculate temporal offset between datasets
    tm_offset = int(np.min(movieInfo2_partial['frames']) - np.min(movieInfo1_partial['frames']))
    
    # Temporary storage for frame-level matches (frame_idx, local_idx1, local_idx2)
    temp_matches = []
    
    for i in range(tm_offset, len(refine_res1)):
        i2 = i - tm_offset
        map1 = refine_res1[i]
        map2 = refine_res2[i2]
        
        count1 = movieInfo1_partial['perframe'][i]
        count2 = movieInfo2_partial['perframe'][i2]

        if count1 == 0 or count2 == 0:
            continue

        # Find overlapping pixels where both maps have non-zero labels
        overlap_mask = (map1 > 0) & (map2 > 0)
        
        if not np.any(overlap_mask):
            continue
            
        # Get the labels at overlapping positions
        labels1_overlap = map1[overlap_mask]
        labels2_overlap = map2[overlap_mask]
        
        # Combine labels into pairs for unique identification
        # Using a 64-bit integer to combine both labels for hashing
        combined_labels = labels1_overlap.astype(np.int64) * (count2 + 1) + labels2_overlap
        
        # Get unique pairs (avoiding duplicates from multiple overlapping pixels)
        unique_combined = np.unique(combined_labels)
        
        # Decode the combined labels back to individual labels
        for combined in unique_combined:
            label1 = combined // (count2 + 1)
            label2 = combined % (count2 + 1)
            
            # Convert to local indices (labels are 1-based)
            local_idx1 = label1 - 1
            local_idx2 = label2 - 1
            
            # Store frame and local indices for vectorized processing
            temp_matches.append((i, i2, local_idx1, local_idx2))
    
    # Vectorized conversion of local indices to global indices
    if temp_matches:
        # Convert to numpy array for vectorized operations
        temp_matches = np.array(temp_matches)
        frame_indices1 = temp_matches[:, 0].astype(int)
        frame_indices2 = temp_matches[:, 1].astype(int)
        local_indices1 = temp_matches[:, 2].astype(int)
        local_indices2 = temp_matches[:, 3].astype(int)
        
        # Vectorized global index calculation
        global_indices1 = global_offset1[frame_indices1] + local_indices1
        global_indices2 = global_offset2[frame_indices2] + local_indices2
        
        # Create matches list from vectorized results
        matches = list(zip(global_indices1, global_indices2))
    
    return matches

def movieInfo2csv(movieInfo, save_folder: str, saveSeg: bool = False) -> None:
    """
    Convert movieInfo tracking results to CSV format.
    Each voxIdx element is expanded into multiple columns.
    
    Parameters:
    -----------
    movieInfo : dict or object
        Tracking results as either:
        - Dictionary with keys: 'xCoord', 'yCoord', 'zCoord', 'frames', 'parents', 'voxIdx', 'kids'
        - Object with same attributes
    save_folder : str
        Path to save the CSV file
    """
    
    movieInfo_dict = movieInfo
    
    # Extract data with default values
    xCoord = movieInfo_dict.get('xCoord', [])
    yCoord = movieInfo_dict.get('yCoord', [])
    zCoord = movieInfo_dict.get('zCoord', [])
    frames = movieInfo_dict.get('frames', [])
    parents = [p + 1 if p >= 0 else p for p in movieInfo_dict.get('parents', [])]
    voxIdx = movieInfo_dict.get('voxIdx', [])
    kids = movieInfo_dict.get('kids', [])
    
    # Determine the number of cells (detections)
    n_detections = len(xCoord)
    
    # Create ID array
    id_array = list(range(1, n_detections + 1))
    
    # Calculate incoming links (number of parents)
    parentLengths = []
    for i in range(n_detections):
        if i < len(parents):
            if isinstance(parents[i], (list, np.ndarray)):
                parentLengths.append(len(parents[i]))
            elif parents[i] is None or parents[i] == '':
                parentLengths.append(0)
            else:
                parentLengths.append(1)
        else:
            parentLengths.append(0)
    
    # Calculate outgoing links (number of children)
    childLengths = []
    for i in range(n_detections):
        if i < len(kids):
            if isinstance(kids[i], (list, np.ndarray)):
                childLengths.append(len(kids[i]))
            elif kids[i] is None or kids[i] == '':
                childLengths.append(0)
            else:
                childLengths.append(1)
        else:
            childLengths.append(0)
    
    # Convert parents to string representation (single string per cell)
    parents_str = []
    for i in range(n_detections):
        if i < len(parents):
            parent_val = parents[i]
            if isinstance(parent_val, (list, np.ndarray)):
                if len(parent_val) > 0: 
                    parent_val = parent_val[0]+1
                else: 
                    parent_val = np.nan
            if parent_val is None or parent_val == '' or (isinstance(parent_val, (list, np.ndarray)) and len(parent_val) == 0):
                parents_str.append('')
            elif isinstance(parent_val, (list, np.ndarray)):
                # Convert list to string with space separation
                if len(parent_val) == 1:
                    parents_str.append(str(parent_val[0]))
                else:
                    # For multiple parents, join with spaces
                    parents_str.append(' '.join(str(p) for p in parent_val))
            else:
                # Single value
                parents_str.append(str(parent_val))
        else:
            parents_str.append('')
    
    if saveSeg == True:
        # Find maximum number of voxels to determine number of columns needed
        max_voxels = 0
        for i in range(n_detections):
            if i < len(voxIdx):
                vox_val = voxIdx[i]
                if vox_val is not None and vox_val != '':
                    if isinstance(vox_val, (list, np.ndarray)):
                        max_voxels = max(max_voxels, len(vox_val))
                    else:
                        max_voxels = max(max_voxels, 1)
    
    # Create a list to hold all rows (each row is a list of values)
    all_rows = []
    
    # Process each detection
    for i in range(n_detections):
        row = []
        
        # Add fixed columns
        row.append(id_array[i])  # ID
        row.append(parentLengths[i])  # incoming links
        row.append(childLengths[i])  # outgoing links
        row.append(frames[i] if i < len(frames) else 0)  # frames
        row.append(xCoord[i] if i < len(xCoord) else np.nan)  # x
        row.append(yCoord[i] if i < len(yCoord) else np.nan)  # y
        
        # Add z if available
        if i < len(zCoord):
            row.append(zCoord[i])
        
        # Add parents string
        row.append(parents_str[i])
        
        if saveSeg == True:
            # Add voxIdx values as separate columns
            if i < len(voxIdx):
                vox_val = voxIdx[i]
                if vox_val is None or vox_val == '':
                    # Add empty strings for all voxel columns
                    for _ in range(max_voxels):
                        row.append('')
                elif isinstance(vox_val, (list, np.ndarray)):
                    # Add each voxel value as a separate column
                    num_voxels = len(vox_val)
                    for j in range(max_voxels):
                        if j < num_voxels:
                            row.append(vox_val[j])
                        else:
                            row.append('')
                else:
                    # Single value
                    row.append(vox_val)
                    # Add empty strings for remaining columns
                    for _ in range(max_voxels - 1):
                        row.append('')
            else:
                # No voxIdx data for this detection
                for _ in range(max_voxels):
                    row.append('')
            
        all_rows.append(row)
    
    # Create column names
    column_names = ['ID', 'incoming links', 'outgoing links', 'frames', 'x', 'y']
    
    # Add z column if zCoord data exists
    if len(zCoord) > 0:
        column_names.append('z')
    
    column_names.append('parents')
    
    if saveSeg == True:
        # Add voxIdx column names (voxIdx_1, voxIdx_2, ...)
        for i in range(1, max_voxels + 1):
            if i == 1:
                column_names.append(f'voxIdx')
            else:
                column_names.append(f'')
    
    # Create DataFrame
    df = pd.DataFrame(all_rows, columns=column_names)
    
    # Ensure save_folder has .csv extension
    if not save_folder.lower().endswith('.csv'):
        save_folder = save_folder + '.csv'
    
    # Save to CSV
    df.to_csv(save_folder, index=False)
    print(f"CSV file saved to: {save_folder}")
    return df