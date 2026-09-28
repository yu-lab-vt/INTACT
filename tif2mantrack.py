import os
import glob
import tifffile
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set, Optional
# import warnings
def generate_tracking_file():
    """
    Generate man_track.txt file from cell segmentation TIFF files.
    
    The function processes TIFF files in the specified folder where each file 
    represents a 2D frame with cells labeled by unique intensity values.
    It outputs a text file with four columns: cell_id, start_frame, end_frame, 0.
    """
    
    # Define the input folder path
    folder_path = r"E:\embryo_data\Neutrophils\01_GT"
    
    # Define the output file path
    output_file = os.path.join(folder_path, "man_track.txt")
    
    # Get all TIFF files matching the pattern and sort them by frame number
    tiff_files = sorted(glob.glob(os.path.join(folder_path, "Cells*.tif")))
    
    if not tiff_files:
        print(f"No TIFF files found in {folder_path}")
        return
    
    # Dictionary to track cell appearances: {cell_id: [first_frame, last_frame]}
    cell_tracks = defaultdict(lambda: [float('inf'), float('-inf')])
    
    # Process each frame
    for frame_idx, tiff_file in enumerate(tiff_files):
        try:
            # Read the TIFF file
            img = tifffile.imread(tiff_file)
            
            # Find unique cell IDs (excluding background 0)
            unique_cells = np.unique(img)
            cell_ids = unique_cells[unique_cells > 0]
            
            # Update tracking information for each cell
            for cell_id in cell_ids:
                # Convert cell_id to integer for consistent output
                cell_id_int = int(cell_id)
                
                # Update first and last appearance frames
                if frame_idx < cell_tracks[cell_id_int][0]:
                    cell_tracks[cell_id_int][0] = frame_idx
                if frame_idx > cell_tracks[cell_id_int][1]:
                    cell_tracks[cell_id_int][1] = frame_idx
                    
        except Exception as e:
            print(f"Error processing {tiff_file}: {e}")
    
    # Write results to output file
    with open(output_file, 'w') as f:
        # Sort cells by ID for consistent output
        for cell_id in sorted(cell_tracks.keys()):
            start_frame = cell_tracks[cell_id][0]
            end_frame = cell_tracks[cell_id][1]
            
            # Only include cells that were actually found
            if start_frame != float('inf') and end_frame != float('-inf'):
                f.write(f"{cell_id} {start_frame} {end_frame} 0\n")
    
    print(f"Tracking file generated: {output_file}")
    print(f"Total cells tracked: {len(cell_tracks)}")

import pandas as pd

def ultrack2csv(csv_path: str, output_path: str):
    """
    Convert ultrack CSV format to a custom format with remapped IDs.
    
    Parameters:
    -----------
    csv_path : str
        Path to input CSV file with columns: track_id, t, y, x, id, parent_track_id, parent_id
    output_path : str
        Path to save the output CSV file with columns: ID, incoming links, outgoing links, frames, x, y, z, parents
    """
    # Read the input CSV file
    df = pd.read_csv(csv_path)
    
    # Create a mapping from original 'id' to new 'ID' (1-based index)
    # New ID starts from 1 and increments for each row
    id_to_new_id = {orig_id: new_id for new_id, orig_id in enumerate(df['id'], start=1)}
    
    # Create the output dataframe with new IDs
    output_df = pd.DataFrame()
    output_df['ID'] = range(1, len(df) + 1)
    
    # Set incoming and outgoing links to 0 (not relevant)
    output_df['incoming links'] = 0
    output_df['outgoing links'] = 0
    
    # Copy the time column and add 1 to make it 1-based
    output_df['frames'] = df['t'] + 1
    
    # Copy x and y coordinates
    output_df['x'] = df['x']
    output_df['y'] = df['y']
    
    # Set z to 1 (placeholder since not in input)
    output_df['z'] = 1
    
    # Map parents using the new ID mapping
    # If original parent_id is -1, keep it as -1
    # Otherwise, find the corresponding new ID
    output_df['parents'] = df['parent_id'].apply(
        lambda x: -1 if x == -1 else id_to_new_id.get(x, -1)
    )
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)


import os
import numpy as np
import pandas as pd
import tifffile
from collections import defaultdict

def getmovieInfo(data_folder, csv_file):
    """
    Process cell tracking data and segmentation masks.
    
    Args:
        data_folder: Path to folder containing segmentation_XXXX.tif files
        csv_file: Path to tracks CSV file
        
    Returns:
        movieInfo: Dictionary containing cell information with keys:
                  xCoord, yCoord, zCoord, parents, frames, voxIdx
    """
    
    # Initialize output dictionary
    movieInfo = {
        'xCoord': [],
        'yCoord': [],
        'zCoord': [],
        'parents': [],
        'frames': [],
        'voxIdx': []
    }
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Process each row in the CSV
    for _, row in df.iterrows():
        movieInfo['xCoord'].append(row['x'])
        movieInfo['yCoord'].append(row['y'])
        movieInfo['zCoord'].append(row['z'])
        
        # Handle parent IDs: convert to -1 if needed, otherwise subtract 1
        parent = row['parents']
        if parent != -1:
            parent -= 1
        movieInfo['parents'].append(parent)
        
        # Adjust frame number (subtract 1 for 0-based indexing)
        movieInfo['frames'].append(row['frames'] - 1)
    
    # Process segmentation TIF files
    # Get all segmentation files and sort them
    seg_files = sorted([f for f in os.listdir(data_folder) 
                       if f.startswith('segmentation_') and f.endswith('.tif')])
    
    movieInfo['voxIdx'] = [[] for _ in range(len(movieInfo['frames']))]
    # Process each frame
    for frame_idx, seg_file in enumerate(seg_files):
        # Load the 3D segmentation mask
        seg_path = os.path.join(data_folder, seg_file)
        seg_data = tifffile.imread(seg_path)
        
        # For each cell that appears in this frame
        for cell_idx in range(len(movieInfo['frames'])):
            if movieInfo['frames'][cell_idx] == frame_idx:
                # Get cell coordinates
                x, y, z = (movieInfo['xCoord'][cell_idx], 
                          movieInfo['yCoord'][cell_idx], 
                          movieInfo['zCoord'][cell_idx])
                
                # Find all pixels belonging to this cell (using the cell ID as the value in segmentation)
                # Assuming segmentation mask contains cell IDs as pixel values

                target_label = seg_data[int(y)][int(x)]
                # Get mask for this cell
                cell_mask = (seg_data == target_label)
                
                # Get all voxel indices for this cell
                voxel_indices = np.where(cell_mask)
                
                # Flatten indices and convert to 1D array
                flat_indices = np.ravel_multi_index(voxel_indices, seg_data.shape)
                
                # Store in movieInfo
                movieInfo['voxIdx'][cell_idx] = flat_indices

    for key in ['xCoord', 'yCoord', 'zCoord', 'frames']:
        if movieInfo[key]:  # Check if list is not empty
            movieInfo[key] = np.array(movieInfo[key], dtype=float)
        else:
            movieInfo[key] = np.array([], dtype=float)
    movieInfo['parents'] = [int(p) for p in movieInfo['parents']]

    return movieInfo


import h5py
import numpy as np
import pickle
import scipy.io as sio

def _matlab_vox_to_zyx(vox_array):
    """
    Convert MATLAB voxel coordinate array to Python z-y-x coordinates.

    MATLAB movieInfo.vox format is assumed to be:
        [y, x, z], 1-based

    Python code expects:
        [z, y, x], 0-based

    Handles both:
        (N, 3)  and  (3, N)
    """

    arr = np.asarray(vox_array, dtype=np.float64)
    arr = np.squeeze(arr)

    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    if arr.ndim != 2:
        raise ValueError(f"Invalid vox array ndim={arr.ndim}, shape={arr.shape}")

    # MATLAB v7.3 / h5py often returns transposed arrays: (3, N)
    if arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T

    if arr.shape[1] != 3:
        raise ValueError(f"Invalid vox array shape={arr.shape}, expected (N,3) or (3,N)")

    # MATLAB [y, x, z], 1-based -> Python [z, y, x], 0-based
    arr = arr - 1
    arr = arr[:, [2, 1, 0]]

    return np.round(arr).astype(np.int64)

def matlab_struct_to_python_dict(mat_path, pkl_path, img_shape=None):
    """
    Convert MATLAB movieInfo struct to Python dictionary and save as .pkl file
    Supports both MATLAB v7.3 (HDF5) and earlier formats
    
    Parameters:
    -----------
    mat_path : str
        Path to MATLAB .mat file containing movieInfo struct
    pkl_path : str
        Path to save the output .pkl file
    """
    
    # Try to read with scipy.io first (for non-v7.3 files)
    try:
        mat_data = sio.loadmat(mat_path, mat_dtype=True)
        is_hdf5 = False
    except NotImplementedError as e:
        if 'v7.3' in str(e):
            # It's a MATLAB v7.3 file, use h5py
            is_hdf5 = True
            mat_data = h5py.File(mat_path, 'r')
        else:
            raise e
    
    if not is_hdf5:
        # Original code for non-v7.3 files
        # Extract movieInfo struct
        if 'movieInfo' in mat_data:
            movieInfo_mat = mat_data['movieInfo']
        else:
            # Try to find the struct with case-insensitive search
            for key in mat_data.keys():
                if key.lower() == 'movieinfo' and not key.startswith('__'):
                    movieInfo_mat = mat_data[key]
                    break
            else:
                raise KeyError(f"movieInfo struct not found in {mat_path}")
    else:
        # For v7.3 HDF5 files
        if 'movieInfo' in mat_data:
            movieInfo_mat = mat_data['movieInfo']
        else:
            # Try to find with case-insensitive search
            for key in mat_data.keys():
                if key.lower() == 'movieinfo':
                    movieInfo_mat = mat_data[key]
                    break
            else:
                raise KeyError(f"movieInfo struct not found in {mat_path}")
    
    # Initialize Python dictionary
    movieInfo_py = {}
    
    # Flag to indicate if vox field exists in MATLAB struct
    has_vox_field = False
    
    if not is_hdf5:
        # Non-v7.3 format (using the previous logic)
        # Check if it's a 1x1 struct with field arrays (new format)
        if movieInfo_mat.dtype.names is not None and movieInfo_mat.shape == (1, 1):
            # This is a 1x1 struct with n×1 arrays for each field
            element = movieInfo_mat[0, 0]
            field_names = element.dtype.names
            
            # Check if vox field exists
            has_vox_field = 'vox' in field_names
            
            # Extract arrays for each field
            xCoord_mat = element['xCoord'] if 'xCoord' in field_names else np.array([])
            yCoord_mat = element['yCoord'] if 'yCoord' in field_names else np.array([])
            zCoord_mat = element['zCoord'] if 'zCoord' in field_names else np.array([])
            frames_mat = element['frames'] if 'frames' in field_names else np.array([])
            voxIdx_mat = element['voxIdx'] if 'voxIdx' in field_names else np.array([])
            parents_mat = element['parents'] if 'parents' in field_names else np.array([])
            vox_mat = element['vox'] if has_vox_field else np.array([])
            
            # Get number of cells from frames array
            n_cells = frames_mat.shape[0] if frames_mat.size > 0 else 0
            
            # Batch process xCoord, yCoord, zCoord, frames
            # These are numeric arrays and can be converted in batch
            if xCoord_mat.size > 0:
                movieInfo_py['xCoord'] = xCoord_mat.flatten().tolist()
            else:
                movieInfo_py['xCoord'] = [0.0] * n_cells
                
            if yCoord_mat.size > 0:
                movieInfo_py['yCoord'] = yCoord_mat.flatten().tolist()
            else:
                movieInfo_py['yCoord'] = [0.0] * n_cells
                
            if zCoord_mat.size > 0:
                movieInfo_py['zCoord'] = zCoord_mat.flatten().tolist()
            else:
                movieInfo_py['zCoord'] = [0.0] * n_cells
            
            # frames: batch convert from 1-based to 0-based
            if frames_mat.size > 0:
                movieInfo_py['frames'] = (frames_mat.flatten() - 1).tolist()
            else:
                movieInfo_py['frames'] = []
            
            # voxIdx and parents are cell arrays, need to process individually
            movieInfo_py['voxIdx'] = []
            movieInfo_py['parents'] = []
            movieInfo_py['vox'] = []
            
            # Only loop through cell arrays
            for i in range(n_cells):
                # voxIdx: n×1 cell array, each cell contains an array
                if voxIdx_mat.size > 0:
                    voxIdx_val = voxIdx_mat[i, 0] if voxIdx_mat.shape[1] == 1 else voxIdx_mat[i]
                    if voxIdx_val.size == 0:
                        voxIdx_py = []
                    else:
                        # Flatten to 1D list
                        voxIdx_py = voxIdx_val.flatten().astype(np.int64).tolist()
                else:
                    voxIdx_py = []
                
                # parents: n×1 cell array, empty cell becomes -1
                if parents_mat.size > 0:
                    parents_val = parents_mat[i, 0] if parents_mat.shape[1] == 1 else parents_mat[i]
                    if parents_val.size == 0:
                        parents_py = -1
                    else:
                        # Convert 1-based to 0-based
                        parents_flat = parents_val.flatten()
                        if len(parents_flat) == 1:
                            parents_py = int(parents_flat[0]) - 1
                        else:
                            parents_py = (parents_flat - 1).tolist()
                else:
                    parents_py = -1
                
                # vox: n×1 cell array, each cell contains an m×3 array (yxz coordinates)
                if has_vox_field and vox_mat.size > 0:
                    vox_val = vox_mat[i, 0] if vox_mat.shape[1] == 1 else vox_mat[i]
                    if vox_val.size == 0:
                        vox_py = np.zeros((0, 3), dtype=np.int64)
                    else:
                        vox_py = _matlab_vox_to_zyx(vox_val)
                else:
                    vox_py = np.zeros((0, 3), dtype=np.int64)
                
                movieInfo_py['voxIdx'].append(voxIdx_py)
                movieInfo_py['parents'].append(parents_py)
                movieInfo_py['vox'].append(vox_py)
        else:
            # Fallback to original format (struct array)
            raise ValueError("Only 1x1 struct format is supported for non-v7.3 files")
    
    else:
        # HDF5 v7.3 format
        # For v7.3 files, MATLAB structures are stored as HDF5 groups
        # Each field is a dataset within the group
        
        # Get the fields from the movieInfo group
        xCoord_ds = movieInfo_mat['xCoord'] if 'xCoord' in movieInfo_mat else None
        yCoord_ds = movieInfo_mat['yCoord'] if 'yCoord' in movieInfo_mat else None
        zCoord_ds = movieInfo_mat['zCoord'] if 'zCoord' in movieInfo_mat else None
        frames_ds = movieInfo_mat['frames'] if 'frames' in movieInfo_mat else None
        voxIdx_ds = movieInfo_mat['voxIdx'] if 'voxIdx' in movieInfo_mat else None
        parents_ds = movieInfo_mat['parents'] if 'parents' in movieInfo_mat else None
        vox_ds = movieInfo_mat['vox'] if 'vox' in movieInfo_mat else None
        
        # Check if vox field exists
        has_vox_field = vox_ds is not None
        
        # Get number of cells
        n_cells = 0
        if frames_ds is not None:
            frames_data = frames_ds[()][0]
            n_cells = len(frames_data) if len(frames_data.shape) > 0 else 1
        
        # Batch process xCoord, yCoord, zCoord, frames
        if frames_ds is not None:
            frames_data = frames_ds[()][0]
            # Batch convert to list and subtract 1
            if frames_data.ndim == 2:
                movieInfo_py['frames'] = (frames_data[:, 0] - 1).tolist()
            elif frames_data.ndim == 1:
                movieInfo_py['frames'] = (frames_data - 1).tolist()
            else:
                movieInfo_py['frames'] = [int(frames_data[()]) - 1]
        else:
            movieInfo_py['frames'] = []
        
        if xCoord_ds is not None:
            xCoord_data = xCoord_ds[()][0]
            if xCoord_data.ndim == 2:
                movieInfo_py['xCoord'] = xCoord_data[:, 0].tolist()
            elif xCoord_data.ndim == 1:
                movieInfo_py['xCoord'] = xCoord_data.tolist()
            else:
                movieInfo_py['xCoord'] = [float(xCoord_data[()])]
        else:
            movieInfo_py['xCoord'] = [0.0] * n_cells
            
        if yCoord_ds is not None:
            yCoord_data = yCoord_ds[()][0]
            if yCoord_data.ndim == 2:
                movieInfo_py['yCoord'] = yCoord_data[:, 0].tolist()
            elif yCoord_data.ndim == 1:
                movieInfo_py['yCoord'] = yCoord_data.tolist()
            else:
                movieInfo_py['yCoord'] = [float(yCoord_data[()])]
        else:
            movieInfo_py['yCoord'] = [0.0] * n_cells
            
        if zCoord_ds is not None:
            zCoord_data = zCoord_ds[()][0]
            if zCoord_data.ndim == 2:
                movieInfo_py['zCoord'] = zCoord_data[:, 0].tolist()
            elif zCoord_data.ndim == 1:
                movieInfo_py['zCoord'] = zCoord_data.tolist()
            else:
                movieInfo_py['zCoord'] = [float(zCoord_data[()])]
        else:
            movieInfo_py['zCoord'] = [0.0] * n_cells
        
        # Initialize lists
        movieInfo_py['voxIdx'] = []
        movieInfo_py['parents'] = []
        movieInfo_py['vox'] = []
        
        # Process voxIdx references using list comprehension
        if voxIdx_ds is not None:
            voxIdx_data = voxIdx_ds[()][0]
            # Flatten reference array
            voxIdx_refs = voxIdx_data.flatten() if voxIdx_data.ndim >= 1 else np.array([voxIdx_data[()]])
            
            # Use list comprehension to process all references
            for ref in voxIdx_refs:
                # Check if it's a valid h5py reference object
                if isinstance(ref, h5py.Reference) and ref:
                    try:
                        voxIdx_cell = mat_data[ref]
                        movieInfo_py['voxIdx'].append(voxIdx_cell[()].flatten().astype(int).tolist())
                    except:
                        movieInfo_py['voxIdx'].append([])
                else:
                    movieInfo_py['voxIdx'].append([])
        else:
            movieInfo_py['voxIdx'] = [[]] * n_cells

        # Process parents references
        if parents_ds is not None:
            parents_data = parents_ds[()][0]
            
            # Simplify processing: only focus on first reference
            for i in range(n_cells):
                # Default value
                parent_val = -1
                
                # Get reference (try different dimensions)
                try:
                    if parents_data.ndim == 1:
                        ref = parents_data[i] if i < len(parents_data) else None
                    else:
                        # Take first element of each row
                        ref = parents_data[i, 0] if i < parents_data.shape[0] else None
                except:
                    ref = None
                
                # Process reference
                if ref is not None and isinstance(ref, h5py.Reference):
                    try:
                        cell_data = mat_data[ref][()]
                        if cell_data.size > 0:
                            # Convert to 0-based index
                            parent_val = int(cell_data.flat[0]) - 1
                    except:
                        pass  # Keep -1
                
                movieInfo_py['parents'].append(parent_val)
        else:
            movieInfo_py['parents'] = [-1] * n_cells
            
        # Process vox field if it exists
        if has_vox_field:
            vox_data = vox_ds[()][0]
            vox_refs = vox_data.flatten() if vox_data.ndim >= 1 else np.array([vox_data[()]])
            
            for ref in vox_refs:
                if isinstance(ref, h5py.Reference) and ref:
                    try:
                        vox_cell = mat_data[ref]
                        vox_array = vox_cell[()]
                        vox_py = _matlab_vox_to_zyx(vox_array)
                        movieInfo_py['vox'].append(vox_py)
                    except:
                        movieInfo_py['vox'].append(np.zeros((0, 3)))
                else:
                    movieInfo_py['vox'].append(np.zeros((0, 3)))
        else:
            movieInfo_py['vox'] = [np.zeros((0, 3))] * n_cells
            
        # Close the HDF5 file
        mat_data.close()
    
    # Convert to numpy arrays
    movieInfo_py['frames'] = np.array(movieInfo_py['frames'], dtype=int)
    movieInfo_py['xCoord'] = np.array(movieInfo_py['xCoord'], dtype=float)
    movieInfo_py['yCoord'] = np.array(movieInfo_py['yCoord'], dtype=float)
    movieInfo_py['zCoord'] = np.array(movieInfo_py['zCoord'], dtype=float)
    movieInfo_py['parents'] = np.array(movieInfo_py['parents'], dtype=int)
    
    # If vox field doesn't exist in MATLAB struct, compute from voxIdx using img_shape
    if not has_vox_field and img_shape is not None:
        vox_list = []

        if len(img_shape) == 3:
            Z, Y, X = img_shape

            for flat_idx_list in movieInfo_py['voxIdx']:
                if len(flat_idx_list) > 0:
                    idx_arr = np.asarray(flat_idx_list, dtype=np.int64)
                    z, x, y = np.unravel_index(idx_arr - 1, (Z, X, Y))
                    vox_list.append(np.column_stack((z, y, x)).astype(np.int64))
                else:
                    vox_list.append(np.zeros((0, 3), dtype=np.int64))

        elif len(img_shape) == 2:
            H, W = img_shape

            for flat_idx_list in movieInfo_py['voxIdx']:
                if len(flat_idx_list) > 0:
                    idx_arr = np.asarray(flat_idx_list, dtype=np.int64)
                    x, y = np.unravel_index(idx_arr - 1, (W, H))
                    vox_list.append(np.column_stack((y, x)).astype(np.int64))
                else:
                    vox_list.append(np.zeros((0, 2), dtype=np.int64))

        movieInfo_py['vox'] = vox_list

    # Save as pickle file
    with open(pkl_path, 'wb') as f:
        pickle.dump(movieInfo_py, f)
    
    print(f"Successfully converted MATLAB struct to Python dictionary and saved to {pkl_path}")
    return movieInfo_py

def python_dict_to_matlab_struct(pkl_path, mat_path):
    """
    Convert Python movieInfo dictionary to MATLAB struct (1x1 struct with n×1 arrays) and save as .mat file
    
    Parameters:
    -----------
    pkl_path : str
        Path to Python .pkl file containing movieInfo dictionary
    mat_path : str
        Path to save the output .mat file
    """
    
    # Load Python data
    with open(pkl_path, 'rb') as f:
        movieInfo_py = pickle.load(f)
    
    # Get the number of cells
    n_cells = len(movieInfo_py['frames'])
    
    # Prepare arrays for each field
    # xCoord, yCoord, zCoord: n×1 double arrays
    xCoord_mat = np.zeros((n_cells, 1), dtype=np.float64)
    yCoord_mat = np.zeros((n_cells, 1), dtype=np.float64)
    zCoord_mat = np.zeros((n_cells, 1), dtype=np.float64)
    
    # frames: n×1 double array (convert 0-based to 1-based)
    frames_mat = np.zeros((n_cells, 1), dtype=np.float64)
    
    # voxIdx: n×1 cell array (each cell contains a list/array)
    voxIdx_mat = np.empty((n_cells, 1), dtype=np.object_)
    
    # parents: n×1 cell array (each cell contains a list/array or empty)
    parents_mat = np.empty((n_cells, 1), dtype=np.object_)
    
    # Fill the arrays
    for i in range(n_cells):
        # Get Python data
        xCoord_val = movieInfo_py['xCoord'][i]
        yCoord_val = movieInfo_py['yCoord'][i]
        zCoord_val = movieInfo_py['zCoord'][i]
        frames_val = movieInfo_py['frames'][i]
        voxIdx_val = movieInfo_py['voxIdx'][i]
        parents_val = movieInfo_py['parents'][i]
        
        # xCoord, yCoord, zCoord: convert to scalar
        # Assuming these are single values, not lists
        if isinstance(xCoord_val, (list, np.ndarray)) and len(xCoord_val) > 0:
            xCoord_mat[i, 0] = float(xCoord_val[0])
        else:
            xCoord_mat[i, 0] = float(xCoord_val) if xCoord_val is not None else 0.0
            
        if isinstance(yCoord_val, (list, np.ndarray)) and len(yCoord_val) > 0:
            yCoord_mat[i, 0] = float(yCoord_val[0])
        else:
            yCoord_mat[i, 0] = float(yCoord_val) if yCoord_val is not None else 0.0
            
        if isinstance(zCoord_val, (list, np.ndarray)) and len(zCoord_val) > 0:
            zCoord_mat[i, 0] = float(zCoord_val[0])
        else:
            zCoord_mat[i, 0] = float(zCoord_val) if zCoord_val is not None else 0.0
        
        # frames: convert 0-based to 1-based
        if isinstance(frames_val, (list, np.ndarray)) and len(frames_val) > 0:
            frames_mat[i, 0] = float(frames_val[0]) + 1
        else:
            frames_mat[i, 0] = float(frames_val) + 1 if frames_val is not None else 1.0
        
        # voxIdx: store as MATLAB cell (list -> numpy array)
        if isinstance(voxIdx_val, list):
            if len(voxIdx_val) > 0:
                voxIdx_mat[i, 0] = np.array(voxIdx_val, dtype=np.float64).reshape(-1, 1)
            else:
                voxIdx_mat[i, 0] = np.array([], dtype=np.float64).reshape(0, 1)
        elif isinstance(voxIdx_val, np.ndarray):
            voxIdx_mat[i, 0] = voxIdx_val.astype(np.float64).reshape(-1, 1)
        else:
            voxIdx_mat[i, 0] = np.array([], dtype=np.float64).reshape(0, 1)
        
        # parents: convert -1 to empty array, 0-based to 1-based
        if parents_val == -1 or (isinstance(parents_val, list) and len(parents_val) == 0):
            parents_mat[i, 0] = np.array([], dtype=np.float64)
        elif isinstance(parents_val, list) and len(parents_val) > 0:
            # Convert to 1-based indexing
            parents_mat[i, 0] = np.array(parents_val, dtype=np.float64) + 1
        elif isinstance(parents_val, (int, float, np.int64, np.float64)):
            # Single parent, convert to 1-based
            parents_mat[i, 0] = np.array([parents_val + 1], dtype=np.float64)
        elif isinstance(parents_val, np.ndarray):
            # Already a numpy array
            parents_mat[i, 0] = parents_val.astype(np.float64) + 1
        else:
            parents_mat[i, 0] = np.array([], dtype=np.float64)
    
    # Create MATLAB struct as a 1x1 struct with fields as n×1 arrays
    # We'll create a dictionary that will be saved as a MATLAB struct
    movieInfo_struct = {
        'xCoord': xCoord_mat,
        'yCoord': yCoord_mat,
        'zCoord': zCoord_mat,
        'frames': frames_mat,
        'voxIdx': voxIdx_mat,
        'parents': parents_mat,
        'orgCoord': np.hstack([xCoord_mat, yCoord_mat, zCoord_mat])
    }
    
    # Save as .mat file
    sio.savemat(mat_path, {'movieInfo': movieInfo_struct})
    
    print(f"Successfully converted Python dictionary to 1x1 MATLAB struct and saved to {mat_path}")
    return movieInfo_struct

def save_tif(array, file_path, compress=0):
    """
    Save a 2D or 3D array as a TIFF file
    
    Parameters:
    - array: Input array, can be numpy array or cupy array, supports 2D or 3D
    - file_path: File path to save the TIFF
    - compress: Compression level (0-9), 0 means no compression, default is 0
    
    Returns:
    - Returns True if successful, False otherwise
    """
    try:
        # Check file extension
        if not file_path.lower().endswith(('.tif', '.tiff')):
            file_path += '.tiff'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Handle cupy arrays
        try:
            import cupy as cp
            if isinstance(array, cp.ndarray):
                array = cp.asnumpy(array)  # Convert to numpy array
                print("Detected cupy array, automatically converted to numpy")
        except ImportError:
            pass
        
        # Validate array dimensions
        if array.ndim not in [2, 3]:
            raise ValueError(f"Unsupported array dimension: {array.ndim}. Only 2D or 3D arrays are supported")
        
        # Handle data types
        dtype = array.dtype
        supported_dtypes = [np.uint8, np.uint16, np.uint32, np.uint64,
                           np.int8, np.int16, np.int32, np.int64,
                           np.float32, np.float64, bool]
        
        if dtype not in supported_dtypes:
            warnings.warn(f"Data type {dtype} may not be fully supported by all TIFF readers")
        
        # Save as TIFF
        if compress > 0:
            tifffile.imwrite(file_path, array, compression=compress)
        else:
            tifffile.imwrite(file_path, array)
        
        print(f"Array successfully saved as: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error saving TIFF file: {str(e)}")
        return False
    

def create_gt_images_from_csv(
        
    spot_file: str,
    link_file: str,
    img_size: Tuple[int, int],
    output_path: str
) -> None:
    """
    Create track visualization images from cell tracking data stored in CSV files.
    
    Parameters:
    -----------
    spot_file : str
        Path to Spot CSV file with columns: X, Y, Z, Spot frame, ID
    link_file : str
        Path to Link CSV file with columns: Source spot id, Target spot id
    img_size : Tuple[int, int]
        Size of output images (height, width) for 2D or (depth, height, width) for 3D
    output_path : str
        Directory path to save output files
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # ===========================================================================
    # Step 1: Read and process CSV files to build movieInfo structure
    # ===========================================================================
    
    # Read Spot CSV file
    spot_df = pd.read_csv(spot_file,encoding='gbk')
    spot_df = spot_df[(spot_df['Spot frame'] >= 150) & (spot_df['Spot frame'] <= 199)]
    # Ensure column names match enpected format (case-insensitive)
    spot_df.columns = spot_df.columns.str.strip().str.lower()
    retained_spot_ids = set(spot_df['id'].astype(int).tolist())
    # Read Link CSV file
    link_df = pd.read_csv(link_file,encoding='gbk')
    link_df = link_df[(link_df['Source spot id'].isin(retained_spot_ids)) & 
                  (link_df['Target spot id'].isin(retained_spot_ids))]
    link_df.columns = link_df.columns.str.strip().str.lower()
    
    # Create movieInfo structure similar to original function
    movieInfo = {
        'frames': [],
        'xCoord': [],
        'yCoord': [],
        'zCoord': [],
        'parents': []
    }
    
    # Create mapping from spot ID to index in movieInfo arrays
    spot_id_to_index = {}
    
    # Process each spot
    for idx, row in spot_df.iterrows():
        # Convert coordinates to appropriate format
        x = float(row['x'])
        y = float(row['y'])
        z = float(row['z']) if 'z' in row else 0.0
        frame = int(row['spot frame'])
        spot_id = int(row['id'])
        
        # Store mapping from spot ID to array index
        spot_id_to_index[spot_id] = len(movieInfo['frames'])
        
        # Append to movieInfo arrays
        movieInfo['frames'].append(frame)
        movieInfo['xCoord'].append(x)
        movieInfo['yCoord'].append(y)
        movieInfo['zCoord'].append(z)
        movieInfo['parents'].append([])  # Initialize empty parent list
    
    # Convert lists to numpy arrays
    movieInfo['frames'] = np.array(movieInfo['frames']) - 150
    movieInfo['xCoord'] = np.array(movieInfo['xCoord'])
    movieInfo['yCoord'] = np.array(movieInfo['yCoord'])
    movieInfo['zCoord'] = np.array(movieInfo['zCoord'])
    
    # Process links to set parent relationships
    for _, row in link_df.iterrows():
        source_id = int(row['source spot id'])
        target_id = int(row['target spot id'])
        
        if source_id in spot_id_to_index and target_id in spot_id_to_index:
            target_idx = spot_id_to_index[target_id]
            # In Link file, target spot id's parent is source spot id
            movieInfo['parents'][target_idx] = source_id
    
    # Convert parent spot IDs to parent indices in movieInfo arrays
    for idx, parent_list in enumerate(movieInfo['parents']):
        if isinstance(parent_list, list) and len(parent_list) > 0:
            # Convert spot ID to array index
            parent_spot_id = parent_list[0]
            if parent_spot_id in spot_id_to_index:
                movieInfo['parents'][idx] = spot_id_to_index[parent_spot_id]
            else:
                movieInfo['parents'][idx] = -1
        elif isinstance(parent_list, (int, np.integer)) and parent_list > 0:
            parent_spot_id = parent_list
            if parent_spot_id in spot_id_to_index:
                movieInfo['parents'][idx] = spot_id_to_index[parent_spot_id]
            else:
                movieInfo['parents'][idx] = -1
        else:
            movieInfo['parents'][idx] = -1
    
    # ===========================================================================
    # Step 2: Track assignment and event recording (same as original)
    # ===========================================================================
    
    # Initialize data structures
    frames_list = movieInfo['frames'].astype(int)
    total_frames = max(frames_list) + 1 if len(frames_list) > 0 else 0
    split_events = []
    current_track_colors = {}
    next_gray_value = 1
    cell_to_track = {}
    event_table = {}
    
    # Group cells by frame for easier processing
    cells_by_frame = [[] for _ in range(total_frames)]
    for cell_idx, frame in enumerate(frames_list):
        if frame < total_frames:
            cells_by_frame[frame].append(cell_idx)
    
    # First pass: Assign track IDs and handle splits
    for frame in range(total_frames):
        cell_indices_in_frame = cells_by_frame[frame]
        
        for cell_idx in cell_indices_in_frame:
            parent_list = movieInfo['parents'][cell_idx]
            cell_key = (frame, cell_idx)
            
            # Check if cell has parent
            has_parent = False
            if isinstance(parent_list, (int, np.integer)):
                has_parent = parent_list >= 0
                if has_parent:
                    parent_idx = parent_list
            elif isinstance(parent_list, (list, np.ndarray)):
                has_parent = len(parent_list) > 0
                if has_parent:
                    parent_idx = parent_list[0] if len(parent_list) > 0 else -1
            else:
                has_parent = False
            
            if not has_parent:
                # No parent - start new track
                track_id = len(current_track_colors) + 1
                current_track_colors[track_id] = next_gray_value
                event_table[next_gray_value] = [frame, frame, 0]
                next_gray_value += 1
                cell_to_track[cell_key] = track_id
            else:
                # Has parent
                parent_frame = frame - 1
                parent_key = (parent_frame, parent_idx)
                
                if parent_key in cell_to_track:
                    division_flag = False
                    parent_track_id = cell_to_track[parent_key]
                    
                    # Check if this is a division (multiple children from same parent in same frame)
                    for other_cell_idx in cell_indices_in_frame:
                        if other_cell_idx != cell_idx:
                            other_parent_list = movieInfo['parents'][other_cell_idx]
                            if isinstance(other_parent_list, (int, np.integer)):
                                if other_parent_list == parent_idx:
                                    division_flag = True
                            elif isinstance(other_parent_list, (list, np.ndarray)):
                                if len(other_parent_list) > 0 and other_parent_list[0] == parent_idx:
                                    division_flag = True
                    
                    if division_flag:
                        # Second or later child - new track, record split
                        new_track_id = len(current_track_colors) + 1
                        current_track_colors[new_track_id] = next_gray_value
                        parent_gray = current_track_colors[parent_track_id]
                        event_table[next_gray_value] = [frame, frame, parent_gray]
                        split_events.append([
                            next_gray_value,
                            parent_frame,
                            frame,
                            parent_gray
                        ])
                        cell_to_track[cell_key] = new_track_id
                        next_gray_value += 1
                    else:
                        # First child - inherit parent's track
                        cell_to_track[cell_key] = parent_track_id
                        gray_value = current_track_colors[parent_track_id]
                        # Update end frame for this track
                        event_table[gray_value][1] = frame
                else:
                    # Parent not found or not in cell_to_track - start new track
                    track_id = len(current_track_colors) + 1
                    current_track_colors[track_id] = next_gray_value
                    event_table[next_gray_value] = [frame, frame, 0]
                    next_gray_value += 1
                    cell_to_track[cell_key] = track_id
    
    # ===========================================================================
    # Step 3: Create images using coordinate-based approach
    # ===========================================================================
    
    for frame in range(total_frames):
        # Create empty image
        if len(img_size) == 3:
            img = np.zeros(img_size, dtype=np.uint16)  # 3D: (depth, height, width)
        else:
            img = np.zeros(img_size, dtype=np.uint16)  # 2D: (height, width)
        
        # Get all cells in this frame
        cell_indices_in_frame = cells_by_frame[frame]
        
        for cell_idx in cell_indices_in_frame:
            if (frame, cell_idx) not in cell_to_track:
                continue
                
            track_id = cell_to_track[(frame, cell_idx)]
            gray_value = current_track_colors[track_id]
            
            # Get coordinates for this cell
            x = movieInfo['xCoord'][cell_idx]
            y = movieInfo['yCoord'][cell_idx]
            z = movieInfo['zCoord'][cell_idx] if len(img_size) == 3 else 0
            
            # Round coordinates to nearest integer
            x_int = int(round(x))
            y_int = int(round(y))
            
            # Handle 3D case
            if len(img_size) == 3:
                z_int = int(round(z))
                # Check bounds
                if (0 <= z_int < img_size[0] and 
                    0 <= y_int < img_size[1] and 
                    0 <= x_int < img_size[2]):
                    img[z_int, y_int, x_int] = gray_value
            # Handle 2D case
            else:
                # Check bounds (img_size is (height, width))
                if 0 <= y_int < img_size[0] and 0 <= x_int < img_size[1]:
                    img[y_int, x_int] = gray_value
        
        # Save image
        filename = os.path.join(output_path, f"man_track{frame:03d}.tif")
        tifffile.imwrite(filename, img)
    
    # ===========================================================================
    # Step 4: Save events to text file (renamed to man_track.txt)
    # ===========================================================================
    
    if event_table:
        events_list = []
        for gray_value, (start_frame, end_frame, parent_gray) in event_table.items():
            events_list.append([gray_value, start_frame, end_frame, parent_gray])
        
        events_array = np.array(events_list, dtype=np.uint32)
        events_path = os.path.join(output_path, "man_track.txt")
        np.savetxt(events_path, events_array, fmt='%d', delimiter=' ')
        
        print(f"Recorded {len(event_table)} trajectory events")
    
    print(f"Successfully saved {total_frames} images to {output_path}")
    if split_events:
        print(f"Recorded {len(split_events)} split events")



import numpy as np
import tifffile
from typing import Union
import warnings

def randomize_cell_ids(tif_path: str, output_path: str = None, 
                       seed: int = None, preserve_background: bool = True) -> Union[np.ndarray, None]:
    """
    Randomly reassign cell IDs in a 2D or 3D TIFF segmentation mask.
    
    Parameters:
    -----------
    tif_path : str
        Path to input TIFF file containing segmentation mask.
        Background should be 0, cell regions positive integers.
    output_path : str, optional
        Path to save the randomized TIFF. If None, file is not saved.
    seed : int, optional
        Random seed for reproducibility.
    preserve_background : bool, default=True
        If True, keeps background as 0. If False, background may get reassigned.
        
    Returns:
    --------
    np.ndarray or None
        Randomized segmentation array if output_path is provided, else None.
    """
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Read TIFF file
    try:
        segmentation = tifffile.imread(tif_path)
    except Exception as e:
        raise IOError(f"Failed to read TIFF file: {e}")
    
    # Validate input data
    if segmentation.dtype not in [np.uint8, np.uint16, np.uint32, np.int32, np.int64]:
        warnings.warn(f"Input dtype {segmentation.dtype} may not be optimal for segmentation masks. "
                      "Consider using uint16 or uint32.")
    
    # Get unique cell IDs (exclude background if preserve_background is True)
    unique_ids = np.unique(segmentation)
    
    if preserve_background:
        # Exclude background (0) from randomization
        unique_ids = unique_ids[unique_ids != 0]
    
    if len(unique_ids) == 0:
        warnings.warn("No cell IDs found in the segmentation mask.")
        if output_path:
            tifffile.imwrite(output_path, segmentation)
        return segmentation if output_path is None else None
    
    # Create mapping from original IDs to new random IDs
    # Generate a random permutation of the unique IDs
    random_ids = np.random.permutation(unique_ids)
    
    # Create a mapping dictionary for efficient lookup
    id_mapping = {orig: new for orig, new in zip(unique_ids, random_ids)}
    
    # Apply the mapping to the segmentation array
    # Vectorized approach using numpy indexing
    randomized_seg = np.zeros_like(segmentation)
    
    # For each original ID, assign the new random ID
    for orig_id, new_id in id_mapping.items():
        randomized_seg[segmentation == orig_id] = new_id
    
    # If not preserving background, ensure 0 remains 0 (no cell should be 0)
    if not preserve_background:
        # Find what new ID was assigned to original 0 (if it existed)
        if 0 in id_mapping:
            # The cell that was originally 0 now has a new ID
            # But we want to keep actual background as 0
            # This is a corner case when preserve_background=False
            # We'll leave it as is since user explicitly set preserve_background=False
            pass
    
    # Save to file if output path provided
    if output_path:
        try:
            tifffile.imwrite(output_path, randomized_seg)
            print(f"Randomized segmentation saved to: {output_path}")
        except Exception as e:
            raise IOError(f"Failed to write TIFF file: {e}")
    
    return randomized_seg if output_path is None else None


def randomize_cell_ids_inplace(segmentation: np.ndarray, seed: int = None, 
                               preserve_background: bool = True) -> np.ndarray:
    """
    Randomly reassign cell IDs in a segmentation array (in-place operation).
    
    Parameters:
    -----------
    segmentation : np.ndarray
        Input segmentation array.
    seed : int, optional
        Random seed for reproducibility.
    preserve_background : bool, default=True
        If True, keeps background as 0.
        
    Returns:
    --------
    np.ndarray
        Randomized segmentation array.
    """
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Get unique cell IDs
    unique_ids = np.unique(segmentation)
    
    if preserve_background:
        unique_ids = unique_ids[unique_ids != 0]
    
    if len(unique_ids) == 0:
        return segmentation
    
    # Create mapping and apply
    random_ids = np.random.permutation(unique_ids)
    id_mapping = {orig: new for orig, new in zip(unique_ids, random_ids)}
    
    # Create output array
    randomized_seg = np.zeros_like(segmentation)
    
    # Apply mapping
    for orig_id, new_id in id_mapping.items():
        randomized_seg[segmentation == orig_id] = new_id
    
    return randomized_seg

import numpy as np
import tifffile
from typing import Union, List, Dict, Optional

def remap_cell_ids(
    tif_path: str,
    output_path: str = None,
    id_mapping: Optional[Dict[int, int]] = None,
    id_list: Optional[List[int]] = None,
    new_values: Optional[List[int]] = None,
    preserve_unmapped: bool = True,
    preserve_background: bool = True
) -> Union[np.ndarray, None]:
    """
    Remap specific cell IDs to new values in a segmentation TIFF.
    
    Parameters:
    -----------
    tif_path : str
        Path to input TIFF segmentation file.
    output_path : str, optional
        Path to save the remapped TIFF. If None, file is not saved.
    id_mapping : dict, optional
        Dictionary mapping original IDs to new values.
        e.g., {1: 10, 2: 20, 3: 30}
    id_list : list, optional
        List of original IDs to remap (alternative to id_mapping).
    new_values : list, optional
        List of new values corresponding to id_list (must be same length as id_list).
    preserve_unmapped : bool, default=True
        If True, IDs not in mapping keep their original values.
        If False, IDs not in mapping are set to 0 (background).
    preserve_background : bool, default=True
        If True, background (0) is always preserved unless explicitly remapped.
        
    Returns:
    --------
    np.ndarray or None
        Remapped segmentation array if output_path is provided, else None.
        
    Raises:
    -------
    ValueError:
        If id_mapping and (id_list, new_values) are both provided or both missing,
        or if id_list and new_values have different lengths.
    """
    
    # Validate input parameters
    if id_mapping is not None and (id_list is not None or new_values is not None):
        raise ValueError("Provide either id_mapping OR (id_list and new_values), not both.")
    
    if id_mapping is None and (id_list is None or new_values is None):
        raise ValueError("Provide either id_mapping OR both id_list and new_values.")
    
    if id_list is not None and new_values is not None:
        if len(id_list) != len(new_values):
            raise ValueError("id_list and new_values must have the same length.")
        # Convert lists to dictionary
        id_mapping = {orig: new for orig, new in zip(id_list, new_values)}
    
    # Read the segmentation TIFF
    try:
        segmentation = tifffile.imread(tif_path)
    except Exception as e:
        raise IOError(f"Failed to read TIFF file: {e}")
    
    # Create a copy of the segmentation array
    remapped = segmentation.copy()
    
    # Get all unique IDs in the image
    all_ids = np.unique(segmentation)
    
    # Check for potential ID conflicts
    if preserve_background and 0 in id_mapping:
        print("Warning: Background (0) is being remapped. Set preserve_background=False to suppress this warning.")
    
    # Apply the ID mapping
    for orig_id, new_id in id_mapping.items():
        # Find where the original ID appears
        mask = segmentation == orig_id
        
        # Skip if this ID doesn't exist in the image
        if not np.any(mask):
            print(f"Warning: ID {orig_id} not found in the segmentation.")
            continue
        
        # Apply the remapping
        remapped[mask] = new_id
    
    # Handle unmapped IDs
    if not preserve_unmapped:
        # Get all IDs that exist in the original image
        existing_ids = np.unique(segmentation)
        
        # For each existing ID, if it's not in the mapping and not background (unless background is being changed)
        for orig_id in existing_ids:
            if orig_id not in id_mapping and (orig_id != 0 or not preserve_background):
                mask = segmentation == orig_id
                remapped[mask] = 0
    
    # Check for ID collisions (different original IDs mapped to same new ID)
    unique_new_ids = list(id_mapping.values())
    if len(set(unique_new_ids)) < len(unique_new_ids):
        print("Warning: Multiple original IDs are mapped to the same new ID. This may merge cell regions.")
    
    # Save to file if output path provided
    if output_path:
        try:
            tifffile.imwrite(output_path, remapped)
            print(f"Remapped segmentation saved to: {output_path}")
        except Exception as e:
            raise IOError(f"Failed to write TIFF file: {e}")
    
    return remapped if output_path is None else None


def remap_cell_ids_inplace(
    segmentation: np.ndarray,
    id_mapping: Optional[Dict[int, int]] = None,
    id_list: Optional[List[int]] = None,
    new_values: Optional[List[int]] = None,
    preserve_unmapped: bool = True,
    preserve_background: bool = True
) -> np.ndarray:
    """
    Remap specific cell IDs to new values in a segmentation array (in-place operation).
    
    Parameters:
    -----------
    segmentation : np.ndarray
        Input segmentation array.
    id_mapping : dict, optional
        Dictionary mapping original IDs to new values.
    id_list : list, optional
        List of original IDs to remap.
    new_values : list, optional
        List of new values corresponding to id_list.
    preserve_unmapped : bool, default=True
        If True, IDs not in mapping keep their original values.
    preserve_background : bool, default=True
        If True, background (0) is always preserved.
        
    Returns:
    --------
    np.ndarray
        Remapped segmentation array.
    """
    
    # Validate input parameters
    if id_mapping is not None and (id_list is not None or new_values is not None):
        raise ValueError("Provide either id_mapping OR (id_list and new_values), not both.")
    
    if id_mapping is None and (id_list is None or new_values is None):
        raise ValueError("Provide either id_mapping OR both id_list and new_values.")
    
    if id_list is not None and new_values is not None:
        if len(id_list) != len(new_values):
            raise ValueError("id_list and new_values must have the same length.")
        id_mapping = {orig: new for orig, new in zip(id_list, new_values)}
    
    # Create a copy to avoid modifying the original
    remapped = segmentation.copy()
    
    # Apply the ID mapping
    for orig_id, new_id in id_mapping.items():
        mask = segmentation == orig_id
        if not np.any(mask):
            continue
        remapped[mask] = new_id
    
    # Handle unmapped IDs
    if not preserve_unmapped:
        existing_ids = np.unique(segmentation)
        for orig_id in existing_ids:
            if orig_id not in id_mapping and (orig_id != 0 or not preserve_background):
                mask = segmentation == orig_id
                remapped[mask] = 0
    
    return remapped


def remap_ids_with_auto_gap_filling(
    tif_path: str,
    output_path: str = None,
    id_list: List[int] = None,
    start_id: int = 1,
    compact_ids: bool = True
) -> Union[np.ndarray, None]:
    """
    Remap specific IDs and optionally fill gaps or compact all IDs.
    
    Parameters:
    -----------
    tif_path : str
        Path to input TIFF segmentation file.
    output_path : str, optional
        Path to save the remapped TIFF.
    id_list : list, optional
        List of IDs to remap in order. If None, all non-zero IDs are remapped.
    start_id : int, default=1
        Starting ID for the new numbering.
    compact_ids : bool, default=True
        If True, remap all IDs to a compact sequence (1, 2, 3, ...).
        If False, only remap the IDs in id_list.
        
    Returns:
    --------
    np.ndarray or None
        Remapped segmentation array.
    """
    
    # Read the segmentation
    segmentation = tifffile.imread(tif_path)
    
    # Get all unique non-zero IDs
    all_ids = np.unique(segmentation)
    all_ids = all_ids[all_ids != 0]
    
    # If id_list not provided, use all non-zero IDs
    if id_list is None:
        id_list = sorted(all_ids)
    
    # Create mapping based on the desired operation
    id_mapping = {}
    
    if compact_ids:
        # Create compact mapping for all non-zero IDs
        for i, orig_id in enumerate(sorted(all_ids), start=start_id):
            id_mapping[orig_id] = i
    else:
        # Only remap the specified IDs
        for i, orig_id in enumerate(id_list, start=start_id):
            id_mapping[orig_id] = i
    
    # Apply the remapping
    remapped = segmentation.copy()
    for orig_id, new_id in id_mapping.items():
        mask = segmentation == orig_id
        if np.any(mask):
            remapped[mask] = new_id
    
    # Save if output path provided
    if output_path:
        tifffile.imwrite(output_path, remapped)
        print(f"Remapped segmentation saved to: {output_path}")
    
    return remapped if output_path is None else None


import numpy as np
import tifffile
from typing import Union
import warnings

def randomize_cell_ids_fast(
    tif_path: str, 
    output_path: str = None, 
    seed: int = None, 
    preserve_background: bool = True,
    memory_efficient: bool = False
) -> Union[np.ndarray, None]:
    """
    Fast version: Randomly reassign cell IDs in a 2D or 3D TIFF segmentation mask.
    Uses vectorized operations for better performance with large images.
    
    Parameters:
    -----------
    tif_path : str
        Path to input TIFF file containing segmentation mask.
    output_path : str, optional
        Path to save the randomized TIFF.
    seed : int, optional
        Random seed for reproducibility.
    preserve_background : bool, default=True
        If True, keeps background as 0.
    memory_efficient : bool, default=False
        If True, uses a more memory-efficient method that may be slower for small images.
        If False, uses the fastest method that may require more memory.
        
    Returns:
    --------
    np.ndarray or None
        Randomized segmentation array.
    """
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Read TIFF file
    try:
        segmentation = tifffile.imread(tif_path)
    except Exception as e:
        raise IOError(f"Failed to read TIFF file: {e}")
    
    # Call the optimized in-place function
    randomized_seg = randomize_cell_ids_inplace_fast(
        segmentation, 
        seed, 
        preserve_background, 
        memory_efficient
    )
    
    # Save to file if output path provided
    if output_path:
        try:
            tifffile.imwrite(output_path, randomized_seg)
            print(f"Randomized segmentation saved to: {output_path}")
        except Exception as e:
            raise IOError(f"Failed to write TIFF file: {e}")
    
    return randomized_seg if output_path is None else None


def randomize_cell_ids_inplace_fast(
    segmentation: np.ndarray,
    seed: int = None,
    preserve_background: bool = True,
    memory_efficient: bool = False
) -> np.ndarray:
    """
    Fast in-place randomization of cell IDs using vectorized operations.
    
    Parameters:
    -----------
    segmentation : np.ndarray
        Input segmentation array.
    seed : int, optional
        Random seed for reproducibility.
    preserve_background : bool, default=True
        If True, keeps background as 0.
    memory_efficient : bool, default=False
        If True, uses memory-efficient method (slower but uses less memory).
        If False, uses fast method (faster but uses more memory).
        
    Returns:
    --------
    np.ndarray
        Randomized segmentation array.
    """
    
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # Get unique cell IDs
    unique_ids = np.unique(segmentation)
    
    if preserve_background:
        unique_ids = unique_ids[unique_ids != 0]
    
    if len(unique_ids) == 0:
        return segmentation
    
    # Generate random permutation of IDs
    random_ids = np.random.permutation(unique_ids)
    
    # METHOD 1: Fastest for most cases (uses more memory)
    if not memory_efficient:
        # Create a lookup array that maps each original ID to its new random ID
        max_id = np.max(unique_ids)
        
        # Create mapping array (0 will map to 0 if preserve_background is True)
        mapping = np.zeros(max_id + 1, dtype=segmentation.dtype)
        
        # Fill the mapping array
        mapping[unique_ids] = random_ids
        
        # Apply mapping using vectorized indexing (FAST!)
        randomized_seg = mapping[segmentation]
    
    # METHOD 2: More memory efficient (better for very large ID ranges)
    else:
        # Create a copy of the segmentation
        randomized_seg = segmentation.copy()
        
        # Sort unique IDs and random IDs for efficient processing
        sorted_indices = np.argsort(unique_ids)
        sorted_orig_ids = unique_ids[sorted_indices]
        sorted_random_ids = random_ids[sorted_indices]
        
        # Use searchsorted to find positions efficiently
        # This creates a mask and assignment without explicit loops
        # Flatten the array for processing
        flat_seg = randomized_seg.ravel()
        
        # For each original ID, find positions and assign new ID
        # This is still a loop but much faster than the original
        # because it uses vectorized operations inside
        for orig_id, new_id in zip(sorted_orig_ids, sorted_random_ids):
            flat_seg[flat_seg == orig_id] = new_id
    
    return randomized_seg


def randomize_cell_ids_ultrafast(
    segmentation: np.ndarray,
    seed: int = None,
    preserve_background: bool = True
) -> np.ndarray:
    """
    Ultra-fast version using advanced numpy techniques.
    This is the fastest method but may be less intuitive.
    
    Parameters:
    -----------
    segmentation : np.ndarray
        Input segmentation array.
    seed : int, optional
        Random seed for reproducibility.
    preserve_background : bool, default=True
        If True, keeps background as 0.
        
    Returns:
    --------
    np.ndarray
        Randomized segmentation array.
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Get unique IDs
    unique_ids = np.unique(segmentation)
    
    if preserve_background:
        unique_ids = unique_ids[unique_ids != 0]
    
    if len(unique_ids) == 0:
        return segmentation
    
    # Generate random permutation
    random_ids = np.random.permutation(unique_ids)
    
    # Create a dictionary mapping (faster for sparse ID spaces)
    id_mapping = dict(zip(unique_ids, random_ids))
    
    # Flatten the array
    flat_seg = segmentation.ravel()
    
    # Use numpy's vectorize function (compiled C code, very fast)
    # Create a vectorized mapping function
    vectorized_map = np.vectorize(lambda x: id_mapping.get(x, x))
    
    # Apply the mapping
    randomized_flat = vectorized_map(flat_seg)
    
    # Reshape back to original shape
    randomized_seg = randomized_flat.reshape(segmentation.shape)
    
    return randomized_seg


def randomize_cell_ids_batch_optimized(
    segmentation: np.ndarray,
    seed: int = None,
    preserve_background: bool = True,
    batch_size: int = 100
) -> np.ndarray:
    """
    Optimized for very large images with many cells.
    Processes cells in batches to balance memory and speed.
    
    Parameters:
    -----------
    segmentation : np.ndarray
        Input segmentation array.
    seed : int, optional
        Random seed for reproducibility.
    preserve_background : bool, default=True
        If True, keeps background as 0.
    batch_size : int, default=100
        Number of cell IDs to process in each batch.
        
    Returns:
    --------
    np.ndarray
        Randomized segmentation array.
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Get unique IDs
    unique_ids = np.unique(segmentation)
    
    if preserve_background:
        unique_ids = unique_ids[unique_ids != 0]
    
    if len(unique_ids) == 0:
        return segmentation
    
    # Generate random permutation
    random_ids = np.random.permutation(unique_ids)
    
    # Create a copy for the result
    randomized_seg = segmentation.copy()
    
    # Process in batches
    n_batches = (len(unique_ids) + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(unique_ids))
        
        batch_orig_ids = unique_ids[start_idx:end_idx]
        batch_random_ids = random_ids[start_idx:end_idx]
        
        # Create a mask for all IDs in this batch
        # This is efficient because we create one mask per batch, not per ID
        mask = np.isin(segmentation, batch_orig_ids)
        
        if np.any(mask):
            # Create a temporary array for this batch
            temp_seg = segmentation[mask]
            
            # Create mapping for this batch
            max_id = np.max(batch_orig_ids)
            mapping = np.zeros(max_id + 1, dtype=segmentation.dtype)
            mapping[batch_orig_ids] = batch_random_ids
            
            # Apply mapping to this batch
            randomized_seg[mask] = mapping[temp_seg]
    
    return randomized_seg


# Benchmark function to compare performance
def benchmark_randomization(
    segmentation: np.ndarray,
    methods: list = None,
    n_runs: int = 3
) -> dict:
    """
    Benchmark different randomization methods.
    
    Parameters:
    -----------
    segmentation : np.ndarray
        Input segmentation array.
    methods : list, optional
        List of method names to benchmark.
    n_runs : int, default=3
        Number of runs for each method.
        
    Returns:
    --------
    dict
        Dictionary with timing results for each method.
    """
    import time
    
    if methods is None:
        methods = ['original', 'fast', 'ultrafast', 'batch_optimized']
    
    results = {}
    
    for method_name in methods:
        print(f"Benchmarking {method_name}...")
        times = []
        
        for run in range(n_runs):
            start_time = time.time()
            
            if method_name == 'original':
                # Original slow method
                unique_ids = np.unique(segmentation)
                unique_ids = unique_ids[unique_ids != 0]
                random_ids = np.random.permutation(unique_ids)
                id_mapping = dict(zip(unique_ids, random_ids))
                randomized_seg = np.zeros_like(segmentation)
                for orig_id, new_id in id_mapping.items():
                    randomized_seg[segmentation == orig_id] = new_id
            
            elif method_name == 'fast':
                randomize_cell_ids_inplace_fast(segmentation, memory_efficient=False)
            
            elif method_name == 'memory_efficient':
                randomize_cell_ids_inplace_fast(segmentation, memory_efficient=True)
            
            elif method_name == 'ultrafast':
                randomize_cell_ids_ultrafast(segmentation)
            
            elif method_name == 'batch_optimized':
                randomize_cell_ids_batch_optimized(segmentation, batch_size=50)
            
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        avg_time = np.mean(times[1:])  # Skip first run (warm-up)
        results[method_name] = avg_time
        print(f"  Average time: {avg_time:.4f} seconds")
    
    return results


