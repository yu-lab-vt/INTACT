import numpy as np
import pandas as pd
import copy
import os
from typing import Dict, List, Tuple, Any, Set, Optional
from scipy import stats
import time
import gurobipy as gp
from gurobipy import GRB
import pulp
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'C_package'))
import ctypes

class CandidateFusionGraph:
    """Class representing the candidate fusion graph."""
    
    def __init__(
        self,
        movieInfo1: Dict[str, Any],
        movieInfo2: Dict[str, Any],
        matches: List[Tuple[int, int, float]],
        resolution: List[float] = [1.0, 1.0, 1.0],
    ):
        """
        Initialize candidate fusion graph.
        
        Parameters:
        -----------
        movieInfo1, movieInfo2 : Dict[str, Any]
            Input tracklets
        matches : List[Tuple[int, int, float]]
            Matched node pairs with confidence scores
        resolution : List[float]
            Resolution scaling factors [z, y, x]
        edge_confidence_function : str
            Type of edge confidence score function
        """
        self.movieInfo1 = movieInfo1
        self.movieInfo2 = movieInfo2
        self.matches = matches
        self.resolution = resolution
        
        # Build graph structure
        self._build_graph()
        
    def _build_graph(self):
        """Build the candidate fusion graph with nodes and edges."""
        # Total number of nodes from both tracklets
        n1 = len(self.movieInfo1['xCoord'])
        n2 = len(self.movieInfo2['xCoord'])
        self.n_nodes = n1 + n2
        self.n1 = n1  # Offset for movieInfo2 nodes
        
        # Node indices: 0 to n1-1 are from movieInfo1, n1 to n1+n2-1 are from movieInfo2
        self.nodes = list(range(self.n_nodes))
        
        # Store node attributes
        self.node_frames = np.concatenate([
            self.movieInfo1['frames'],
            self.movieInfo2['frames']
        ])
        is_2d = len(self.resolution) == 2
        if is_2d:
            self.node_coords = np.vstack([
                np.column_stack([
                    self.movieInfo1['yCoord'] * self.resolution[0],
                    self.movieInfo1['xCoord'] * self.resolution[1]
                ]),
                np.column_stack([
                    self.movieInfo2['yCoord'] * self.resolution[0],
                    self.movieInfo2['xCoord'] * self.resolution[1]
                ])
            ])
        else:
            self.node_coords = np.vstack([
                np.column_stack([
                    self.movieInfo1['zCoord'] * self.resolution[0],
                    self.movieInfo1['yCoord'] * self.resolution[1],
                    self.movieInfo1['xCoord'] * self.resolution[2]
                ]),
                np.column_stack([
                    self.movieInfo2['zCoord'] * self.resolution[0],
                    self.movieInfo2['yCoord'] * self.resolution[1],
                    self.movieInfo2['xCoord'] * self.resolution[2]
                ])
            ])
        
        # Initialize edge lists
        self.edges = []  # List of (u, v) tuples
        self.edge_costs = []  # List of edge costs C_uv = -log(P(x_u | x_v))
        self.edge_types = []  # 'original1', 'original2', or 'cross'
        
        # Add original edges from movieInfo1
        self._add_original_edges(self.movieInfo1, 0, 'original1')
        
        # Add original edges from movieInfo2 (with offset)
        self._add_original_edges(self.movieInfo2, n1, 'original2')
        
        # Build children information from parents
        self.children1 = self._build_children_from_parents(self.movieInfo1['parents'])
        self.children2 = self._build_children_from_parents(self.movieInfo2['parents'])

        # Add cross edges between matched nodes and their neighbors
        self._add_cross_edges(self.movieInfo1,self.movieInfo2)
        
        self.edges = np.array(self.edges, dtype=[('node1', int), ('node2', int), ('cost', float)])

        # Calculate node entry/exit costs based on chi-squared distribution
        threshold = stats.chi2.ppf(1 - 0.01 / self.n_nodes, 1) / 2
        self.dat_in = np.zeros((self.n_nodes, 3))
        self.dat_in[:, 0] = threshold 
        self.dat_in[:, 1] = threshold 
        self.dat_in[:, 2] = -2 * threshold - 0.00001

    def _build_children_from_parents(self, parents):
            """Build children list from parents information."""
            n = len(parents)
            children = [[] for _ in range(n)]
            
            for child_idx, parent_idx in enumerate(parents):
                if parent_idx >= 0:
                    children[parent_idx].append(child_idx)
            
            return children

    def _add_original_edges(self, movieInfo: Dict[str, Any], offset: int, edge_type: str):
        """Add original edges from a tracklet."""
        parents = np.asarray(movieInfo['parents'], dtype=int)
        n = len(movieInfo['xCoord'])
        valid = parents >= 0
        child_idx = np.nonzero(valid)[0]
        parent_idx = parents[valid]
        for c, p in zip(child_idx, parent_idx):
            u = p + offset
            v = c + offset
            if u < self.n_nodes and v < self.n_nodes:
                # Calculate edge cost based on distance
                cost = ovDistanceRegion(movieInfo['vox'][c],movieInfo['vox'][p])[0]
                if np.isfinite(cost):
                    self.edges.append((u, v, cost))
                    self.edge_types.append(edge_type)
    
    def _add_cross_edges(self, movieInfo1: Dict[str, Any], movieInfo2: Dict[str, Any]):
        """Add cross edges between matched nodes and their neighbors (parents and children)."""
        # Use a set to track added edges to avoid duplicates
        added_edges = set()
        
        for i1, i2 in self.matches:
            # Convert i2 to global index
            i2_global = i2 + self.n1
            
            # Get parents of i1 from tracklet1
            parent_i1 = None
            if i1 < len(self.movieInfo1['parents']):
                parent_i1 = self.movieInfo1['parents'][i1]
            
            # Get parents of i2 from tracklet2 (original index)
            parent_i2 = None
            if i2 < len(self.movieInfo2['parents']):
                parent_i2 = self.movieInfo2['parents'][i2]
            
            # Get children of i1 from tracklet1
            children_i1 = []
            if i1 < len(self.children1):
                children_i1 = self.children1[i1]
            
            # Get children of i2 from tracklet2
            children_i2 = []
            if i2 < len(self.children2):
                children_i2 = self.children2[i2]
            
            # Add cross edges: parent of i1 -> i2_global
            if parent_i1 is not None and parent_i1 >= 0 and parent_i1 < self.n1:
                edge_key = (parent_i1, i2_global)
                if edge_key not in added_edges:
                    cost = ovDistanceRegion(movieInfo1['vox'][parent_i1], movieInfo2['vox'][i2])[0]
                    if np.isfinite(cost):
                        self.edges.append((parent_i1, i2_global, cost))
                        self.edge_types.append('cross')
                        added_edges.add(edge_key)
            
            # Add cross edges: parent of i2 -> i1
            if parent_i2 is not None and parent_i2 >= 0 and parent_i2 < len(movieInfo2['xCoord']):
                parent_i2_global = parent_i2 + self.n1
                edge_key = (parent_i2_global, i1)
                if edge_key not in added_edges:
                    cost = ovDistanceRegion(movieInfo1['vox'][i1], movieInfo2['vox'][parent_i2])[0]
                    if np.isfinite(cost):
                        self.edges.append((parent_i2_global, i1, cost))
                        self.edge_types.append('cross')
                        added_edges.add(edge_key)
            
            # Add cross edges: i2_global -> children of i1
            for child_i1 in children_i1:
                edge_key = (i2_global, child_i1)
                if edge_key not in added_edges:
                    cost = ovDistanceRegion(movieInfo2['vox'][i2], movieInfo1['vox'][child_i1])[0]
                    if np.isfinite(cost):
                        self.edges.append((i2_global, child_i1, cost))
                        self.edge_types.append('cross')
                        added_edges.add(edge_key)
            
            # Add cross edges: i1 -> children of i2
            for child_i2 in children_i2:
                child_i2_global = child_i2 + self.n1
                edge_key = (i1, child_i2_global)
                if edge_key not in added_edges:
                    cost = ovDistanceRegion(movieInfo1['vox'][i1], movieInfo2['vox'][child_i2])[0]
                    if np.isfinite(cost):
                        self.edges.append((i1, child_i2_global, cost))
                        self.edge_types.append('cross')
                        added_edges.add(edge_key)
    
def solve_ilp_fusion(
    graph,
    use_heuristics: bool = True,
    presolve_level: int = 2
):
    """
    Solve the ILP formulation for tracklet fusion using Gurobi (much faster for large problems).
    
    Parameters:
    -----------
    graph : object
        Candidate fusion graph with nodes and edges
    use_heuristics : bool
        Whether to use Gurobi's heuristics
    presolve_level : int
        Presolve level (0=off, 1=conservative, 2=aggressive)
        
    Returns:
    --------
    solution : Dict[str, Any]
        Dictionary containing solution variables
    solver_info : Dict[str, Any]
        Dictionary containing solver statistics
    """

    # Create model
    model = gp.Model("TrackletFusion")
    
    # Set solver parameters
    model.setParam('OutputFlag', 1)  # Enable output to see progress
    model.setParam('Threads', min(16, os.cpu_count() or 1))  # Use multiple threads
    model.setParam('MIPGap', 0.01)  # 1% optimality gap
    model.setParam('Presolve', presolve_level)
    model.setParam('Heuristics', 0.05 if use_heuristics else 0.0)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions quickly
    
    # Get data
    node_costs = graph.dat_in
    edges = graph.edges
    
    n_nodes = graph.n_nodes
    n_edges = len(edges)
    
    print(f"Creating model with {n_nodes} nodes and {n_edges} edges...")
    
    # ============================================================================
    # Efficient variable creation using Gurobi's batch methods
    # ============================================================================
    
    # Node selection variables
    print("Creating node variables...")
    y_v = model.addVars(n_nodes, vtype=GRB.BINARY, name="y")
    
    # Entry/exit variables
    y_en = model.addVars(n_nodes, vtype=GRB.BINARY, name="y_en")
    y_ex = model.addVars(n_nodes, vtype=GRB.BINARY, name="y_ex")
    
    # Edge selection variables - create in batches for efficiency
    print("Creating edge variables...")
    edge_vars = {}
    
    # Pre-process edges to create variables efficiently
    unique_edges = {}
    for i, (u, v, _) in enumerate(edges):
        unique_edges[(u, v)] = i
    
    # Create edge variables in one batch
    edge_var_list = model.addVars(len(unique_edges), vtype=GRB.BINARY, name="edge")
    
    # Create mapping from (u,v) to variable index
    edge_index_mapping = {}
    for idx, (u, v) in enumerate(unique_edges.keys()):
        edge_index_mapping[(u, v)] = idx
        edge_vars[(u, v)] = edge_var_list[idx]
    
    print("Variables created. Setting objective...")
    
    # ============================================================================
    # Objective function
    # ============================================================================
    
    # Build objective efficiently
    obj_expr = gp.LinExpr()
    
    # Add node costs in batches
    for v in range(n_nodes):
        Cv1v2 = float(node_costs[v, 2])
        if np.isfinite(Cv1v2) and Cv1v2 != 0:
            obj_expr += Cv1v2 * y_v[v]
        elif not np.isfinite(Cv1v2):
            print(f"  WARNING: Node {v} Cv1v2 is {Cv1v2}, skipping...")
    # Add entry/exit costs
    for v in range(n_nodes):
        Cuen = float(node_costs[v, 0])
        Cuex = float(node_costs[v, 1])
        obj_expr += Cuen * y_en[v] + Cuex * y_ex[v]
    
    # Add edge costs
    for (u, v), var in edge_vars.items():
        # Find the cost for this edge
        edge_idx = unique_edges[(u, v)]
        Cuv = float(edges[edge_idx][2])
        if np.isfinite(Cuv) and Cuv != 0:
            obj_expr += Cuv * var
        elif not np.isfinite(Cuv):
            print(f"  WARNING: Node {v} Cuv is {Cuv}, skipping...")
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    print("Objective set. Adding constraints...")
    
    # ============================================================================
    # Constraints - optimized for speed
    # ============================================================================
    
    # Constraint 1: Mutually exclusive matches
    if hasattr(graph, 'matches') and len(graph.matches) > 0:
        print(f"Adding {len(graph.matches)} match constraints...")
        for u,v in graph.matches:
            v = v + graph.n1
            model.addConstr(y_v[u] + y_v[v] <= 1, name=f"match_{u}_{v}")
    
    # Precompute adjacency lists for faster constraint generation
    print("Precomputing adjacency lists...")
    incoming_edges = [[] for _ in range(n_nodes)]
    outgoing_edges = [[] for _ in range(n_nodes)]
    
    for (u, v) in edge_vars.keys():
        outgoing_edges[u].append((u, v))
        incoming_edges[v].append((u, v))
    
    # Constraint 2: Inflow constraint - sum_{u} y_{vu} + y_v^{en} = y_v
    print("Adding inflow constraints...")
    for v in range(n_nodes):
        inflow = gp.LinExpr()
        for (u, v2) in incoming_edges[v]:
            inflow += edge_vars[(u, v2)]

        inflow += y_en[v]
        model.addConstr(inflow == y_v[v], name=f"inflow_{v}")

    # Constraint 3: Outflow constraint - sum_{u} y_{uv} + y_v^{ex} = y_v
    print("Adding outflow constraints...")
    for v in range(n_nodes):
        outflow = gp.LinExpr()

        for (v2, u) in outgoing_edges[v]:
            outflow += edge_vars[(v2, u)]

        outflow += y_ex[v]
        model.addConstr(outflow == y_v[v], name=f"outflow_{v}")
    
    print(f"Model built with {model.numVars} variables and {model.numConstrs} constraints")
    print("Starting optimization...")
    
    # ============================================================================
    # Solve
    # ============================================================================
    model.optimize()
    
    solution = {
        'node_selection': np.zeros(n_nodes, dtype=int),
        'edge_selection': np.zeros(n_edges, dtype=int),
        'entry_selection': np.zeros(n_nodes, dtype=int),
        'exit_selection': np.zeros(n_nodes, dtype=int),
        'objective_value': model.ObjVal if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL else np.nan,
        'status': model.status
    }
    
    # Node selection
    if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
        for v in range(n_nodes):
            solution['node_selection'][v] = int(round(y_v[v].X))
    
        # Edge selection
        for i, (u, v, _) in enumerate(edges):
            if (u, v) in edge_vars:
                solution['edge_selection'][i] = int(round(edge_vars[(u, v)].X))
    
        # Entry/exit selection
        for v in range(n_nodes):
            solution['entry_selection'][v] = int(round(y_en[v].X))
            solution['exit_selection'][v] = int(round(y_ex[v].X))
    
    # Solver info
    solver_info = {
        'status_str': model.status,
        'runtime': model.Runtime,
        'node_count': model.NodeCount,
        'obj_bound': model.ObjBound,
        'mip_gap': model.MIPGap,
        'num_vars': model.numVars,
        'num_constrs': model.numConstrs
    }

        # Get selected nodes and edges
    selected_nodes = np.where(solution['node_selection'] == 1)[0]
    selected_edges = np.array([e.tolist() for i, e in enumerate(graph.edges) if solution['edge_selection'][i] == 1])

    return solution, solver_info, selected_nodes, selected_edges

def solve_ilp_fusion_pulp(
    graph,
    use_heuristics: bool = True,
    presolve_level: int = 2
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Solve the ILP formulation for tracklet fusion using PuLP with CBC solver (open-source alternative).
    
    Parameters:
    -----------
    graph : object
        Candidate fusion graph with nodes and edges
    use_heuristics : bool
        Whether to use solver heuristics (CBC supports this)
    presolve_level : int
        Presolve level (0=off, 1=conservative, 2=aggressive)
        
    Returns:
    --------
    solution : Dict[str, Any]
        Dictionary containing solution variables
    solver_info : Dict[str, Any]
        Dictionary containing solver statistics
    selected_nodes : np.ndarray
        Array of selected node indices
    selected_edges : np.ndarray
        Array of selected edges
    """

    # Create model
    model = pulp.LpProblem("TrackletFusion", pulp.LpMinimize)
    
    # Get data
    node_costs = graph.dat_in
    edges = graph.edges
    
    n_nodes = graph.n_nodes
    n_edges = len(edges)
    
    print(f"Creating model with {n_nodes} nodes and {n_edges} edges...")
    
    # ============================================================================
    # Efficient variable creation
    # ============================================================================
    
    # Node selection variables
    print("Creating node variables...")
    y_v = pulp.LpVariable.dicts("y", range(n_nodes), lowBound=0, upBound=1, cat='Binary')
    
    # Entry/exit variables
    y_en = pulp.LpVariable.dicts("y_en", range(n_nodes), lowBound=0, upBound=1, cat='Binary')
    y_ex = pulp.LpVariable.dicts("y_ex", range(n_nodes), lowBound=0, upBound=1, cat='Binary')
    
    # Edge selection variables
    print("Creating edge variables...")
    edge_vars = {}
    
    # Create unique edges dictionary
    unique_edges = {}
    for i, (u, v, _) in enumerate(edges):
        unique_edges[(u, v)] = i
    
    # Create edge variables
    for (u, v), idx in unique_edges.items():
        edge_vars[(u, v)] = pulp.LpVariable(f"edge_{u}_{v}", lowBound=0, upBound=1, cat='Binary')
    
    print("Variables created. Setting objective...")
    
    # ============================================================================
    # Objective function
    # ============================================================================
    
    # Build objective
    objective = pulp.LpAffineExpression()
    
    # Add node costs
    for v in range(n_nodes):
        Cv1v2 = float(node_costs[v, 2])
        if np.isfinite(Cv1v2) and Cv1v2 != 0:
            objective += Cv1v2 * y_v[v]
        elif not np.isfinite(Cv1v2):
            print(f"  WARNING: Node {v} Cv1v2 is {Cv1v2}, skipping...")
    
    # Add entry/exit costs
    for v in range(n_nodes):
        Cuen = float(node_costs[v, 0])
        Cuex = float(node_costs[v, 1])
        if np.isfinite(Cuen) and Cuen != 0:
            objective += Cuen * y_en[v]
        if np.isfinite(Cuex) and Cuex != 0:
            objective += Cuex * y_ex[v]
    
    # Add edge costs
    for (u, v), var in edge_vars.items():
        edge_idx = unique_edges[(u, v)]
        Cuv = float(edges[edge_idx][2])
        if np.isfinite(Cuv) and Cuv != 0:
            objective += Cuv * var
        elif not np.isfinite(Cuv):
            print(f"  WARNING: Edge ({u},{v}) Cuv is {Cuv}, skipping...")
    
    model += objective, "Objective"
    
    print("Objective set. Adding constraints...")
    
    # ============================================================================
    # Constraints
    # ============================================================================
    
    # Constraint 1: Mutually exclusive matches
    if hasattr(graph, 'matches') and len(graph.matches) > 0:
        print(f"Adding {len(graph.matches)} match constraints...")
        for u, v in graph.matches:
            v = v + graph.n1
            model += y_v[u] + y_v[v] <= 1, f"match_{u}_{v}"
    
    # Precompute adjacency lists for faster constraint generation
    print("Precomputing adjacency lists...")
    incoming_edges = [[] for _ in range(n_nodes)]
    outgoing_edges = [[] for _ in range(n_nodes)]
    
    for (u, v) in edge_vars.keys():
        outgoing_edges[u].append((u, v))
        incoming_edges[v].append((u, v))
    
    # Constraint 2: Inflow constraint - sum_{u} y_{vu} + y_v^{en} = y_v
    print("Adding inflow constraints...")
    for v in range(n_nodes):
        inflow_expr = pulp.LpAffineExpression()
        for (u, v2) in incoming_edges[v]:
            inflow_expr += edge_vars[(u, v2)]
        inflow_expr += y_en[v]
        model += inflow_expr == y_v[v], f"inflow_{v}"
    
    # Constraint 3: Outflow constraint - sum_{u} y_{uv} + y_v^{ex} = y_v
    print("Adding outflow constraints...")
    for v in range(n_nodes):
        outflow_expr = pulp.LpAffineExpression()
        for (v2, u) in outgoing_edges[v]:
            outflow_expr += edge_vars[(v2, u)]
        outflow_expr += y_ex[v]
        model += outflow_expr == y_v[v], f"outflow_{v}"
    
    print(f"Model built with {len(model.variables())} variables and {len(model.constraints)} constraints")
    print("Starting optimization...")
    
    # ============================================================================
    # Solve with CBC (open-source solver)
    # ============================================================================
    
    # Configure solver options
    solver_options = []
    
    # Set threads (CBC uses --threads)
    max_threads = min(16, os.cpu_count() or 1)
    solver_options.append(f"threads={max_threads}")
    
    # Set gap tolerance (CBC uses --ratio)
    solver_options.append("ratio=0.01")  # 1% gap
    
    # Set time limit if needed (optional)
    # solver_options.append("sec=3600")  # 1 hour time limit
    
    # Set presolve level
    if presolve_level == 0:
        solver_options.append("presolve=off")
    elif presolve_level == 1:
        solver_options.append("presolve=on")
    elif presolve_level == 2:
        solver_options.append("presolve=on")
        solver_options.append("strong=5")  # More aggressive presolve
    
    # Set heuristics
    if use_heuristics:
        solver_options.append("heuristics=on")
        solver_options.append("heur=0.05")  # Heuristic frequency
    else:
        solver_options.append("heuristics=off")
    
    # Solve with CBC
    try:
        solver = pulp.PULP_CBC_CMD(
            msg=1,  # Enable solver output
            timeLimit=None,  # No time limit by default
            gapRel=0.01,  # 1% relative gap
            threads=max_threads,
            options=solver_options
        )
        model.solve(solver)
    except Exception as e:
        print(f"Warning: CBC solver failed with error: {e}")
        print("Falling back to default solver...")
        model.solve()
    
    # ============================================================================
    # Collect solution
    # ============================================================================
    
    solution = {
        'node_selection': np.zeros(n_nodes, dtype=int),
        'edge_selection': np.zeros(n_edges, dtype=int),
        'entry_selection': np.zeros(n_nodes, dtype=int),
        'exit_selection': np.zeros(n_nodes, dtype=int),
        'objective_value': pulp.value(model.objective) if model.status == pulp.LpStatusOptimal else np.nan,
        'status': model.status
    }
    
    # Status mapping from PuLP to Gurobi-like status
    status_map = {
        pulp.LpStatusOptimal: "OPTIMAL",
        pulp.LpStatusInfeasible: "INFEASIBLE",
        pulp.LpStatusUnbounded: "UNBOUNDED",
        pulp.LpStatusUndefined: "UNDEFINED"
    }
    
    # Get solver information
    solver_info = {
        'status_str': status_map.get(model.status, str(model.status)),
        'runtime': None,  # CBC doesn't easily expose runtime through PuLP
        'node_count': None,  # Not directly available in PuLP
        'obj_bound': None,  # Not directly available in PuLP
        'mip_gap': None,  # Can be computed if needed
        'num_vars': len(model.variables()),
        'num_constrs': len(model.constraints)
    }
    
    # Extract solution if optimal
    if model.status == pulp.LpStatusOptimal:
        # Node selection
        for v in range(n_nodes):
            solution['node_selection'][v] = int(round(pulp.value(y_v[v])))
        
        # Edge selection
        for i, (u, v, _) in enumerate(edges):
            if (u, v) in edge_vars:
                solution['edge_selection'][i] = int(round(pulp.value(edge_vars[(u, v)])))
        
        # Entry/exit selection
        for v in range(n_nodes):
            solution['entry_selection'][v] = int(round(pulp.value(y_en[v])))
            solution['exit_selection'][v] = int(round(pulp.value(y_ex[v])))
    
    # Get selected nodes and edges
    selected_nodes = np.where(solution['node_selection'] == 1)[0]
    selected_edges = np.array([e.tolist() for i, e in enumerate(graph.edges) 
                               if solution['edge_selection'][i] == 1])
    
    return solution, solver_info, selected_nodes, selected_edges


# Alternative implementation using OR-Tools (often faster than CBC)
def solve_ilp_fusion_ortools(
    graph,
    use_heuristics: bool = True,
    presolve_level: int = 2,
    time_limit_seconds: int = 3600
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Solve the ILP formulation for tracklet fusion using Google OR-Tools (open-source, often faster than CBC).
    Requires: pip install ortools
    
    Parameters:
    -----------
    graph : object
        Candidate fusion graph with nodes and edges
    use_heuristics : bool
        Whether to use solver heuristics
    presolve_level : int
        Presolve level (0=off, 1=conservative, 2=aggressive)
    time_limit_seconds : int
        Maximum solving time in seconds
        
    Returns:
    --------
    solution : Dict[str, Any]
        Dictionary containing solution variables
    solver_info : Dict[str, Any]
        Dictionary containing solver statistics
    selected_nodes : np.ndarray
        Array of selected node indices
    selected_edges : np.ndarray
        Array of selected edges
    """
    try:
        from ortools.linear_solver import pywraplp
    except ImportError:
        raise ImportError("OR-Tools is not installed. Please install with: pip install ortools")
    
    # Create solver
    solver = pywraplp.Solver.CreateSolver('SCIP')  # SCIP is open-source and fast
    if not solver:
        solver = pywraplp.Solver.CreateSolver('CBC')  # Fallback to CBC
    
    solver.SetNumThreads(min(16, os.cpu_count() or 1))
    
    # Set time limit
    if time_limit_seconds:
        solver.SetTimeLimit(int(time_limit_seconds * 1000))  # Convert to milliseconds
    
    # Get data
    node_costs = graph.dat_in
    edges = graph.edges
    n_nodes = graph.n_nodes
    n_edges = len(edges)
    
    # Create variables
    y_v = [solver.BoolVar(f'y_{i}') for i in range(n_nodes)]
    y_en = [solver.BoolVar(f'y_en_{i}') for i in range(n_nodes)]
    y_ex = [solver.BoolVar(f'y_ex_{i}') for i in range(n_nodes)]
    
    # Create edge variables
    edge_vars = {}
    unique_edges = {}
    for i, (u, v, _) in enumerate(edges):
        unique_edges[(u, v)] = i
        edge_vars[(u, v)] = solver.BoolVar(f'edge_{u}_{v}')
    
    # Objective function
    objective = solver.Objective()
    
    for v in range(n_nodes):
        Cv1v2 = float(node_costs[v, 2])
        if np.isfinite(Cv1v2) and Cv1v2 != 0:
            objective.SetCoefficient(y_v[v], Cv1v2)
    
    for v in range(n_nodes):
        Cuen = float(node_costs[v, 0])
        Cuex = float(node_costs[v, 1])
        if np.isfinite(Cuen) and Cuen != 0:
            objective.SetCoefficient(y_en[v], Cuen)
        if np.isfinite(Cuex) and Cuex != 0:
            objective.SetCoefficient(y_ex[v], Cuex)
    
    for (u, v), var in edge_vars.items():
        edge_idx = unique_edges[(u, v)]
        Cuv = float(edges[edge_idx][2])
        if np.isfinite(Cuv) and Cuv != 0:
            objective.SetCoefficient(var, Cuv)
    
    objective.SetMinimization()
    
    # Constraints
    # Mutually exclusive matches
    if hasattr(graph, 'matches') and len(graph.matches) > 0:
        for u, v in graph.matches:
            v = v + graph.n1
            constraint = solver.Constraint(-solver.infinity(), 1)
            constraint.SetCoefficient(y_v[u], 1)
            constraint.SetCoefficient(y_v[v], 1)
    
    # Precompute adjacency lists
    incoming_edges = [[] for _ in range(n_nodes)]
    outgoing_edges = [[] for _ in range(n_nodes)]
    
    for (u, v) in edge_vars.keys():
        outgoing_edges[u].append((u, v))
        incoming_edges[v].append((u, v))
    
    # Inflow constraints
    for v in range(n_nodes):
        constraint = solver.Constraint(0, 0)
        constraint.SetCoefficient(y_v[v], -1)
        constraint.SetCoefficient(y_en[v], 1)
        for (u, v2) in incoming_edges[v]:
            constraint.SetCoefficient(edge_vars[(u, v2)], 1)
    
    # Outflow constraints
    for v in range(n_nodes):
        constraint = solver.Constraint(0, 0)
        constraint.SetCoefficient(y_v[v], -1)
        constraint.SetCoefficient(y_ex[v], 1)
        for (v2, u) in outgoing_edges[v]:
            constraint.SetCoefficient(edge_vars[(v2, u)], 1)
    
    # Solve
    print("Starting OR-Tools optimization...")
    result_status = solver.Solve()
    
    # Collect results
    solution = {
        'node_selection': np.zeros(n_nodes, dtype=int),
        'edge_selection': np.zeros(n_edges, dtype=int),
        'entry_selection': np.zeros(n_nodes, dtype=int),
        'exit_selection': np.zeros(n_nodes, dtype=int),
        'objective_value': objective.Value() if result_status == pywraplp.Solver.OPTIMAL else np.nan,
        'status': result_status
    }
    
    if result_status == pywraplp.Solver.OPTIMAL:
        for v in range(n_nodes):
            solution['node_selection'][v] = int(round(y_v[v].solution_value()))
        
        for i, (u, v, _) in enumerate(edges):
            if (u, v) in edge_vars:
                solution['edge_selection'][i] = int(round(edge_vars[(u, v)].solution_value()))
        
        for v in range(n_nodes):
            solution['entry_selection'][v] = int(round(y_en[v].solution_value()))
            solution['exit_selection'][v] = int(round(y_ex[v].solution_value()))
    
    # Solver info
    status_map = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
    }
    
    solver_info = {
        'status_str': status_map.get(result_status, str(result_status)),
        'runtime': solver.wall_time() / 1000.0,  # Convert to seconds
        'node_count': solver.nodes(),
        'obj_bound': solver.Objective().BestBound(),
        'mip_gap': solver.MipGap(),
        'num_vars': solver.NumVariables(),
        'num_constrs': solver.NumConstraints()
    }
    
    # Get selected nodes and edges
    selected_nodes = np.where(solution['node_selection'] == 1)[0]
    selected_edges = np.array([e.tolist() for i, e in enumerate(graph.edges) 
                               if solution['edge_selection'][i] == 1])
    
    return solution, solver_info, selected_nodes, selected_edges


def create_fused_movieInfo(nodes, edges, movieInfo1, movieInfo2, movieInfo1_partial, movieInfo2_partial, idmap1=None, idmap2=None, keep_parent_idx=None, crop_area=[], tm_shift = 0, img_shape=None, seg_indice = 'python'):
    """
    Create a fused movieInfo by combining information from movieInfo1 and movieInfo2 based on tracking results.
    
    Args:
        edges (array): Array of trajectories
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
    edges = edges.astype(int)

    # Initialize result structure
    if len(crop_area) > 0:
        if len(crop_area) == 2:
            movieInfo1_partial['xCoord'] = movieInfo1_partial['xCoord'] + crop_area[1][0]
            movieInfo1_partial['yCoord'] = movieInfo1_partial['yCoord'] + crop_area[0][0]
            movieInfo2_partial['xCoord'] = movieInfo2_partial['xCoord'] + crop_area[1][0]
            movieInfo2_partial['yCoord'] = movieInfo2_partial['yCoord'] + crop_area[0][0]
            movieInfo1_partial['vox'] = [vox + np.array([crop_area[0][0], crop_area[1][0]]) 
                for vox in movieInfo1_partial['vox']]
            movieInfo2_partial['vox'] = [vox + np.array([crop_area[0][0], crop_area[1][0]]) 
                for vox in movieInfo2_partial['vox']]
        if len(crop_area) == 3:
            movieInfo1_partial['xCoord'] = movieInfo1_partial['xCoord'] + crop_area[2][0]
            movieInfo1_partial['yCoord'] = movieInfo1_partial['yCoord'] + crop_area[1][0]
            movieInfo1_partial['zCoord'] = movieInfo1_partial['zCoord'] + crop_area[0][0]
            movieInfo2_partial['xCoord'] = movieInfo2_partial['xCoord'] + crop_area[2][0]
            movieInfo2_partial['yCoord'] = movieInfo2_partial['yCoord'] + crop_area[1][0]
            movieInfo2_partial['zCoord'] = movieInfo2_partial['zCoord'] + crop_area[0][0]
            movieInfo1_partial['vox'] = [vox + np.array([crop_area[0][0], crop_area[1][0], crop_area[2][0]]) 
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
        
        # Process all nodes, not just edges
        all_nodes_set = set(nodes.astype(int))
        # Original: for track in tracks:
        # Replace with edges-based parent-child mapping
        parent_child_map = {}  # key: child_id, value: parent_id
        children_map = {}

        for edge in edges:
            parent_id = edge[0]
            child_id = edge[1]
            parent_child_map[child_id] = parent_id
            children_map[parent_id] = child_id
        
        # Process all cells in edges (maintain track continuity via parent_child_map)
        processed_ids = set()
        # First process root cells (no parent in edges)
        root_ids = []
        for node_id in all_nodes_set:
            if node_id not in parent_child_map:  # No parent in edges
                root_ids.append(node_id)

        for root_id in root_ids:
            current_id = root_id
            prev_new_id = -1
            # Traverse the edge chain to reconstruct track
            while current_id in all_nodes_set and current_id not in processed_ids:
                if current_id < n1:
                    # Cell from movieInfo1_partial
                    cell_data = {key: movieInfo1_partial[key][current_id] for key in 
                                ['xCoord', 'yCoord', 'zCoord', 'frames', 'vox', 'voxIdx']}
                else:
                    # Cell from movieInfo2_partial (adjust index)
                    adjusted_id = current_id - n1
                    cell_data = {key: movieInfo2_partial[key][adjusted_id] for key in 
                                ['xCoord', 'yCoord', 'zCoord', 'frames', 'vox', 'voxIdx']}
                
                # Determine parent_id (from prev_new_id in track chain)
                parent_id = -1
                if current_id in parent_child_map:
                    # This cell has a parent in edges
                    parent_old_id = parent_child_map[current_id]
                    # Use the new ID of the parent if it's already processed
                    parent_id = old_to_new_id.get(parent_old_id, -1)

                new_id = add_cell(cell_data, parent_id)
                
                # Store mapping and update state
                old_to_new_id[current_id] = new_id
                processed_ids.add(current_id)
                prev_new_id = new_id
                new_id_counter += 1
                
                # Move to next child in the chain
                current_id = children_map.get(current_id, None)
            # Add nodes that are not connected by edges (isolated nodes)
        for node_id in all_nodes_set:
            if node_id not in processed_ids:
                if node_id < n1:
                    cell_data = {key: movieInfo1_partial[key][node_id] for key in 
                                ['xCoord', 'yCoord', 'zCoord', 'frames', 'vox', 'voxIdx']}
                else:
                    adjusted_id = node_id - n1
                    cell_data = {key: movieInfo2_partial[key][adjusted_id] for key in 
                                ['xCoord', 'yCoord', 'zCoord', 'frames', 'vox', 'voxIdx']}
                
                # Isolated nodes have no parent
                new_id = add_cell(cell_data, -1)
                old_to_new_id[node_id] = new_id
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

        # Original: for track in tracks:
        # Replace with edges-based parent-child mapping for full IDs
        parent_child_map = {}  # key: child_full_id, value: parent_full_id
        all_full_ids = set()
        n1_all = len(movieInfo1['frames'])

        for edge in edges:
            parent_cell_id = edge[0]
            child_cell_id = edge[1]
            
            # Convert partial cell IDs from edges to full IDs
            if parent_cell_id < n1:
                parent_full_id = rev_idmap1[parent_cell_id]
            else:
                parent_full_id = rev_idmap2[parent_cell_id - n1] + n1_all
            
            if child_cell_id < n1:
                child_full_id = rev_idmap1[child_cell_id]
            else:
                child_full_id = rev_idmap2[child_cell_id - n1] + n1_all
            
            parent_child_map[child_full_id] = parent_full_id
            all_full_ids.add(parent_full_id)
            all_full_ids.add(child_full_id)


        # Add cells in edges to cells_in_tracks (full ID + marker)
        # Process all nodes (not just edges)
        # Convert nodes (partial IDs) to full IDs
        all_nodes_full_ids = set()
        for node_id in nodes.astype(int):
            if node_id < n1:
                # From movieInfo1_partial
                full_id = rev_idmap1[node_id]
            else:
                # From movieInfo2_partial
                full_id = rev_idmap2[node_id - n1] + n1_all
            all_nodes_full_ids.add(full_id)

        # Update full_id_parent_map using edge-derived parent-child relationships
        full_id_parent_map = {}
        # For all nodes (including those not in edges), set parent to -1 by default
        for full_id in all_nodes_full_ids:
            full_id_parent_map[full_id] = -1

        # Then update with edges relationships
        for child_full_id, parent_full_id in parent_child_map.items():
            full_id_parent_map[child_full_id] = parent_full_id

        # Build parent-child relationships for all nodes in edges
        parent_child_map = {}  # key: child_full_id, value: parent_full_id
        for edge in edges:
            parent_cell_id = edge[0]
            child_cell_id = edge[1]
            
            # Convert partial cell IDs from edges to full IDs
            if parent_cell_id < n1:
                parent_full_id = rev_idmap1[parent_cell_id]
            else:
                parent_full_id = rev_idmap2[parent_cell_id - n1] + n1_all
            
            if child_cell_id < n1:
                child_full_id = rev_idmap1[child_cell_id]
            else:
                child_full_id = rev_idmap2[child_cell_id - n1] + n1_all
            
            parent_child_map[child_full_id] = parent_full_id

        # Track which full IDs are in nodes
        cells_in_nodes = set()
        for full_id in all_nodes_full_ids:
            if full_id < n1_all:
                cells_in_nodes.add((full_id, 'm1'))
            else:
                cells_in_nodes.add((full_id, 'm2'))
        
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
        # Add cells that appear in nodes (from partial movieInfos)
        for full_id, marker in cells_in_nodes:
            if marker == 'm1':
                cell_data = {key: movieInfo1[key][full_id] for key in 
                            ['xCoord', 'yCoord', 'zCoord', 'frames', 'vox', 'voxIdx']}
                parent_id = full_id_parent_map.get(full_id, -1) 
                new_id = add_cell(cell_data, parent_id)
                old_to_new_id[full_id] = new_id
            else:
                cell_data = {key: movieInfo2[key][full_id-n1_all] for key in 
                            ['xCoord', 'yCoord', 'zCoord', 'frames', 'vox', 'voxIdx']}
                parent_id = full_id_parent_map.get(full_id, -1) 
                new_id = add_cell(cell_data, parent_id)
                old_to_new_id[full_id] = new_id

            new_id_counter += 1
        for idx in keep_parent_idx:
            if idx in old_to_new_id:
                if idx < n1_all:
                    movieInfo_new['parents'][old_to_new_id[idx]] = movieInfo1['parents'][idx]
                else:
                    movieInfo_new['parents'][old_to_new_id[idx]] = -1 if movieInfo2['parents'][idx - n1_all] == -1 else movieInfo2['parents'][idx - n1_all] + n1_all
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