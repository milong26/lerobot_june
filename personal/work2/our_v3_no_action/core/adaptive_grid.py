"""
Adaptive Grid Data Structure

Implements the adaptive spatial grid that starts as a coarse uniform grid
and recursively splits high-value cells into finer sub-cells.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class AdaptiveCell:
    """A single cell in the adaptive grid."""
    cell_id: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    depth: int = 0
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    sample_episode_indices: List[int] = field(default_factory=list)
    sample_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    priority: float = 0.0
    active: bool = True

    @property
    def n_samples(self) -> int:
        return len(self.sample_episode_indices)


def cell_center(cell: AdaptiveCell) -> Tuple[float, float, float]:
    """Return the geometric center of the cell as a target init_pos (z=0)."""
    cx = (cell.x_min + cell.x_max) / 2.0
    cy = (cell.y_min + cell.y_max) / 2.0
    return (cx, cy, 0.0)


def contains(cell: AdaptiveCell, pos: Tuple[float, float, float]) -> bool:
    """Check if a position falls within the cell bounds (x,y only)."""
    return (cell.x_min <= pos[0] <= cell.x_max and
            cell.y_min <= pos[1] <= cell.y_max)


def compute_cell_area(cell: AdaptiveCell) -> float:
    """Compute the 2D area of the cell."""
    return (cell.x_max - cell.x_min) * (cell.y_max - cell.y_min)


def split_cell(cell: AdaptiveCell, split_x: int = 2, split_y: int = 2) -> List[AdaptiveCell]:
    """Split a cell into split_x * split_y child cells."""
    x_edges = np.linspace(cell.x_min, cell.x_max, split_x + 1)
    y_edges = np.linspace(cell.y_min, cell.y_max, split_y + 1)

    children = []
    for i in range(split_x):
        for j in range(split_y):
            child_id = f"{cell.cell_id}_{i}{j}"
            child = AdaptiveCell(
                cell_id=child_id,
                x_min=float(x_edges[i]),
                x_max=float(x_edges[i + 1]),
                y_min=float(y_edges[j]),
                y_max=float(y_edges[j + 1]),
                depth=cell.depth + 1,
                parent_id=cell.cell_id,
            )
            children.append(child)

    cell.children = [c.cell_id for c in children]
    cell.active = False
    return children


def build_initial_grid(
    workspace_bounds: Dict[str, Tuple[float, float]],
    grid_x: int,
    grid_y: int,
) -> List[AdaptiveCell]:
    """Build the initial coarse grid covering the workspace."""
    x_min, x_max = workspace_bounds["x"]
    y_min, y_max = workspace_bounds["y"]

    x_edges = np.linspace(x_min, x_max, grid_x + 1)
    y_edges = np.linspace(y_min, y_max, grid_y + 1)

    cells = []
    for i in range(grid_x):
        for j in range(grid_y):
            cell_id = f"c{i}_{j}"
            cell = AdaptiveCell(
                cell_id=cell_id,
                x_min=float(x_edges[i]),
                x_max=float(x_edges[i + 1]),
                y_min=float(y_edges[j]),
                y_max=float(y_edges[j + 1]),
                depth=0,
            )
            cells.append(cell)

    return cells


def get_leaf_cells(all_cells: Dict[str, AdaptiveCell]) -> List[AdaptiveCell]:
    """Return all active leaf cells (cells with no active children)."""
    leaves = []
    for cell in all_cells.values():
        if cell.active and not cell.children:
            leaves.append(cell)
    return leaves


def get_neighbor_cells(
    cell: AdaptiveCell,
    all_cells: Dict[str, AdaptiveCell],
) -> List[AdaptiveCell]:
    """Get spatially adjacent leaf cells (sharing an edge or corner)."""
    neighbors = []
    cx = (cell.x_min + cell.x_max) / 2.0
    cy = (cell.y_min + cell.y_max) / 2.0

    for other in all_cells.values():
        if other.cell_id == cell.cell_id or not other.active:
            continue
        ox = (other.x_min + other.x_max) / 2.0
        oy = (other.y_min + other.y_max) / 2.0

        # Check if cells are adjacent (share boundary or are within one cell width)
        dx = abs(cx - ox)
        dy = abs(cy - oy)
        max_dx = (cell.x_max - cell.x_min + other.x_max - other.x_min) / 2.0 + 1e-9
        max_dy = (cell.y_max - cell.y_min + other.y_max - other.y_min) / 2.0 + 1e-9

        if dx <= max_dx and dy <= max_dy:
            neighbors.append(other)

    return neighbors


def pick_next_target_in_cell(
    cell: AdaptiveCell,
    selected_positions: List[Tuple[float, float, float]],
    split_x: int = 2,
    split_y: int = 2,
) -> Tuple[float, float, float]:
    """
    Generate a new target init_pos inside the cell.
    If the cell has children, use child centers.
    Otherwise use maximin to pick a position far from already-sampled positions.
    """
    if cell.children:
        # Use child cell centers, pick the one farthest from existing samples
        child_centers = []
        for i in range(split_x):
            for j in range(split_y):
                cx = cell.x_min + (cell.x_max - cell.x_min) * (i + 0.5) / split_x
                cy = cell.y_min + (cell.y_max - cell.y_min) * (j + 0.5) / split_y
                child_centers.append((cx, cy, 0.0))

        if not selected_positions:
            return child_centers[0]

        # Maximin: pick child center farthest from all existing samples
        best_pos = None
        best_min_dist = -1.0
        for pos in child_centers:
            min_dist = min(
                np.sqrt((pos[0] - s[0])**2 + (pos[1] - s[1])**2)
                for s in selected_positions
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = pos

        return best_pos if best_pos else cell_center(cell)

    # No children: use maximin within cell bounds
    n_candidates = 20
    rng = np.random.RandomState(hash(cell.cell_id) % (2**31))

    candidates = []
    for _ in range(n_candidates):
        px = rng.uniform(cell.x_min, cell.x_max)
        py = rng.uniform(cell.y_min, cell.y_max)
        candidates.append((px, py, 0.0))

    if not selected_positions:
        return candidates[0]

    best_pos = None
    best_min_dist = -1.0
    for pos in candidates:
        min_dist = min(
            np.sqrt((pos[0] - s[0])**2 + (pos[1] - s[1])**2)
            for s in selected_positions
        )
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_pos = pos

    return best_pos if best_pos else cell_center(cell)