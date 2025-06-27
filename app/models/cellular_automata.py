"""
Cellular Automata Fire Spread Simulation for Forest Fire Dynamics
Grid-based probabilistic model for simulating fire spread
"""

import os
import logging
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from app.utils.config import Config
from app.utils.logging import performance_logger
from app.utils.database import SimulationResult

logger = logging.getLogger(__name__)

class CellularAutomataFireSpread:
    """
    Cellular automata model for simulating fire spread
    Grid-based, wind-driven, probabilistic fire propagation
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or Config.MODEL_CONFIG['cellular_automata']
        self.grid_size = self.config['grid_size']
        self.cell_size = self.config['cell_size']
        self.time_step = self.config['time_step']
        self.max_iterations = self.config['max_iterations']
        logger.info(f"Initialized Cellular Automata with config: {self.config}")

    def simulate_fire_spread(self, ignition_point: Tuple[int, int],
                            wind_speed: float = 0.0, wind_dir: float = 0.0,
                            fuel_map: Optional[np.ndarray] = None,
                            duration_hours: int = 6) -> Dict[str, Any]:
        grid = np.zeros(self.grid_size, dtype=np.uint8)
        grid[ignition_point] = 1  # 1 = burning
        burned = np.zeros_like(grid)
        steps = int(duration_hours * 3600 // self.time_step)
        wind_vector = np.array([np.cos(np.deg2rad(wind_dir)), np.sin(np.deg2rad(wind_dir))])
        for t in range(min(steps, self.max_iterations)):
            new_grid = grid.copy()
            for i in range(1, grid.shape[0] - 1):
                for j in range(1, grid.shape[1] - 1):
                    if grid[i, j] == 1:
                        burned[i, j] = 1
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                ni, nj = i + di, j + dj
                                if (di != 0 or dj != 0) and grid[ni, nj] == 0:
                                    prob = self._spread_probability(wind_speed, wind_vector, di, dj, fuel_map, ni, nj)
                                    if np.random.rand() < prob:
                                        new_grid[ni, nj] = 1
                        new_grid[i, j] = 2  # 2 = burned out
            grid = new_grid
        affected_area = np.sum(burned) * (self.cell_size ** 2) / 1e6  # km^2
        logger.info(f"Fire spread simulation complete. Affected area: {affected_area:.2f} km^2")
        return {
            'final_grid': grid,
            'burned_area_km2': affected_area,
            'burned_mask': burned
        }

    def _spread_probability(self, wind_speed, wind_vector, di, dj, fuel_map, ni, nj):
        base_prob = 0.2
        wind_effect = 1.0
        if wind_speed > 0:
            cell_vector = np.array([di, dj])
            wind_alignment = np.dot(wind_vector, cell_vector) / (np.linalg.norm(wind_vector) * np.linalg.norm(cell_vector) + 1e-6)
            wind_effect = 1 + 0.5 * wind_speed * max(0, wind_alignment)
        fuel_effect = 1.0
        if fuel_map is not None:
            fuel_effect = 0.5 + 0.5 * fuel_map[ni, nj]
        return min(1.0, base_prob * wind_effect * fuel_effect)

    def animate_fire_spread(self, simulation_result: Dict[str, Any], output_path: str = None):
        burned = simulation_result['burned_mask']
        fig, ax = plt.subplots()
        ims = []
        for t in range(burned.shape[0]):
            im = ax.imshow(burned, animated=True, cmap='hot', vmin=0, vmax=1)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
        if output_path:
            ani.save(output_path, writer='imagemagick')
            logger.info(f"Fire spread animation saved to {output_path}")
        plt.close(fig)
        return ani

    def export_result(self, simulation_result: Dict[str, Any], output_path: str):
        np.savez_compressed(output_path, **simulation_result)
        logger.info(f"Simulation result exported to {output_path}") 