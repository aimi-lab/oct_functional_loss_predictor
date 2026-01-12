from __future__ import annotations 
from typing import Dict, Tuple 

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import matplotlib.pyplot as plt

from eyemod.definitions import Laterality

@dataclass
class VisualFieldData:
    """Container for visual field test data and computed properties.
    
    This class holds visual field measurement data including coordinates, sensitivity values,
    and optional normative values. It provides computed properties for common operations
    like deviation calculations and coordinate access.
    
    The deviation calculation follows the Octopus implementation:
        deviation = normative - sensitivity 

    Attributes:
        x_coordinates: Array of x-coordinates for test locations.
        y_coordinates: Array of y-coordinates for test locations.
        sensitivity_values: Array of measured sensitivity values at each location.
        normative_values: Optional array of normative/expected values for comparison.
            If None, deviation_values property will return None.
        laterality: Eye laterality ("OD"/"right", "OS"/"left", "OU"/"both", or None).

    Raises:
        ValueError: If coordinate arrays and sensitivity values have different lengths,
            or if normative_values length doesn't match when provided.
    """
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    sensitivity_values: np.ndarray
    normative_values: np.ndarray = None
    laterality: Laterality = Laterality.U 
    
    def __post_init__(self):
        lengths = [len(self.x_coordinates), len(self.y_coordinates), len(self.sensitivity_values)]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError("Array length mismatch")
        
        if self.normative_values is not None and len(self.normative_values) != lengths[0]:
            raise ValueError("Normative values length mismatch")
        
        if isinstance(self.laterality, str):
            self.laterality = Laterality.from_str(self.laterality)
    
    @property
    def coordinates(self) -> np.ndarray:
        return np.column_stack([self.x_coordinates, self.y_coordinates])
    
    @property
    def deviation_values(self) -> np.ndarray:
        if self.normative_values is not None:
            # Currently, we only implement the Octopus deviation computation
            # Humphrey: deviation = sensitivity - normative
            return self.normative_values - self.sensitivity_values
        else:
            return None
    
    @property
    def n_locations(self) -> int:
        return len(self.x_coordinates)

# Minimal protocols for the few things that might vary
class Interpolator(Protocol):
    def interpolate(self, data: VisualFieldData, new_coords: np.ndarray) -> VisualFieldData: ...

class Renderer(Protocol):
    def render_voronoi(self, data: VisualFieldData, ax: plt.Axes, **kwargs) -> Tuple[plt.Figure, plt.Axes]: ...
    def render_scatter(self, data: VisualFieldData, ax: plt.Axes, **kwargs) -> Tuple[plt.Figure, plt.Axes]: ...
    def render_numbering(self, data: VisualFieldData, ax: plt.Axes, **kwargs) -> Tuple[plt.Figure, plt.Axes]: ...

@runtime_checkable
class Filter(Protocol):
    def evaluate(self, data: VisualFieldData) -> np.ndarray:
        """Return boolean mask for filtering locations"""
    ...