from __future__ import annotations 
from typing import Dict, Tuple 

import numpy as np
import matplotlib.pyplot as plt

from eyemod.visual_fields.core import VisualFieldData, Laterality, Interpolator, Renderer, Filter
from eyemod.visual_fields.interpolation import VisualFieldInterpolator
from eyemod.visual_fields.visualization import VisualFieldRenderer

class VisualField:
    """Simplified visual field with minimal dependencies"""
    
    def __init__(
        self, 
        data: VisualFieldData,
        interpolator: Interpolator = None,
        renderer: Renderer = None
    ):
        self.data = data
        self._interpolator = interpolator or VisualFieldInterpolator()
        self._renderer = renderer or VisualFieldRenderer()
    
    def mean_deviation(self) -> float:
        """Calculate mean deviation if normative values available"""
        if self.data.normative_values is None:
            raise ValueError("Normative values required for deviation calculation")
        return float(np.mean(self.data.deviation_values))
    
    def pattern_standard_deviation(self) -> float:
        """Calculate pattern standard deviation"""
        if self.data.normative_values is None:
            raise ValueError("Normative values required for PSD calculation")
        return float(np.std(self.data.deviation_values))
    
    def sensitivity_stats(self) -> Dict[str, float]:
        """Get basic sensitivity statistics"""
        sens = self.data.sensitivity_values
        return {
            'mean': float(np.mean(sens)),
            'std': float(np.std(sens)),
            'min': float(np.min(sens)),
            'max': float(np.max(sens)),
            'median': float(np.median(sens))
        }
    
    def interpolate_to(self, new_coordinates: np.ndarray) -> 'VisualField':
        """Create new visual field with interpolated values"""
        new_data = self._interpolator.interpolate(self.data, new_coordinates)
        return VisualField(new_data, self._interpolator, self._renderer)

    def plot_voronoi(self, ax: plt.Axes = None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        return self._renderer.render_voronoi(self.data, ax, **kwargs)
    
    def plot_scatter(self, ax: plt.Axes = None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        return self._renderer.render_scatter(self.data, ax, **kwargs)
    
    def plot_numbering(self, ax: plt.Axes = None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        return self._renderer.render_numbering(self.data, ax, **kwargs)
    
    def filter(self, *criteria: Filter) -> 'VisualField':
        """Apply multiple FilterCriteria objects with AND logic"""
        mask = np.ones(self.data.n_locations, dtype=bool)

        for criterion in criteria:
            if isinstance(criterion, Filter):
                criterion_mask = criterion.evaluate(data=self.data)
                mask &= criterion_mask
            else:
                raise TypeError(f"Expected VisualFieldFilter or callable, got {type(criterion)}")
        
        # Create new filtered data
        filtered_data = VisualFieldData(
            x_coordinates=self.data.x_coordinates[mask],
            y_coordinates=self.data.y_coordinates[mask],
            sensitivity_values=self.data.sensitivity_values[mask],
            normative_values=self.data.normative_values[mask] if self.data.normative_values is not None else None,
            laterality=self.data.laterality
        )
        
        return VisualField(filtered_data, self._interpolator, self._renderer)
    
    # Factory methods for easy creation
    @classmethod
    def from_arrays(
        cls, 
        x: np.ndarray, 
        y: np.ndarray, 
        sensitivity: np.ndarray,
        normative: np.ndarray = None,
        laterality: Laterality = None
    ) -> 'VisualField':
        data = VisualFieldData(x, y, sensitivity, normative, laterality)
        return cls(data)
    
    @classmethod
    def from_dict(cls, data_dict: dict) -> 'VisualField':
        """Create from dictionary (useful for deserialization)"""
        return cls.from_arrays(
            x=np.array(data_dict['x_coordinates']),
            y=np.array(data_dict['y_coordinates']),
            sensitivity=np.array(data_dict['sensitivity_values']),
            normative=np.array(data_dict['normative_values']) if 'normative_values' in data_dict else None,
            laterality=data_dict.get('laterality')
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary (useful for serialization)"""
        result = {
            'x_coordinates': self.data.x_coordinates.tolist(),
            'y_coordinates': self.data.y_coordinates.tolist(),
            'sensitivity_values': self.data.sensitivity_values.tolist(),
        }
        if self.data.normative_values is not None:
            result['normative_values'] = self.data.normative_values.tolist()
        if self.data.laterality is not None:
            result['laterality'] = self.data.laterality.value
        return result