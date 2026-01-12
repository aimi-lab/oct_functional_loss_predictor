from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata

from eyemod.visual_fields.core import VisualFieldData, Interpolator

class VisualFieldInterpolator:
    """Interpolates visual field data to new coordinate locations."""

    def interpolate(self, data: VisualFieldData, new_coords: np.ndarray) -> VisualFieldData:
        """Interpolate visual field data to new coordinate locations.

        Use linear interpolation within the for values within the convex hull of the original coordinates
        and nearest neighbor interpolation for values outside of the convex hull.
        
        Args:
            data: Original VisualFieldData containing coordinates and values to interpolate from.
            new_coords: Array of new coordinates with shape (n_points, 2) where each row
                       is [x, y] coordinates.
        
        Returns:
            New VisualFieldData object with interpolated values at the new coordinates.
            If the original data contains normative values, they will also be interpolated.
        
        Raises:
            ValueError: If new_coords doesn't have the correct shape (n_points, 2).
        """
        if new_coords.ndim != 2 or new_coords.shape[1] != 2:
            raise ValueError(f"new_coords must have shape (n_points, 2), got {new_coords.shape}")

        new_sensitivity = self._interpolate(
            data.sensitivity_values, data.coordinates, new_coords
        )
        new_normative = (
            self._interpolate(data.normative_values, data.coordinates, new_coords)
            if data.normative_values is not None
            else None
        )

        return VisualFieldData(
            x_coordinates=new_coords[:, 0],
            y_coordinates=new_coords[:, 1],
            sensitivity_values=new_sensitivity,
            normative_values=new_normative,
        )

    def _interpolate(self, values: np.ndarray, old_coords: np.ndarray, new_coords: np.ndarray) -> np.ndarray:
        """Perform interpolation using linear method with nearest neighbor for values outside of the convex hull.
        
        Linear interpolation is used where possible, with nearest neighbor interpolation 
        used for points outside the convex hull of the original data.
        
        Args:
            values: Array of values to interpolate from, shape (n_old_points,).
            old_coords: Array of original coordinates, shape (n_old_points, 2).
            new_coords: Array of target coordinates, shape (n_new_points, 2).
            
        Returns:
            Array of interpolated values at new coordinates, shape (n_new_points,).
            
        Note:
            Points outside the convex hull of old_coords will use nearest neighbor
            interpolation to avoid NaN values.
        """
        # Linear interpolation first
        new_values_lin = griddata(old_coords, values, new_coords, method='linear')
        
        # Nearest neighbor interpolation for values outside of the convex hull
        new_values_nn = griddata(old_coords, values, new_coords, method='nearest')

        # Fill NaN values from linear interpolation with nearest neighbor values
        missing_mask = np.isnan(new_values_lin)
        new_values_lin[missing_mask] = new_values_nn[missing_mask]

        return new_values_lin
