from typing import Callable, Union, List 
import logging

import numpy as np

from eyemod.visual_fields.core import VisualFieldData, Filter
from eyemod.definitions import Laterality

# Set up logger
logger = logging.getLogger(__name__)

class LambdaFilter:
    def __init__(self, function: Callable):
        self.filter_fn = function

    def evaluate(self, data: VisualFieldData) -> np.ndarray:
        return self.filter_fn(data)


class EccentricityFilter:
    def __init__(self, min_ecc: float = 0, max_ecc: float = np.inf):
        self.min_ecc = min_ecc
        self.max_ecc = max_ecc

    def evaluate(self, data: VisualFieldData) -> np.ndarray:
        ecc = np.sqrt(data.x_coordinates**2 + data.y_coordinates**2)
        return (ecc >= self.min_ecc) & (ecc <= self.max_ecc)


class ValueFilter:
    def __init__(self, min_val: float = -np.inf, max_val: float = np.inf, value_type: str = 'sensitivity'):
        self.min_val = min_val
        self.max_val = max_val
        self.val_type = value_type

    def evaluate(self, data: VisualFieldData) -> np.ndarray:
        if self.val_type == "sensitivity":
            values = data.sensitivity_values
        elif self.val_type == "deviation":
            values = data.deviation_values
        if values is None:
            raise ValueError(f'Missing values of type {self.val_type}')
        return (values >= self.min_val) & (values <= self.max_val)
    
class HemifieldFilter:
    def __init__(self, hemifields: Union[str, List[str]], logic: str = 'and'):
        if isinstance(hemifields, str):
            hemifields = [hemifields]
            
        self.hemifields = [sector.lower() for sector in hemifields]
        self.logic = logic
    
    def evaluate(self, data: VisualFieldData) -> np.ndarray:
        x = data.x_coordinates
        y = data.y_coordinates

        if not self.hemifields:
            return np.ones_like(x, dtype=bool)
        
        if data.laterality is None or data.laterality == Laterality.U:
            logger.warning('Hemifiled filtering is may be incorrect. Laterality is unknown.')
        elif data.laterality == Laterality.L:
            x = -1 * x

        masks = []
        for hemifield in self.hemifields:
            if hemifield == "nasal":
                mask = x <= 0
            elif hemifield == "temporal":
                mask = x >= 0
            elif hemifield == "inferior":
                mask = y <= 0
            elif hemifield == "superior":
                mask = y >= 0
            else:
                raise ValueError(f"Unknown hemifield: {hemifield}")
            masks.append(mask)

        if self.logic == 'and':
            return np.logical_and.reduce(masks)
        elif self.logic == 'or':
            return np.logical_or.reduce(masks)
        else:
            raise ValueError(f"Unknown logic: {self.logic}")
        
class GPatternClusterFilter:
    def __init__(self, clusters: Union[int, List[int]]):
        """
        Filter for specific Glaucoma Pattern clusters.
        
        Args:
            clusters: List of the cluster numbers
        """
        if isinstance(clusters, int):
            clusters = [clusters]
        self.clusters = clusters
    
    def evaluate(self, data: VisualFieldData) -> np.ndarray:
        """
        Evaluate the filter based on G-pattern cluster regions.
        
        Returns:
            Boolean mask indicating which points belong to specified clusters
        """
        x = data.x_coordinates
        y = data.y_coordinates
        
        if not self.clusters:
            return np.ones_like(x, dtype=bool)
        
        if data.laterality is None or data.laterality == Laterality.U:
            logger.warning('G-pattern cluster filtering may be incorrect. Laterality is unknown.')
        elif data.laterality == Laterality.R:
            x = -1 * x
        
        to_include = set()
        g_cluster = self.get_clusters()
        for c in self.clusters:
            to_include.update(g_cluster[c])

        mask = np.zeros_like(x)
        x = x.astype(int)
        y = y.astype(int)

        for i, coord in enumerate(zip(x, y)):
            if coord in to_include:
                mask[i] = 1

        return mask.astype(bool)
        
    def get_clusters(self):
        # clusters given for a left visual field
        return {
            1: {(2, 2), (-2, 2), (-4, 4), (-8, 2)},
            2: {(4, 4), (8, 2), (26, 4), (14, 4), (20, 4), (-2, 8), (2, 8)},
            3: {(-8, 8), (8, 8), (12, 12), (20, 12), (4, 14), (-4, 14)},
            4: {(-12, 12), (-12, 20), (-4, 20), (4, 20), (12, 20), (20, 20), (8, 26), (-8, 26)},
            5: {(-22, 4), (-26, 8), (-20, 20), (-20, 12)},
            6: {(-22, -4), (-26, -8), (-20, -20), (-20, -12)},
            7: {(-12, -12), (-12, -20), (-4, -20), (4, -20), (12, -20), (20, -20), (8, -26), (-8, -26)},
            8: {(20, -12), (4, -14), (-4, -14), (12, -12)},
            9: {(8, -2), (-3, -9), (3, -9), (-8, -8), (8, -8), (4, -4), (14, -4), (20, -4), (26, -4)},
            10: {(2, -2), (-4, -4), (-8, -2), (-2, -2)}
        }

