from __future__ import annotations 
from typing import Dict, Tuple 

import numpy as np
import matplotlib
import matplotlib.axes
import matplotlib.patches
import matplotlib.pyplot as plt

import scipy.spatial as spatial

# from aimitools.utils.voronoi import plot_voronoi
from eyemod.visual_fields.core import VisualFieldData

class VisualFieldRenderer():

    def render_voronoi(self, data: VisualFieldData, ax: plt.Axes = None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Render visual field data as a Voronoi diagram.
        
        Args:
            data: VisualFieldData object containing coordinates and values
            ax: Matplotlib axis to plot on. If None, creates new figure/axis
            **kwargs: Additional plotting options
                - vf_radius: Visual field radius (default: 30)
                - value_range: Range for color normalization (default: (0, 40))
                - value_type: Type of values to plot ('sensitivity', 'deviation', 'normative')
                - cmap: Colormap to use (default: plt.cm.viridis)
                - show_values: Show values as text (default: True)
                - show_lines: Show Voronoi lines (default: True)
                - show_points: Show data points (default: False)
                - show_colorbar: Show colorbar (default: True)
        
        Returns:
            Tuple of (figure, axis)
        """
        vf_radius = kwargs.get('vf_radius', 30)
        value_type = kwargs.get('value_type', 'sensitivity')
        show_values = kwargs.get('show_values', True)
        show_lines = kwargs.get('show_lines', True)
        show_points = kwargs.get('show_points', False)
        show_colorbar = kwargs.get('show_colorbar', True)

        # Get values and appropriate defaults based on type
        values, value_range, default_cmap = self._get_values_and_defaults(data, value_type)

        # Use type-specific defaults if not explicitly provided
        cmap = kwargs.get('cmap', default_cmap)
        value_range = kwargs.get('value_range', value_range)

        assert values.ndim == 1, f"Expected 1D array for values, got {values.ndim}D array"

        # Create figure/axis if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure

        # Generate voronoi tessellation
        voronoi = spatial.Voronoi(data.coordinates)

        # Set up normalization and colormap
        norm = plt.Normalize(*value_range, clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

        # # Plot the Voronoi diagram
        # plot_voronoi(
        #     voronoi,
        #     region_values=values,
        #     ax=ax,
        #     cmap=cmap,
        #     norm=norm,
        #     show_values=show_values,
        #     show_points=show_points,
        #     show_lines=show_lines,
        # )

        # Style the plot
        _add_common_style(ax, vf_radius)
        ax.axis('off')

        if show_lines:
            _add_circle_line(ax, vf_radius)   

        if show_colorbar:
            plt.colorbar(mapper, label=value_type, ax=ax)

        return fig, ax

    def render_scatter(self, data: VisualFieldData, ax: plt.Axes = None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Render visual field data as a scatter plot.
        
        Args:
            data: VisualFieldData object containing coordinates and values
            ax: Matplotlib axis to plot on. If None, creates new figure/axis
            **kwargs: Additional plotting options
                - vf_radius: Visual field radius (default: 30)
                - value_range: Range for color normalization (default: (0, 40))
                - value_type: Type of values to plot ('sensitivity', 'deviation', 'normative')
                - cmap: Colormap to use (default: plt.cm.viridis)
                - show_values: Show values as text (default: False)
                - show_colorbar: Show colorbar (default: True)
                - alpha: Transparency of markers (default: 0.8)
                - Additional scatter plot kwargs are passed through
        
        Returns:
            Tuple of (figure, axis)
        """
        vf_radius = kwargs.pop('vf_radius', 30)
        value_type = kwargs.pop('value_type', 'sensitivity')
        cmap = kwargs.pop('cmap', plt.cm.viridis)
        show_values = kwargs.pop('show_values', False)
        show_colorbar = kwargs.pop('show_colorbar', True)
    

        # Get values and appropriate defaults based on type
        values, value_range, default_cmap = self._get_values_and_defaults(data, value_type)

        # Use type-specific defaults if not explicitly provided
        cmap = kwargs.get('cmap', default_cmap)
        value_range = kwargs.get('value_range', value_range)

        # Create figure/axis if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure

        # Set up normalization and colormap
        norm = plt.Normalize(*value_range, clip=True)

        # Create scatter plot
        scatter = ax.scatter(
            data.coordinates[:, 0], 
            data.coordinates[:, 1],
            c=values,
            cmap=cmap,
            norm=norm,
            linewidth=0.5,
            **kwargs
        )

        # Add values as text if requested
        if show_values:
            for i, (x, y) in enumerate(data.coordinates):
                value = values[i]
                text = f"{value:.1f}" if isinstance(value, float) else str(value)
                ax.text(
                    x, y, text,
                    verticalalignment='center',
                    horizontalalignment='center',
                    fontsize=8,
                    fontweight='bold',
                    color='white' if values[i] < np.mean(value_range) else 'black'
                )

        # Style the plot
        _add_common_style(ax, vf_radius)
        _add_circle_line(ax, vf_radius)
        ax.axis('off')

        if show_colorbar:
            plt.colorbar(scatter, label=value_type, ax=ax)

        return fig, ax

    def render_numbering(self, data: VisualFieldData, ax: plt.Axes = None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Render visual field data showing location numbering.
        
        Args:
            data: VisualFieldData object containing coordinates
            ax: Matplotlib axis to plot on. If None, creates new figure/axis
            **kwargs: Additional plotting options
                - vf_radius: Visual field radius (default: 30)
                - show_ticks: Show axis ticks (default: False)
                - fontsize: Font size for numbers (default: 10)
                - fontcolor: Font color for numbers (default: 'black')
                - fontweight: Font weight for numbers (default: 'bold')
                - background_color: Background color for numbers (default: None)
                - show_grid: Show grid lines (default: False)
        
        Returns:
            Tuple of (figure, axis)
        """
        vf_radius = kwargs.get('vf_radius', 30)
        show_ticks = kwargs.get('show_ticks', False)
        fontsize = kwargs.get('fontsize', 10)
        fontcolor = kwargs.get('fontcolor', 'black')
        fontweight = kwargs.get('fontweight', 'bold')

        # Create figure/axis if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure

        text_kwargs = {
            'verticalalignment': 'center',
            'horizontalalignment': 'center',
            'fontsize': fontsize,
            'color': fontcolor,
            'weight': fontweight
        }

        # Plot the location numbers
        for i, (x, y) in enumerate(data.coordinates):  
            ax.text(x, y, str(i), **text_kwargs)

        # Style the plot
        _add_common_style(ax, vf_radius)
        _make_axis_central(ax, show_ticks=show_ticks)
        _add_circle_line(ax, vf_radius)
        
        ax.set_title("Location Numbering")
        
        return fig, ax

    def _get_values_and_defaults(self, data: VisualFieldData, value_type: str) -> Tuple[np.ndarray, Tuple, matplotlib.colors.Colormap]:
        """Get values array and appropriate defaults based on value type.
        
        Args:
            data: VisualFieldData object
            value_type: Type of values ('sensitivity', 'deviation', 'normative')            
        Returns:
            Tuple of (values_array, default_value_range, default_colormap)
            
        Raises:
            ValueError: If deviation/normative values are requested but not available
        """
        if value_type == 'sensitivity':
            values = data.sensitivity_values
            value_range = (0, 40)
            cmap = plt.cm.viridis
        elif value_type == 'deviation':
            if data.deviation_values is None:
                raise ValueError('Deviation values not available - normative_values must be provided in VisualFieldData')
            values = data.deviation_values
            value_range = (-10, 10)  # Typical deviation range
            cmap = plt.cm.RdBu  # Red-blue diverging colormap, red for negative deviations
        elif value_type == 'normative':
            if data.normative_values is None:
                raise ValueError('Normative values not available in VisualFieldData')
            values = data.normative_values
            value_range = (0, 40)
            cmap = plt.cm.viridis
        else:
            raise ValueError(f"Unknown value_type: {value_type}. Must be 'sensitivity', 'deviation', or 'normative'")
            
        return values, value_range, cmap

def _add_common_style(ax: matplotlib.axes.Axes, radius) -> matplotlib.axes.Axes:

    ax.set_aspect('equal')
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)

    _crop_to_circle(ax, radius)
    return ax

def _crop_to_circle(ax: matplotlib.axes.Axes, radius: float) -> None:
    """Crop the visual field plot to a circle."""

    center = (0, 0)
    clipping_circle = matplotlib.patches.Circle(center, radius, transform=ax.transData)

    # the voronoi lines are stored in collections
    for col in ax.collections:
        col.set_clip_path(clipping_circle)

    # the colored voronoi regions are stored in patches
    for patch in ax.patches:
        patch.set_clip_path(clipping_circle)

    return ax

def _add_circle_line(ax: matplotlib.axes.Axes, radius: float) -> matplotlib.axes.Axes:
    """Add a circle line to the plot."""
    circle = matplotlib.patches.Circle((0, 0), radius, fill=False, clip_on=False )
    ax.add_patch(circle)
    return ax

def _make_axis_central(ax: matplotlib.axes.Axes, show_ticks: bool = True) -> matplotlib.axes.Axes:
    """Make the axis central by moving spines to center and optionally showing ticks.
    
    Args:
        ax: Matplotlib axis to modify
        show_ticks: Whether to show tick labels (default: True)
        
    Returns:
        Modified matplotlib axis
    """
    # move axis to the center
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # remove the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if show_ticks:
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()

        # remove the 0 tick label at the origin
        x_ticks = x_ticks[x_ticks != 0]
        y_ticks = y_ticks[y_ticks != 0]

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
    else:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    return ax