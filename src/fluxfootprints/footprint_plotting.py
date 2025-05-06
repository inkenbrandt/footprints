import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from typing import Optional, Tuple, List, Dict, Any
import xarray as xr
import pandas as pd


class FootprintPlotter:
    """
    Class to handle footprint function visualization and contour extraction.

    This class includes methods for calculating contour levels, extracting
    contour boundaries, and generating footprint plots including heatmaps
    and labeled isolines.

    Attributes
    ----------
    default_colormap : matplotlib.colors.Colormap
        Default colormap for footprint heatmaps (set to `cm.jet`).
    default_line_width : float
        Default line width for contour lines.
    """

    def __init__(self):
        self.default_colormap = cm.jet
        self.default_line_width = 0.5

    def get_contour_levels(
        self, f: np.ndarray, dx: float, dy: float, rs: Optional[List[float]] = None
    ) -> List[Tuple[float, float, float]]:
        """
        Calculate contour levels at specified percentages of total flux contribution.

        Parameters
        ----------
        f : numpy.ndarray
            2D footprint array representing flux contributions.
        dx : float
            Grid spacing in the x-direction (meters).
        dy : float
            Grid spacing in the y-direction (meters).
        rs : list of float, optional
            List of fractional cumulative contributions (e.g., [0.5, 0.8, 0.9]).
            If None, defaults to np.linspace(0.10, 0.90, 9).

        Returns
        -------
        list of tuple
            List of tuples, each containing:
            - cumulative fraction (float)
            - actual integrated area under that fraction (float)
            - footprint value at that contour level (float)

        Notes
        -----
        The function sorts the footprint values in descending order and
        calculates the contour levels that correspond to cumulative sums
        of those sorted values
        """
        if not isinstance(rs, (list, np.ndarray)):
            rs = np.linspace(0.10, 0.90, 9)

        pclevs = np.empty(len(rs))
        pclevs[:] = np.nan
        ars = np.empty(len(rs))
        ars[:] = np.nan

        # Sort footprint values in descending order
        sf = np.sort(f.flatten())[::-1]
        sf = np.ma.masked_array(sf, mask=(np.isnan(sf) | np.isinf(sf)))
        csf = sf.cumsum().filled(np.nan) * dx * dy

        # Find levels for each requested percentage
        for ix, r in enumerate(rs):
            dcsf = np.abs(csf - r)
            pclevs[ix] = sf[np.nanargmin(dcsf)]
            ars[ix] = csf[np.nanargmin(dcsf)]

        return [(round(r, 3), ar, pclev) for r, ar, pclev in zip(rs, ars, pclevs)]

    def get_contour_vertices(
        self, x: np.ndarray, y: np.ndarray, f: np.ndarray, lev: float
    ) -> Tuple[Optional[List], Optional[List]]:
        """
        Extract the (x, y) vertices of a contour line at a given footprint level.

        Parameters
        ----------
        x : numpy.ndarray
            2D array of x-coordinate grid points.
        y : numpy.ndarray
            2D array of y-coordinate grid points.
        f : numpy.ndarray
            2D footprint array representing flux contributions.
        lev : float
            Contour level value for which to extract vertices.

        Returns
        -------
        tuple
            Tuple of two lists:
            - x-coordinates of the contour vertices
            - y-coordinates of the contour vertices

            Returns (None, None) if contour is missing or touches domain boundary.

        Notes
        -----
        If the contour reaches the edge of the domain, it is assumed incomplete
        and will be ignored.
        """
        cs = plt.contour(x, y, f, [lev])
        plt.close()

        try:
            segs = cs.allsegs[0][0]
            xr = [vert[0] for vert in segs]
            yr = [vert[1] for vert in segs]

            # Check if contour reaches domain boundaries
            if (
                x.min() >= min(segs[:, 0])
                or max(segs[:, 0]) >= x.max()
                or y.min() >= min(segs[:, 1])
                or max(segs[:, 1]) >= y.max()
            ):
                return None, None

            return xr, yr
        except IndexError:
            return None, None

    def plot_footprint(
        self,
        x_2d: np.ndarray,
        y_2d: np.ndarray,
        fs: np.ndarray,
        clevs: Optional[List[float]] = None,
        show_heatmap: bool = True,
        normalize: Optional[str] = None,
        colormap=None,
        line_width: float = 0.5,
        iso_labels: Optional[bool] = None,
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the footprint heatmap and optional contour isolines.

        Parameters
        ----------
        x_2d : numpy.ndarray
            2D array of x-coordinate grid points (meshgrid).
        y_2d : numpy.ndarray
            2D array of y-coordinate grid points (meshgrid).
        fs : numpy.ndarray
            2D footprint array representing flux contributions.
        clevs : list of float, optional
            Contour levels to plot as isolines. If None, only heatmap is shown.
        show_heatmap : bool, optional
            If True, displays heatmap of the footprint function, by default True.
        normalize : str, optional
            If `'log'`, applies logarithmic normalization to the colormap.
            Default is None.
        colormap : matplotlib.colors.Colormap, optional
            Colormap to use for plotting. If None, uses `default_colormap`.
        line_width : float, optional
            Width of contour lines, by default 0.5.
        iso_labels : bool, optional
            If True, displays percentage labels on contour lines, by default None.
        title : str, optional
            Title for the plot.

        Returns
        -------
        tuple
            - matplotlib.figure.Figure object
            - matplotlib.axes.Axes object

        Notes
        -----
        - A marker is placed at (0, 0) to represent the flux tower location.
        - Log-normalization is useful for emphasizing lower-magnitude fluxes.
        """

        if colormap is None:
            colormap = self.default_colormap

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Handle contour levels if provided
        if clevs is not None:
            clevs = clevs[::-1]  # Reverse order for pyplot.contour
            clevs = [clev for clev in clevs if clev is not None]

            # Plot contours
            levs = clevs
            if show_heatmap:
                cp = ax.contour(x_2d, y_2d, fs, levs, colors="w", linewidths=line_width)
            else:
                cp = ax.contour(
                    x_2d, y_2d, fs, levs, colors=colormap(0.5), linewidths=line_width
                )

            # Add contour labels if requested
            if iso_labels:
                pers = [f"{int(clev*100)}%" for clev in clevs]
                fmt = dict(zip(cp.levels, pers))
                ax.clabel(cp, cp.levels, inline=1, fmt=fmt, fontsize=7)

        # Plot heatmap if requested
        if show_heatmap:
            norm = LogNorm() if normalize == "log" else None
            extent = (x_2d.min(), x_2d.max(), y_2d.min(), y_2d.max())
            im = ax.imshow(
                fs,
                cmap=colormap,
                extent=extent,
                norm=norm,
                origin="lower",
                aspect="equal",
            )
            plt.colorbar(
                im, ax=ax, shrink=0.8, format="%.2e", label="Flux contribution (m⁻²)"
            )

        # Set labels and title
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if title:
            ax.set_title(title)

        # Add tower location marker
        ax.plot(0, 0, "k^", markersize=10, label="Tower")
        ax.legend()

        return fig, ax


def add_plotting_to_footprint(FluxFootprint):
    """
    Adds plotting methods to the FluxFootprint class.

    This function dynamically attaches two plotting methods to a FluxFootprint class:
    `plot_footprint_climatology` and `plot_footprint_snapshot`, enabling visualization
    of footprint climatology and individual time-step snapshots using Matplotlib.

    Parameters
    ----------
    FluxFootprint : class
        A class representing flux footprint calculations, typically containing
        methods and properties for handling xarray Datasets of footprint data.

    Returns
    -------
    FluxFootprint : class
        The input class with additional methods:
            - `plot_footprint_climatology(self, ds, show_heatmap=True, title=None)`
            - `plot_footprint_snapshot(self, ds, time_index, show_heatmap=True)`
    """

    plotter = FootprintPlotter()

    def plot_footprint_climatology(
        self, ds: xr.Dataset, show_heatmap: bool = True, title: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the mean flux footprint (climatology) with contour overlays.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset containing the footprint variable over time and space.
            Must include attributes `grid_spacing`, and coordinates `x`, `y`, and `time`.
        show_heatmap : bool, optional
            Whether to show the heatmap background of the footprint, by default True.
        title : str, optional
            Title to display on the plot. If None, a default title with number of time steps is used.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object with the plotted footprint.
        """

        # Calculate mean footprint
        mean_footprint = ds.footprint.mean(dim="time")

        # Get contour levels for standard percentages
        dx = float(ds.attrs["grid_spacing"])
        dy = dx
        clevs = plotter.get_contour_levels(
            mean_footprint.values, dx, dy, rs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        )

        # Create plot
        fig, ax = plotter.plot_footprint(
            ds.x.values,
            ds.y.values,
            mean_footprint.values,
            clevs=[x[2] for x in clevs],
            show_heatmap=show_heatmap,
            iso_labels=True,
            title=title or f"Flux Footprint Climatology (n={len(ds.time)})",
        )

        return fig, ax

    def plot_footprint_snapshot(
        self, ds: xr.Dataset, time_index: int, show_heatmap: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot a single footprint snapshot for the specified time index.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset containing the footprint variable over time and space.
            Must include attributes `grid_spacing`, and coordinates `x`, `y`, and `time`.
        time_index : int
            Index of the time dimension for which to plot the footprint.
        show_heatmap : bool, optional
            Whether to show the heatmap background of the footprint, by default True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object with the plotted footprint.
        """

        # Get footprint for specified time
        footprint = ds.footprint.isel(time=time_index)
        timestamp = ds.time.isel(time=time_index).values

        # Calculate contour levels
        dx = float(ds.attrs["grid_spacing"])
        dy = dx
        clevs = plotter.get_contour_levels(
            footprint.values, dx, dy, rs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        )

        # Create plot
        fig, ax = plotter.plot_footprint(
            ds.x.values,
            ds.y.values,
            footprint.values,
            clevs=[x[2] for x in clevs],
            show_heatmap=show_heatmap,
            iso_labels=True,
            title=f"Flux Footprint {pd.to_datetime(timestamp)}",
        )

        return fig, ax

    # Add methods to class
    FluxFootprint.plot_footprint_climatology = plot_footprint_climatology
    FluxFootprint.plot_footprint_snapshot = plot_footprint_snapshot

    return FluxFootprint
