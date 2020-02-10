#!/usr/bin/env python
"""
Miscellaneous yt/matplotlib wrapper methods

Aaron Tran
2019 March 14
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import yt

from axes_grid_AT import AxesGrid

def plot_triple_slice(ds, rho_lim=None, T_lim=None, annotate_grids=True,
        #rho_cmap='cubehelix', T_cmap='kamae',
        rho_cmap='arbre', T_cmap='arbre',  # yt default cmap
        annotate_velocity=False, annotate_cell_edges=False, **kwargs):
    """
    Plot three orthogonal slices of density and temperature through a dataset.
    Returns six-panel plot with two colorbars.

    Intended usage: user may annotate/modify plot via matplotlib interface
    and returned fig, grid objects.  Critically, such annotations must come
    after the call to SlicePlot._setup_plots() that is contained in this
    method.

    **kwargs gets passed to yt.SlicePlot(...)
    Useful kwargs for this method include:

        center=(x, y, z)  # in code units
        center=('min', 'density')
        center=('max', 'temperature')
        origin=(0, 0, 'native')
        window_size=12

    Developer notes:
    - The SlicePlot kwarg window_size controls the overall figure dimensions.
      I've not found how to control figure size via matplotlib calls.
    - requires a modified version of mpl_toolkit.axes_grid1.AxesGrid
    """
    if rho_lim is None or T_lim is None:
        ad = ds.all_data()
        if rho_lim is None:
            rho_lim = ad.quantities.extrema('density')
        if T_lim is None:
            T_lim = ad.quantities.extrema('temperature')

    #######################################

    # Modify dataset so that sliceplots with normal=y (i.e., "xz" slices)
    # will plot x horizontally and z vertically, see:
    # http://lists.spacepope.org/pipermail/yt-users-spacepope.org/2016-June/016926.html
    # http://lists.spacepope.org/pipermail/yt-users-spacepope.org/2016-June/016950.html

    ds_x_1_old = ds.coordinates.x_axis[1]
    ds_x_y_old = ds.coordinates.x_axis['y']
    ds_y_1_old = ds.coordinates.y_axis[1]
    ds_y_y_old = ds.coordinates.y_axis['y']

    ds.coordinates.x_axis[1] = 0
    ds.coordinates.x_axis['y'] = 0
    ds.coordinates.y_axis[1] = 2
    ds.coordinates.y_axis['y'] = 2

    #######################################

    fig = plt.figure()  # figsize does have any effect...
    grid = AxesGrid(fig, (0.01,0.01,0.99,0.99), nrows_ncols=(2,3), axes_pad=0.75,
                    direction='column', label_mode='all', share_all='none',
                    cbar_location='right', cbar_mode='edge',
                    cbar_size='5%', cbar_pad='5%')

    slc_xz = yt.SlicePlot(ds, 'y', ['density', 'temperature'], **kwargs)
    slc_yz = yt.SlicePlot(ds, 'x', ['density', 'temperature'], **kwargs)
    slc_xy = yt.SlicePlot(ds, 'z', ['density', 'temperature'], **kwargs)

    for slc in [slc_xz, slc_yz, slc_xy]:
        slc.set_font_size(12)
        slc.set_zlim('density', *rho_lim)
        slc.set_zlim('temperature', *T_lim)
        if annotate_grids:
            slc.annotate_grids()
        if annotate_velocity:
            slc.annotate_velocity()
        if annotate_cell_edges:
            slc.annotate_cell_edges()
        # TODO allow user configuration...
        slc.set_cmap('density', rho_cmap)
        slc.set_cmap('temperature', T_cmap)

    slc_xz.plots['density'].figure = fig
    slc_xz.plots['density'].axes = grid[0].axes
    slc_xz.plots['density'].cax = grid.cbar_axes[0]
    slc_xz.plots['temperature'].figure = fig
    slc_xz.plots['temperature'].axes = grid[1].axes
    slc_xz.plots['temperature'].cax = grid.cbar_axes[1]

    slc_yz.plots['density'].figure = fig
    slc_yz.plots['density'].axes = grid[2].axes
    #slc_yz.plots['density'].cax = grid.cbar_axes[2]
    slc_yz.plots['temperature'].figure = fig
    slc_yz.plots['temperature'].axes = grid[3].axes
    #slc_yz.plots['temperature'].cax = grid.cbar_axes[3]

    slc_xy.plots['density'].figure = fig
    slc_xy.plots['density'].axes = grid[4].axes
    #slc_xy.plots['density'].cax = grid.cbar_axes[4]
    slc_xy.plots['temperature'].figure = fig
    slc_xy.plots['temperature'].axes = grid[5].axes
    #slc_xy.plots['temperature'].cax = grid.cbar_axes[5]

    slc_xz._setup_plots()
    slc_yz._setup_plots()
    slc_xy._setup_plots()

    # cleanup
    ds.coordinates.x_axis[1] = ds_x_1_old
    ds.coordinates.x_axis['y'] = ds_x_y_old
    ds.coordinates.y_axis[1] = ds_y_1_old
    ds.coordinates.y_axis['y'] = ds_y_y_old

    return fig, grid


def plot_ray_field(ax, ray, field, xunit=None, yunit=None, fmt=None, **kwargs):
    """
    Plots field values contained by a YTRay or YTOrthoRay
    Routine obtained and modified from M. W. Abruzzo 2019 Mar 14

    Inputs:
        ax: matplotlib axes object
        ray: YTRay or YTOrthoRay
        field: YT field contained in dataset associated w/ ray
        xunit: YT unit passed to field in_units(..)
        yunit: YT unit passed to field in_units(..)
        fmt: ax.plot format spec
        **kwargs: passed to ax.plot(...)
    """
    # first we need to compute distance along the ray
    if hasattr(ray,'axis'):
        axis_id = ray.axis
        for axis in ['x','y','z']:
            if ray.ds.coordinates.axis_id[axis] == axis_id:
                dists = ray[axis]
                break
        idx = slice(None)
    else:
        # ray data is not ordered by default; must re-order for plotting
        idx = np.argsort(ray['t'])
        start_i = np.argmin(ray['t'])
        final_i = np.argmax(ray['t'])
        delta_comp = [ray[dim][final_i] - ray[dim][start_i] \
                      for dim in ('x','y','z')]
        delta_comp = yt.YTArray(delta_comp)
        if xunit is None:
            dists = ray['t'][idx] * np.sum(delta_comp**2)**0.5
        else:
            dists = ray['t'][idx] * np.sum(delta_comp.in_units(xunit)**2)**0.5

    if fmt is None:
        args = ()
    else:
        args = (fmt,)

    if yunit is None:
        return ax.plot(dists, ray[field][idx].value, *args, **kwargs)
    else:
        return ax.plot(dists, ray[field][idx].in_units(yunit).value, *args, **kwargs)


def quicklook_xslice(ds):
    """Exactly what it sounds like"""
    fig = plt.figure(figsize=(12,5))  # figsize does not seem to have any effect...
    grid = AxesGrid(fig, (0.05,0.05,0.95,0.95),
                    nrows_ncols=(1,2), axes_pad=1.5,
                    label_mode='L', share_all=True,
                    cbar_location='right', cbar_mode='each',
                    cbar_size='3%', cbar_pad='3%')

    slc = yt.SlicePlot(ds, 'x', ['density', 'temperature'])
    slc.annotate_velocity()
    slc.annotate_grids()

    slc.plots['density'].figure = fig
    slc.plots['density'].axes = grid[0].axes
    slc.plots['density'].cax = grid.cbar_axes[0]
    slc.plots['temperature'].figure = fig
    slc.plots['temperature'].axes = grid[1].axes
    slc.plots['temperature'].cax = grid.cbar_axes[1]

    slc._setup_plots()

    # Annotate using matplotlib interface rather than yt
    # This must follow slc._setup_plots()
    stamp = "t = {:.2e} s = {:.2f} kyr".format(ds.current_time.to('s'),
                                               ds.current_time.to('kyr'))
    ax0 = grid.axes_all[0]
    ax1 = grid.axes_all[1]
    ax0.set_title(ds.basename, fontsize=12)
    ax1.set_title(stamp, fontsize=12)

    return fig, grid


if __name__ == '__main__':
    pass
