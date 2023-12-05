import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, convex_hull_plot_2d, qhull

def hsv_to_rgb(h, s, v):
    """
    Converts a color from HSV to RGB

    Parameters
    ----------
    h : float
    s : float
    v : float

    Returns
    -------
    numpy.ndarray
        The converted color in RGB space.
    """
    i = np.floor(h*6.0)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    i = i % 6

    if i == 0:
        rgb = (v, t, p)
    elif i == 1:
        rgb = (q, v, p)
    elif i == 2:
        rgb = (p, v, t)
    elif i == 3:
        rgb = (p, q, v)
    elif i == 4:
        rgb = (t, p, v)
    else:
        rgb = (v, p, q)

    return np.array(rgb, dtype=np.float32)


def get_distinct_colors(n, min_sat=.5, min_val=.5):
    """
    Generates a list of distinct colors, evenly separated in HSV space.

    Parameters
    ----------
    n : int
        Number of colors to generate.
    min_sat : float
        Minimum saturation.
    min_val : float
        Minimum brightness.

    Returns
    -------
    numpy.ndarray
        Array of shape (n, 3) containing the generated colors.

    """
    huePartition = 1.0 / (n + 1)
    hues = np.arange(0, n) * huePartition
    saturations = np.random.rand(n) * (1-min_sat) + min_sat
    values = np.random.rand(n) * (1-min_val) + min_val
    return np.stack([hsv_to_rgb(h, s, v) for h, s, v in zip(hues, saturations, values)], axis=0)


def colorize_segmentation(seg, ignore_label=None, ignore_color=(0, 0, 0)):
    """
    Randomly colorize a segmentation with a set of distinct colors.

    Parameters
    ----------
    seg : numpy.ndarray
        Segmentation to be colorized. Can have any shape, but counts type must be discrete.
    ignore_label : int
        Label of segment to be colored with ignore_color.
    ignore_color : tuple
        RGB color of segment labeled with ignore_label.

    Returns
    -------
    numpy.ndarray
        The randompy colored segmentation. The RGB channels are in the last axis.
    """
    assert isinstance(seg, np.ndarray)
    assert seg.dtype.kind in ('u', 'i')
    if ignore_label is not None:
        ignore_ind = seg == ignore_label
    seg = seg - np.min(seg)
    colors = get_distinct_colors(np.max(seg) + 1)
    np.random.shuffle(colors)
    result = colors[seg]
    if ignore_label is not None:
        result[ignore_ind] = ignore_color
    return result

def plot_dapi(dapi, polygon = None, draw_polygon = False, fov = None, vmax=None, f_title=0, invert_yaxis = True, invert_xaxis = False, s_units=1, f_lw=2):
    # draw_polygon = False
    offset = 0  # offset for x and y axis
    w = 10  # width of the figure
    f_fig = 0.9  # fraction of figure I want to cover with axes

    if fov is not None:
        xrange = fov[0]
        yrange = fov[1]
        x_min, x_max = xrange[0], xrange[1]
        y_min, y_max = yrange[0], yrange[1]
    else:
        x_min, x_max = 0, dapi.shape[1]
        y_min, y_max = 0, dapi.shape[0]

    rx = x_max - x_min + 2 * offset
    ry = y_max - y_min + 2 * offset

    aspect_ratio = ry / rx  # height/width
    figsize = (w, aspect_ratio * w)
    fig = plt.figure(figsize=(figsize[0], (1 + f_title) * figsize[1]))
    ax = fig.add_axes([(1 - f_fig) / 2, ((1 - f_fig) / 2) / (1 + f_title), f_fig, f_fig / (1 + f_title)])

    ppi = 72
    s_points = s_units * ppi * figsize[0] * f_fig / rx
    lw_points = s_points * f_lw

    ax.imshow(dapi[y_min:y_max, x_min: x_max], origin='lower', extent=(x_min, x_max, y_min, y_max), cmap='binary', vmax=vmax)
    # ax.imshow(polya[y_min:y_max, x_min: x_max], origin='lower', cmap='binary', vmax=0.4)
    if draw_polygon:
        if polygon:
            poly_x = polygon[0]
            poly_y = polygon[1]
            for i in range(len(poly_x)):
                if (min(poly_x[i]) > x_min) and (max(poly_x[i]) < x_max):
                    if (min(poly_y[i]) > y_min) and (max(poly_y[i]) < y_max):
                        ax.plot(poly_x[i], poly_y[i], c='k', linewidth=lw_points)
            # ax.scatter(x[cropped_index] - x_min,y[cropped_index] - y_min,s=0.1)
    ax.set_aspect('equal', 'datalim')
    ax.set_xlim(x_min - offset, x_max + offset)
    ax.set_ylim(y_min - offset, y_max + offset)
    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

def plot_fish_with_labels(col, x, y, z=None, z_selected=None, x_modes=None, y_modes=None, polygon=None, seg = None, dapi = None, bg_size_thresh = None, filter_bg = False, fov = None, vmax=None, fig_title='', f_title=0, font_size=20, invert_yaxis = True, invert_xaxis = False, s_units=1, f_modes=2, f_lw=2, hide_ticks = False, width = None, fraction_fig = None):
    offset = 0  # offset for x and y axis
    if width is None:
        w = 10  # width of the figure
    else:
        w = width
    if fraction_fig is None:
        f_fig = 0.9  # fraction of figure I want to cover with axes
    else:
        f_fig = fraction_fig
    s_units = s_units  # dia of circle in the units of input
    # f_title = 0.05  # percentage of height reserved for title

    if bg_size_thresh is not None:
        if (len(col.shape) == 1):
            uni, uni_inv, uni_counts = np.unique(col, return_counts=True, return_inverse=True)
            flt = uni_counts[uni_inv] < bg_size_thresh
            col[flt] = 0
            filter_bg = True
            if seg is not None:
                seg[flt] = 0
        else:
            if seg is not None:
                uni, uni_inv, uni_counts = np.unique(seg, return_counts=True, return_inverse=True)
                flt = uni_counts[uni_inv] < bg_size_thresh
                seg[flt] = 0

    if (len(col.shape) == 1):
        col = colorize_segmentation(col, ignore_label=0)

    if filter_bg:
        flt_bg = (col.sum(1) != 0)
        col, x, y = col[flt_bg], x[flt_bg], y[flt_bg]
        if z is not None:
            z = z[flt_bg]
        if x_modes is not None:
            x_modes = x_modes[flt_bg]
        if y_modes is not None:
            y_modes = y_modes[flt_bg]
        if seg is not None:
            seg = seg[flt_bg]

    if z is not None:
        if z_selected is None:
            z_selected = int(len(np.unique(z))/2)
        z_slice = np.unique(z)[z_selected]
        flt_slice = z == z_slice
        col, x, y = col[flt_slice], x[flt_slice], y[flt_slice]
        if x_modes is not None:
            x_modes = x_modes[flt_slice]
        if y_modes is not None:
            y_modes = y_modes[flt_slice]
        if seg is not None:
            seg = seg[flt_slice]

    if fov is not None:
        xrange = fov[0]
        yrange = fov[1]
        x_min, x_max = xrange[0], xrange[1]
        y_min, y_max = yrange[0], yrange[1]
    else:
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

    rx = x_max - x_min + 2 * offset
    ry = y_max - y_min + 2 * offset
    aspect_ratio = ry / rx  # height/width
    figsize = (w, aspect_ratio * w)
    ppi = 72
    s_points = s_units * ppi * figsize[0] * f_fig / rx
    lw_points = s_points * f_lw
    range_index = np.array(np.where((x > x_min) & (x < x_max) & (y > y_min) & (y < y_max))).squeeze()

    fig = plt.figure(figsize=(figsize[0], (1 + f_title) * figsize[1]))
    ax = fig.add_axes([(1 - f_fig) / 2, ((1 - f_fig) / 2)/(1 + f_title), f_fig, f_fig/(1 + f_title)])  # fig.add_subplot()
    if vmax is None:
        ax.scatter(x[range_index], y[range_index], s=s_points ** 2, c=col[range_index], marker='o')
    else:
        ax.scatter(x[range_index], y[range_index], s=s_points ** 2, c=col[range_index], marker='o',vmax=vmax)
    if (x_modes is not None) & (y_modes is not None):
        ax.scatter(x_modes[range_index], y_modes[range_index], s = f_modes*s_points**2, c = 'k', marker='o')
    if polygon is not None:
        if isinstance(polygon,tuple):
            poly_x = polygon[0]
            poly_y = polygon[1]
            for i in range(len(poly_x)):
                if (min(poly_x[i]) > x_min) and (max(poly_x[i]) < x_max):
                    if (min(poly_y[i]) > y_min) and (max(poly_y[i]) < y_max):
                        ax.plot(poly_x[i], poly_y[i], c='k', linewidth=lw_points)
        else:
            if z is not None:
                img = polygon[z_selected,y_min:y_max,x_min:x_max]
            else:
                img = polygon[y_min:y_max, x_min:x_max]
            x = np.arange(x_min, x_max, 1)
            y = np.arange(y_min, y_max, 1)
            X, Y = np.meshgrid(x, y)

            ax.contour(X, Y, img, colors='k', levels=len(np.unique(img)), linewidths=lw_points)
    if seg is not None:
        unique_cells = np.unique(seg[range_index])
        for ind in unique_cells:
            if ind!=0:
                x_cord = np.expand_dims(x[range_index][seg[range_index] == ind], axis=1)
                y_cord = np.expand_dims(y[range_index][seg[range_index] == ind], axis=1)
                points = np.concatenate((x_cord, y_cord), axis=1)

                try:
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1], 'k', linewidth=lw_points)
                except qhull.QhullError:
                    # print("here")
                    a = 1
    if dapi is not None:
        ax.imshow(dapi[y_min:y_max, x_min:x_max], origin='lower', extent=(x_min, x_max, y_min, y_max), cmap='binary')

    if not hide_ticks:
        ax.grid(True)
    ax.set_xlim(x_min - offset, x_max + offset)
    ax.set_ylim(y_min - offset, y_max + offset)
    if invert_yaxis:
        ax.invert_yaxis()
    if invert_xaxis:
        ax.invert_xaxis()
    ax.set_aspect('equal', 'datalim')
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(fig_title, fontsize=font_size)
    # fig.show()
    return fig

def plot_fish_data(counts, x, y, z=None, z_selected=None, seg = None, dapi = None, col=None, polygon=None, bg_size_thresh = None, filter_bg = False, filter_before_PCA = None, fov = None, fig_title = '', f_title=0, font_size=20, suppress_print = False, invert_yaxis=True, invert_xaxis=False, s_units=1, f_modes=2, f_lw=2, hide_ticks = False, width = None, fraction_fig = None):
    flag = 0
    if filter_before_PCA is not None:
        flag = 1
        flt = filter_before_PCA > 0
        col_full = np.zeros((filter_before_PCA.shape[0], 3))
    else:
        flt = np.array([True] * counts.shape[0])

    if col is None:
        tic = time.perf_counter()
        pca = PCA(n_components=3)
        pca_in = counts[flt]
        col = pca.fit_transform(pca_in)
        if not suppress_print:
            print('PCA completed: ' + str(time.perf_counter() - tic))

    mu = col.mean(axis=0)
    sigma = np.std(col, axis=0)
    col_znorm = (col - mu) / sigma
    col_clipped = np.zeros(col.shape)
    col_clipped[:, :] = col_znorm[:, :]
    lower = np.array((-1, -1, -1))  # np.array((-0.3, -1.5, -0.3))
    lower_rep = np.tile(lower, [col.shape[0], 1])
    upper = np.array((1.5, 1.5, 1.5))  # np.array((2, 1.5, 1))
    upper_rep = np.tile(upper, [col.shape[0], 1])
    col_clipped[col_clipped < lower_rep] = lower_rep[col_clipped < lower_rep]
    col_clipped[col_clipped > upper_rep] = upper_rep[col_clipped > upper_rep]
    col_clipped = (-lower_rep + col_clipped) / (upper_rep - lower_rep)

    if flag == 1:
        col_full[flt, :] = col_clipped
        col_full[~flt] = np.array([0.75, 0.75, 0.75])
        col_clipped = col_full

    fig = plot_fish_with_labels(col_clipped, x, y, z=z, z_selected=z_selected, polygon=polygon, seg=seg, dapi=dapi,
                                bg_size_thresh=bg_size_thresh, filter_bg=filter_bg, fov=fov, fig_title=fig_title,
                                f_title=f_title, font_size=font_size, invert_yaxis=invert_yaxis,
                                invert_xaxis=invert_xaxis, s_units=s_units, f_modes=f_modes, f_lw=f_lw,
                                hide_ticks=hide_ticks, width=width, fraction_fig=fraction_fig)
    return fig, col