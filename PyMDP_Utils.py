"""
Utilities for PyMDP module
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

def BarycentricInterpolation(bins, pnts):
    """
    barycentricinterpolation for given points, 
    return the barycentric coordinates for points within the grids
    INPUT
    bins -      grids for discretization, 
                m-length array where bins[i] indicates the mesh along dimension i
    pnts -      an array of pnts, each points is an m-length indicates the Cartesian coordinates
                can be n pnts in total
    RETURN
    indices -   an n-length list of indices, each indices is d-length (d=m+1) for interpolating points invovled
    coeffs  -   an n-length list of coefficients, each coefficients is d-length for reconstructing points n

    A pythonic version barycentricinterpolation from Russ' drake utility function
    does not support dcoefs currently...
    """

    #note here the layout of input and output is different from the C++ version of drake
    m = pnts.shape[1]
    n = pnts.shape[0]
    d = m+1

    if len(bins) != m:
        print 'The number of bins must equal to the dimension of the points.'   #validation
        return None, None

    binsize = [len(bins[i]) for i in range(m)]
    nskip = np.concatenate([[1], np.cumprod([binsize[i] for i in range(m-1)])])

    #a list of bary points for future sorting...
    b = [{'dim':0, 'fracway':0.0, 'dfracway':0.0} for i in range(d)]
    indices = np.zeros((n, d))
    coeffs = np.zeros((n, d))
    for j in range(n):
        sidx = 0    # 0-index in our case...
        for i in range(m):
            pt = pnts[j, i]
            curr_bin = bins[i]
            curr_bin_size = binsize[i]

            b[i]['dim'] = i

            if curr_bin_size == 1:  #singleton dimensions
                #sidx is unchanged
                b[i]['fracway'] = 1.0
            elif pt > curr_bin[curr_bin_size-1]:
                #larger than max bound of bin
                sidx += nskip[i] * (curr_bin_size-1)
                b[i]['fracway'] = 1.0
                b[i]['dfracway'] = 0.0
            elif pt < curr_bin[0]:
                #less than min bound of bin
                sidx += nskip[i]
                b[i]['fracway'] = 0.0
                b[i]['dfracway'] = 0.0
            else:
                #Russ commented that smarter search can be done here...
                #i guess we can do it in a pythonic way...
                next_bin_index = np.argmax(curr_bin>pt)

                sidx += nskip[i]*next_bin_index
                b[i]['fracway'] = (pt - curr_bin[next_bin_index-1])/(curr_bin[next_bin_index]- curr_bin[next_bin_index-1])
                b[i]['dfracway'] = 1./(curr_bin[next_bin_index]- curr_bin[next_bin_index-1])

        #sort dimension based on fracway (lowest to highest)
        b_sorted = sorted(b[:-1], key=lambda b_elem: b_elem['fracway'])

        # final element of b_sorted,
        b_sorted.append({'dim':m-1,'fracway':1.0, 'dfracway':0.0})

        # top right corner
        indices[j, 0] = sidx
        coeffs[j, 0] = b_sorted[0]['fracway']

        for i in range(m):
            if binsize[b_sorted[i]['dim']] > 1:
                #support singletone dimension
                sidx -= nskip[b_sorted[i]['dim']]
            indices[j, i+1] = sidx
            coeffs[j, i+1] = b_sorted[i+1]['fracway'] - b_sorted[i]['fracway']

    return indices, coeffs

def add_arrow_to_line2D(
    axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
    arrowstyle='-|>', arrowsize=1, transform=None):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes: 
    line: list of 1 Line2D obbject as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """
    if (not(isinstance(line, list)) or not(isinstance(line[0], 
                                           mlines.Line2D))):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line[0].get_xdata(), line[0].get_ydata()

    arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)

    color = line[0].get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multicolor lines not supported")
    else:
        arrow_kw['color'] = color

    linewidth = line[0].get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows