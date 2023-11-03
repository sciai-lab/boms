# import cppimport
import numpy as np
import time
from scipy.cluster.hierarchy import fclusterdata
import mkl
import boms_wrapper as meanshift_cpp
# import sys
# sys.path.append('./boms/')

def run_boms(x, y, genes, epochs: int, h_s: float, h_r: float, z = None, K: int = 30, use_flows: bool = None, flows = None, alpha: float = 1):
    tic = time.perf_counter()
    # meanshift_cpp = cppimport.imp("meanshift_wrapper")
    # print("done importing")

    assert len(x) == len(y) == len(genes), "x, y, and genes must be the same length"
    coords = np.concatenate((x[:,None],y[:,None]),axis=1)
    genes = genes[:,None]
    if use_flows is None:
        if flows is not None:
            use_flows = 1
        else:
            use_flows = 0
    else:
        use_flows = int(use_flows)
    if flows is None:
        flows = np.array([])
    else:
        if alpha < 0.2:
            max_val = np.max(np.abs(flows))
            flows = flows / max_val
    modes = meanshift_cpp.meanshift_cpp(coords.astype('float32'), genes.astype('float32'), len(np.unique(genes)), K, epochs, h_s, h_r, use_flows, flows, alpha)
    modes = np.reshape(modes, (len(x), -1))
    print(f'time taken for Meanshift: {time.perf_counter() - tic}')

    tic = time.perf_counter()
    modes_unique, modes_unique_inv = np.unique(np.round(modes[:, 0:2], 0), axis=0, return_inverse=True)

    heir_clus = fclusterdata(modes_unique[:, 0:2], h_s / 2 / 4, criterion="distance")
    seg = heir_clus[modes_unique_inv]
    print(f'time taken for mode collapsing: {time.perf_counter() - tic}')
    return modes, seg

def smooth_ge(x, y, genes, K = 30):
    tic = time.perf_counter()
    # meanshift_cpp = cppimport.imp("meanshift_wrapper")
    # print("done importing")

    assert len(x) == len(y) == len(genes), "x, y, and genes must be the same length"
    coords = np.concatenate((x[:, None], y[:, None]), axis=1)
    genes = genes[:, None]
    counts = meanshift_cpp.smooth_ge_cpp(coords.astype('float32'), genes.astype('float32'), len(np.unique(genes)), K)
    counts = np.reshape(counts, (len(x), -1))
    print(f'time taken for smoothing: {time.perf_counter() - tic}')
    return counts[:, 2:]
def seg_for_polygons(x, y, seg, h, thresh_val = 2):
    tic = time.perf_counter()
    # meanshift_cpp = cppimport.imp("meanshift_wrapper")
    # print("done importing")

    coords = np.concatenate((x[:, None], y[:, None]), axis=1)

    density = meanshift_cpp.density_estimator_cpp(coords.astype('float32'), seg.astype('int32'), h)
    print(f'time taken for density estimation: {time.perf_counter() - tic}')

    density = density / (h ** 2)
    a_den = np.bincount(seg, weights=density)
    a_size = np.bincount(seg)
    a_den_mean_per_cell = np.divide(a_den, a_size, where=(a_size != 0), out=np.zeros(a_size.shape))
    thresh = a_den_mean_per_cell[seg]

    flt = density > thresh / thresh_val
    flt2 = seg > 0
    flt = flt * flt2

    seg_in = np.zeros(len(seg), 'int32')
    seg_in[flt] = seg[flt]
    return seg_in
