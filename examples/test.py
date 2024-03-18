from boms import run_boms
import numpy as np
import csv
import time

def read_data(name):
    x = []
    y = []
    g = []
    boms = []
    with open('./examples/example_data.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        line_count = 0
        for row in readCSV:
            if line_count > 0:
                x.append(float(row[1]))
                y.append(float(row[2]))
                g.append((row[3]))
                boms.append(int(row[5]))
            line_count += 1
    x = np.array(x)
    y = np.array(y)
    z = None
    g = np.array(g)
    boms = np.array(boms)

    return x, y, g, boms

tic = time.perf_counter()
name = 'allen_smfish'
x, y, g, boms = read_data(name)


h_s = 17.5
h_r = 0.4
K = 30

modes, seg, count_mat, cell_loc, coords = run_boms(x, y, g, 30, h_s, h_r, K=K, verbose=True)
modes2, seg2, count_mat2, cell_loc2, coords2 = run_boms(x, y, g, 30, h_s, h_r, K=K, verbose=True, x_min=3744, x_max=4494, y_min=2256, y_max=3006)

print(f'Overall time: {time.perf_counter() - tic}')
#%%
from sklearn.metrics import normalized_mutual_info_score

result = np.array_equal(seg, boms)
if result:
	print("BOMS ran successfully!")
else:
	print("Unsuccessful - results don't match.")
	print(f'normalised mutual information between the two seg: {normalized_mutual_info_score(boms, seg)}')
#%%
from boms.utils_vis import plot_fish_with_labels
from boms import seg_for_polygons

fov = [[3744, 4494], [2256, 3006]]
invert_yaxis = True
invert_xaxis = True
s_units = 1
f_lw = 0.5
seg_in = seg_for_polygons(coords[:, 0], coords[:, 1], seg, h_s)
fig = plot_fish_with_labels(seg, coords[:, 0], coords[:, 1], seg=seg_in, filter_bg=True, fov = fov, invert_xaxis=invert_xaxis, invert_yaxis=invert_yaxis, s_units=s_units, f_lw=f_lw)
fig.show()
#%%
seg_in = seg_for_polygons(coords2[:, 0], coords2[:, 1], seg2, h_s)
fig = plot_fish_with_labels(seg2, coords2[:, 0], coords2[:, 1], seg=seg_in, filter_bg=True, fov = fov, invert_xaxis=invert_xaxis, invert_yaxis=invert_yaxis, s_units=s_units, f_lw=f_lw)
fig.show()

