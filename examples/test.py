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

    gene_name = np.unique(g)
    genes = np.zeros(g.shape, 'int32')
    for i in range(len(gene_name)):
        genes[np.where((g == gene_name[i]))[0]] = i

    return x, y, genes, boms

tic = time.perf_counter()
name = 'allen_smfish'
x, y, g, boms = read_data(name)


h_s = 17.5
h_r = 0.4
K = 30

modes, seg = run_boms(x, y, g, 30, h_s, h_r, K=K)

print(f'Overall time: {time.perf_counter() - tic}')
#%%
from sklearn.metrics import normalized_mutual_info_score

result = np.array_equal(seg, boms)
if result:
	print("BOMS ran successfully!")
else:
	print("Unsuccessful - results don't match.")
	print(f'normalised mutual information between the two seg: {normalized_mutual_info_score(boms, seg)}')
