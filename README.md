## BOMS : Cell Segmentation method for Spatial Transcriptomics

![BOMS Overview](images/method_overview.jpg)

BOMS is a tool for cell segmentation in fluorescent in-situ hybridization (FISH) based Spatial Transcriptomics datasets. It takes as input the gene locations and labels. It assumes that a cell body is homogenous in its transcriptional signature and uses the similarity of these neighborhoods to cluster them together as one cell. The method can also incorporate the flows obtained from Cellpose Segmentation on DAPI/Cell Membrane channels to improve its cell segmentation.

### Installation

The package requires Python > 3.9. The package can be installed using pip as follows:

```bash :
pip install boms
```

### Usage

The data for the method is provided in the form of three ```numpy arrays``` : ```x``` representing the x coordinates of the mRNA spots, ```y``` representing the y coordinates of the mRNA spots and ```g``` representing the labels of the mRNA spots. The cell segmentation can be performed as follows:


```python :
from boms import run_boms

"""
:param epochs: Number of iterations for the BOMS algorithm. Recommendation: 30
:param h_s: Spatial Bandwidth. Recommendation: Roughly equal to the radius of the cell body.
:param h_r: Range Bandwidth. Recommendation: 0.3 - 0.5
:param K: Number of Nearest Neighbors to form the Neighborhood Gene Expression Profile. Recommendation: 30

:return modes: N x (2 + no. of genes) array containing the final modes.
:return seg: N x 1 array containing the final segmentation.
"""

modes, seg = run_boms(x, y, g, epochs=30, h_s=10, h_r=0.3, K=30)
```

### Demo

A demo notebook is available to run on Google Colab - [BOMS Demo](https://colab.research.google.com/drive/16YgR92sc3ai9mheYUb8_SCdo9hjc3-xZ?usp=sharing)

