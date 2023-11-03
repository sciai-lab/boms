#define _CRT_SECURE_NO_WARNINGS
//#define EIGEN_VECTORIZE
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
//#define EIGEN_DONT_PARALLELIZE
//#include <pybind11/pybind11.h> //needed for checking ctrl+c
//#include <pybind11/eigen.h> //needed for checking ctrl+c
//namespace py = pybind11; //needed for checking ctrl+c

#include <iostream>
#include <algorithm>
#include <queue>
#include <tuple>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream> 
#include <chrono>
#include <cmath>
#include <numeric>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace std;
using namespace std::chrono;

float* create_dataset(float* arr_s, float* arr_r, int dim_s, int dim_r, int len) {
    float* data = new float[len * (dim_s + dim_r)];

    for (int i = 0; i < len;++i) {
        for (int j = 0;j < dim_s;++j) {
            data[(dim_s + dim_r) * i + j] = arr_s[dim_s * i + j];
        }
        for (int j = dim_s; j < (dim_s + dim_r); ++j) {
            data[(dim_s + dim_r) * i + j] = arr_r[dim_r * i + j - dim_s];
        }
    }
    return data;
}

/*
arr_in: input array
len: length of the array
dim_s: dimension of the spatial coordinates
dim_r: dimension of the range coordinates
arr_out: output array containing the euclidean norm of the range part of each point
*/
void euclidean_norm(float* arr_in, int len, int dim_s, int dim_r, float* arr_out) {
    float* var = arr_in;
    float norm;
    for (int n = 0; n < len; ++n) {
        //s = dim * n;

        norm = 0;
        for (int i = 0; i < dim_s;++i) {
            var++;
        }
        for (int i = 0; i < dim_r;++i) {
            norm += *var * *var;
            var++;
        }
        arr_out[n] = sqrt(norm);
    }
}

/*
function that fills arr_out with integers from 0 to len-1
*/
void get_sample_id(int len, int* arr_out) {
    for (int i = 0; i < len;++i) {
        arr_out[i] = i;
    }
}

struct sort_node
{
    int* sample_id;
    float* data;
};

int partition_qsort(sort_node* arr_node, int lo, int hi) {
    int mid, i, j, pivotIndex;
    float pivot_val_x, pivot_val_y;

    while (lo <= hi) {
        if (lo == hi) {
            return lo;
        }

        //Hoare Partitioning of the dataset
        mid = (lo + hi) / 2;
        if ((arr_node[lo].data[0] > arr_node[hi].data[0]) || ((arr_node[lo].data[0] == arr_node[hi].data[0]) && (arr_node[lo].data[1] > arr_node[hi].data[1]))) { //(arr_node[lo].data[0] > arr_node[hi].data[0])
            swap(arr_node[lo].sample_id, arr_node[hi].sample_id);
            swap(arr_node[lo].data, arr_node[hi].data);
        }
        if ((arr_node[mid].data[0] > arr_node[hi].data[0]) || ((arr_node[mid].data[0] == arr_node[hi].data[0]) && (arr_node[mid].data[1] > arr_node[hi].data[1]))) { //(arr_node[mid].data[0] > arr_node[hi].data[0])
            swap(arr_node[mid].sample_id, arr_node[hi].sample_id);
            swap(arr_node[mid].data, arr_node[hi].data);
        }
        if ((arr_node[lo].data[0] > arr_node[mid].data[0]) || ((arr_node[lo].data[0] == arr_node[mid].data[0]) && (arr_node[lo].data[1] > arr_node[mid].data[1]))) { //(arr_node[lo].data[0] > arr_node[mid].data[0])
            swap(arr_node[lo].sample_id, arr_node[mid].sample_id);
            swap(arr_node[lo].data, arr_node[mid].data);
        }

        pivot_val_x = arr_node[mid].data[0];
        pivot_val_y = arr_node[mid].data[1];

        if (lo != mid) {
            swap(arr_node[lo].sample_id, arr_node[mid].sample_id);
            swap(arr_node[lo].data, arr_node[mid].data);
        }
        i = lo;
        j = hi;
        while (true) {
            i += 1;
            while ((arr_node[i].data[0] < pivot_val_x) || ((arr_node[i].data[0] == pivot_val_x) && (arr_node[i].data[1] < pivot_val_y))) { //(arr_node[i].data[axis] < pivot_val)
                i += 1;
            }

            j -= 1;
            while ((arr_node[j].data[0] > pivot_val_x) || ((arr_node[j].data[0] == pivot_val_x) && (arr_node[j].data[1] > pivot_val_y))) { //(arr_node[j].data[axis] > pivot_val)
                j -= 1;
            }

            if (i >= j) {
                if (lo != j) {
                    swap(arr_node[lo].sample_id, arr_node[j].sample_id);
                    swap(arr_node[lo].data, arr_node[j].data);
                }
                pivotIndex = j;
                return pivotIndex;
            }

            swap(arr_node[i].sample_id, arr_node[j].sample_id);
            swap(arr_node[i].data, arr_node[j].data);
        }
    }
    return -1;
}

void quicksort_2d(sort_node* arr_in, int lo, int hi) {
    int p;
    if ((lo >= 0) && (hi >= 0) && (lo < hi)) {
        p = partition_qsort(arr_in, lo, hi);
        quicksort_2d(arr_in, lo, p);
        quicksort_2d(arr_in, p + 1, hi);
    }
}

/*
function to sort the "data" and store the results in "arr_node"
The sorting is done only on x and y coordinates -> Sorting is done on x, and if x is equal, then on y
arr_node[i].data is the i'th entry in the sorted data
arr_node[i].sample_id is the sample_id of the i'th entry in the sorted data
*/
void sort_data(float* data, int ndim_s, int ndim_r, int* sample_id, int len, sort_node* arr_node) {
    for (int i = 0;i < len;++i) {
        arr_node[i].sample_id = &sample_id[i];
        arr_node[i].data = &data[(ndim_s + ndim_r) * i];
    }

    quicksort_2d(arr_node, 0, len - 1);
}

int sanity_check_sort(sort_node* arr_sorted, int dim_s, int dim_r, int len) {
    int is_sorted = 1;
    for (int i = 1; i < len;++i) {
        if (arr_sorted[i - 1].data[0] > arr_sorted[i].data[0]) {
            is_sorted = 0;
            break;
        }
        else if (arr_sorted[i - 1].data[0] == arr_sorted[i].data[0]) {
            if (arr_sorted[i - 1].data[1] > arr_sorted[i].data[1]) {
                is_sorted = 0;
                break;
            }
        }
    }
    return is_sorted;
}

/*
function to find the unique entries in data 
data -> input with "len" entries of dimension "dim_s + dim_r"
sample_id -> array storing sample_id of each entry in data
arr_node -> placeholder to hold the sorted data
unique -> output array to store the unique entries in data
unique_inv -> output array to store the mapping from unique to data
unique_counts -> output array to store the number of times each unique entry occurs in data
returns:
number of unique entries in data
*/
int find_unique(float* data, int dim_s, int dim_r, int* sample_id, int len, sort_node* arr_node, float* unique, int* unique_inv, int* unique_counts) {
    int j;

    sort_data(data, dim_s, dim_r, sample_id, len, arr_node);

    int current_id = 0;
    unique_inv[arr_node[0].sample_id[0]] = current_id;
    copy(arr_node[0].data, arr_node[0].data + (dim_s + dim_r), unique);
    unique_counts[current_id] += 1;
    for (int i = 1; i < len; ++i) {
        j = 0;
        while (arr_node[i].data[j] == arr_node[i - 1].data[j]) {
            j = j + 1;
            if (j == dim_s + dim_r) {
                break;
            }
        }

        if (j != dim_s + dim_r) {
            current_id = current_id + 1;
            copy(arr_node[i].data, arr_node[i].data + (dim_s + dim_r), unique + (dim_s + dim_r) * current_id);
        }
        unique_inv[arr_node[i].sample_id[0]] = current_id;
        unique_counts[current_id] += 1;
    }
    return (current_id + 1);
}

/*
function to map the unique entries to the original data
input:
unique -> array storing the unique entries
unique_inv -> array storing the mapping from unique to data
arr_in -> array to be reconstructed with "len" entries of dimension "dim_s + dim_r"
*/
void map_unique_inv(float* unique, int* unique_inv, float* arr_in, int dim_s, int dim_r, int len) {
    for (int i = 0; i < len;++i) {
        copy(unique + (dim_s + dim_r) * unique_inv[i], unique + (dim_s + dim_r) * unique_inv[i] + (dim_s + dim_r), arr_in + (dim_s + dim_r) * i);
    }
}

struct kdt_node
{
    kdt_node* parent;
    kdt_node* left;
    kdt_node* right;
    float* norm_r;
    int* sample_id;
    float* data;
    int* data_count;
};

void quickmedian(kdt_node* arr_node, int lo, int hi, int axis) {
    int k, mid, i, j, pivotIndex;
    float pivot_val;

    k = (lo + hi) / 2; //index of median in the array, arr[mid] -> median
    while (lo <= hi) {
        if (lo == hi) {
            return;
        }

        //Hoare Partitioning of the dataset
        mid = (lo + hi) / 2;
        if (arr_node[lo].data[axis] > arr_node[hi].data[axis]) { //(arr_s[ndim_s * lo + axis] > arr_s[ndim_s * hi + axis])B
            swap(arr_node[lo].norm_r, arr_node[hi].norm_r);
            swap(arr_node[lo].sample_id, arr_node[hi].sample_id);
            swap(arr_node[lo].data, arr_node[hi].data);
            swap(arr_node[lo].data_count, arr_node[hi].data_count);
        }
        if (arr_node[mid].data[axis] > arr_node[hi].data[axis]) { //(arr_s[ndim_s * mid + axis] > arr_s[ndim_s * hi + axis])
            swap(arr_node[mid].norm_r, arr_node[hi].norm_r);
            swap(arr_node[mid].sample_id, arr_node[hi].sample_id);
            swap(arr_node[mid].data, arr_node[hi].data);
            swap(arr_node[mid].data_count, arr_node[hi].data_count);
        }
        if (arr_node[lo].data[axis] > arr_node[mid].data[axis]) { //(arr_s[ndim_s * lo + axis] > arr_s[ndim_s * mid + axis])
            swap(arr_node[lo].norm_r, arr_node[mid].norm_r);
            swap(arr_node[lo].sample_id, arr_node[mid].sample_id);
            swap(arr_node[lo].data, arr_node[mid].data);
            swap(arr_node[lo].data_count, arr_node[mid].data_count);
        }

        pivot_val = arr_node[mid].data[axis]; //arr_s[ndim_s * mid + axis];

        if (lo != mid) {
            swap(arr_node[lo].norm_r, arr_node[mid].norm_r);
            swap(arr_node[lo].sample_id, arr_node[mid].sample_id);
            swap(arr_node[lo].data, arr_node[mid].data);
            swap(arr_node[lo].data_count, arr_node[mid].data_count);
        }
        i = lo;
        j = hi;
        while (true) {
            i += 1;
            while (arr_node[i].data[axis] < pivot_val) { //(arr_s[ndim_s * i + axis] < pivot_val)
                i += 1;
            }

            j -= 1;
            while (arr_node[j].data[axis] > pivot_val) { //(arr_s[ndim_s * j + axis] > pivot_val)
                j -= 1;
            }

            if (i >= j) {
                if (lo != j) {
                    swap(arr_node[lo].norm_r, arr_node[j].norm_r);
                    swap(arr_node[lo].sample_id, arr_node[j].sample_id);
                    swap(arr_node[lo].data, arr_node[j].data);
                    swap(arr_node[lo].data_count, arr_node[j].data_count);
                }
                pivotIndex = j;
                break;
            }

            swap(arr_node[i].norm_r, arr_node[j].norm_r);
            swap(arr_node[i].sample_id, arr_node[j].sample_id);
            swap(arr_node[i].data, arr_node[j].data);
            swap(arr_node[i].data_count, arr_node[j].data_count);
        }

        if (pivotIndex == k) {
            return;
        }
        else if (pivotIndex > k) {
            hi = pivotIndex - 1;
        }
        else {
            lo = pivotIndex + 1;
        }
    }
}

//"notinplace" implies that it DOES NOT change the data it works on
/*
	arr_node: array of kdt_node to store the tree. It points to the entries in the dataset
	ndim_s: number of spatial dimensions of the data
	len: length of the input data arr_node
	axis: axis to split on
	parent: parent node of the current node
returns:
    pointer to the root of the tree
*/
kdt_node* build_kdtree_notinplace(kdt_node* arr_node, int ndim_s, int len, int axis, kdt_node* parent) {
    int median_idx;

    if (len == 0) {
        return NULL;
    }
    else {
        quickmedian(arr_node, 0, len - 1, axis);

        median_idx = (len - 1) / 2;

        arr_node[median_idx].parent = parent;
        arr_node[median_idx].left = build_kdtree_notinplace(arr_node, ndim_s, median_idx, (axis + 1) % ndim_s, &arr_node[median_idx]);
        arr_node[median_idx].right = build_kdtree_notinplace(&arr_node[median_idx + 1], ndim_s, len - median_idx - 1, (axis + 1) % ndim_s, &arr_node[median_idx]);
    }
    return &arr_node[median_idx];
}

/*
input:
    data: array of data
	ndim_s: number of spatial dimensions of the data
	ndim_r: number of range dimensions of the data
	norm_r: array containing euclidean norm of the range dimensions
	sample_id: array of sample ids
	data_count: number of times each sample is repeated 
	len: length of the input data
	arr_node: array of kdt_node to store the tree. It points to the entries in the dataset
returns:
    pointer to the root of the tree
*/
kdt_node* build_kdtree(float* data, int ndim_s, int ndim_r, float* norm_r, int* sample_id, int* data_count, int len, kdt_node* arr_node) {
    for (int i = 0; i < len;++i) {
        arr_node[i].parent = NULL;
        arr_node[i].left = NULL;
        arr_node[i].right = NULL;
        arr_node[i].norm_r = &norm_r[i];
        arr_node[i].sample_id = &sample_id[i];
        arr_node[i].data = &data[(ndim_s + ndim_r) * i];
        arr_node[i].data_count = &data_count[i];
    }

    kdt_node* root = build_kdtree_notinplace(arr_node, ndim_s, len, 0, NULL);
    return root;
}

int search_tree(kdt_node* root, float* query, int axis, int dim, float* tree_arr) {
    if (root == NULL) {
        return 0;
    }
    else {
        for (int i = 0; i < dim;i++) {
            if (root->data[i] == query[i]) {
                return 1;
            }
        }
    }

    if (query[axis] <= root->data[axis]) {
        return search_tree(root->left, query, (axis + 1) % dim, dim, tree_arr);
    }
    else {
        return search_tree(root->right, query, (axis + 1) % dim, dim, tree_arr);
    }

}

/*
input:
    data_in: array of data
    N: number of samples
    dim: number of dimensions
    n_comp: number of components to keep
output:
	embedding: array of size N * n_comp to store the embedding
*/
void compute_PCA(float* data_in, int N, int dim, int n_comp, float* embedding) {
    Eigen::Map<Eigen::MatrixXf> data(data_in, dim, N);

    Eigen::MatrixXf X(dim, N);
    float feat_mean;
    for (int i = 0; i < dim;++i) {
        feat_mean = data(i, Eigen::all).mean();
        for (int j = 0; j < N; ++j) {
            X(i, j) = data(i, j) - feat_mean;
        }
    }

    Eigen::MatrixXf XXT = X * X.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(XXT);
    Eigen::MatrixXf comp = es.eigenvectors()(Eigen::all, Eigen::lastN(N).reverse())(Eigen::all, Eigen::seq(0, n_comp - 1));

    Eigen::MatrixXf X_proj = comp.transpose() * X;

    float* out = X_proj.data();
    for (int i = 0; i < N * n_comp;++i) {
        embedding[i] = out[i];
    }
}

/*
function to insert a new element "dist" in the max heap represented by heap_dist
the corresponding 'gene' goes in the same place in heap_genes
len -> no of elements in heap_dist / heap_genes
*/
void max_heap_insert(int len, float* heap_dist, int* heap_genes, float dist, float gene) {
    heap_dist[len] = dist;
    heap_genes[len] = gene;
    int i = len;
    int ind_parent;
    while (i > 0) {
        ind_parent = int((i - 1) / 2);
        if (heap_dist[i] > heap_dist[ind_parent]) {
            swap(heap_dist[i], heap_dist[ind_parent]);
            swap(heap_genes[i], heap_genes[ind_parent]);
            i = ind_parent;
        }
        else {
            break;
        }
    }
}

/*
function to replace the root of the max heap represented by heap_dist with new element "dist"
*/
void max_heap_replace(int len, float* heap_dist, int* heap_genes, float dist, float gene) {
    heap_dist[0] = dist;
    heap_genes[0] = gene;
    int i = 0;
    int child;
    while ((2 * i + 1) < len) {
        if ((2 * i + 1 == len - 1) || (heap_dist[2 * i + 1] > heap_dist[2 * i + 2])) {
            child = 2 * i + 1;
        }
        else {
            child = 2 * i + 2;
        }

        if (heap_dist[i] < heap_dist[child]) {
            swap(heap_dist[i], heap_dist[child]);
            swap(heap_genes[i], heap_genes[child]);
            i = child;
        }
        else {
            break;
        }
    }
}

/*
function that maintains a max heap of size k for distances
if the heap is not full, it inserts the new element in the queue
if the heap is full and heap_dist[0] > dist, it replaces the root of the heap with the new element
*/
void dist_queue_insert(int k, int* len, float* heap_dist, int* heap_genes, float dist, float gene) {
    if (len[0] < k) {
        max_heap_insert(len[0], heap_dist, heap_genes, dist, gene);
        len[0] += 1;
    }
    else {
        if (dist < heap_dist[0]) {
            max_heap_replace(len[0], heap_dist, heap_genes, dist, gene);
        }
    }
}

/*
function to find the k nearest neighbours of a query point in a kd tree
input:
    root -> root of the kd tree
    query_data -> array of size N_query x dim_s containing the query points
    dim_s -> number of spatial dimensions
    k -> number of nearest neighbours to find
    n -> the sample id of the query point for which to find the k nearest neighbours
output:
    heap_dist -> max heap (1d array) of size N_query x k to store the distances
    heap_genes -> max heap (1d array) of size N_query x k to store the corresponding genes
    arr_len -> 1d array of size N_query to store the number of elements in the max heap for each query point
*/
void find_neighbors(kdt_node* root, float* query_data, int dim_s, int n, int k, float* heap_dist, int* heap_genes, int* arr_len) {
    kdt_node* c_node;
    int c_axis;
    int action;
    int sample_coord;
    float min_dist;

    float dist2_s;
    float tmp_a;

    //start of iter
    sample_coord = (dim_s + 1) * n;

    c_node = root;
    c_axis = 0;

    action = 0;
    min_dist = numeric_limits<float>::infinity();

    while (c_node) {
        if (action == 0) {
            /*if the query is in left branch and left branch exists, go left
            if the query is in left branch and left branch does not exist, set action=2 to see if you need to check right branch
            if opposite happens, set action=1 to go right*/
            if (c_node->data[c_axis] >= query_data[sample_coord + c_axis]) {
                if (c_node->left) {
                    c_node = c_node->left;
                    c_axis = (c_axis + 1) % dim_s;
                }
                else {
                    action = 2;
                }
            }
            else {
                action = 1;
            }
        }
        else if (action == 1) {
            /*If the query is in right branch and right branch exists - go right, and set action=0 to go left again
            If the query is in right branch and right branch does not exist, set action=2 to see if you need to check left branch
            Else - set action = 2*/
            if (c_node->data[c_axis] <= query_data[sample_coord + c_axis]) {
                if (c_node->right) {
                    c_node = c_node->right;
                    c_axis = (c_axis + 1) % dim_s;
                    action = 0;
                }
                else {
                    action = 2;
                }
            }
            else {
                action = 3;
            }
        }
        else if (action == 2) {
            /*if the query belongs to the left branch*/
            if (c_node->data[c_axis] >= query_data[sample_coord + c_axis]) {
                //Check if the current node is closer than the current best to decide whether to check right branch
                if (((query_data[sample_coord + c_axis] - c_node->data[c_axis]) * (query_data[sample_coord + c_axis] - c_node->data[c_axis]) < min_dist) && (c_node->right != NULL)) {
                    c_node = c_node->right;
                    c_axis = (c_axis + 1) % dim_s;
                    action = 0;
                }
                else {
                    action = 3;
                }
            }
            /*If the query belongs to the right branch*/
            else {
                //Check if the current node is closer than the current best to decide whether to check left branch
                if (((c_node->data[c_axis] - query_data[sample_coord + c_axis]) * (c_node->data[c_axis] - query_data[sample_coord + c_axis]) < min_dist) && (c_node->left != NULL)) {
                    c_node = c_node->left;
                    c_axis = (c_axis + 1) % dim_s;
                    action = 0;
                }
                else {
                    action = 3;
                }
            }
        }
        else {
            c_axis = (c_axis + dim_s - 1) % dim_s;

            // check if the current node is closer than the current min_dist
            dist2_s = 0;
            for (int j = 0; j < dim_s; ++j) {
                tmp_a = (query_data[sample_coord + j] - c_node->data[j]);
                dist2_s += tmp_a * tmp_a;
            }

            dist_queue_insert(k, &arr_len[n], &heap_dist[n * k], &heap_genes[n * k], dist2_s, c_node->data[dim_s]);
            min_dist = heap_dist[n * k];
            if (arr_len[n] < k) {
                min_dist = numeric_limits<float>::infinity();
            }

            //If the current node has no parent, terminate the loop
            if (c_node->parent == NULL) {
                c_node = NULL;
            }
            else {
                //if the parent is closer than current min_d, see whether you need to check the opposite branch
                if ((query_data[sample_coord + c_axis] - c_node->parent->data[c_axis]) * (query_data[sample_coord + c_axis] - c_node->parent->data[c_axis]) < min_dist) {
                    //if the current node was the left child and the query belongs in the left branch of the parent, go right
                    if ((c_node->parent->left == c_node) && (c_node->parent->data[c_axis] >= query_data[sample_coord + c_axis]) && (c_node->parent->right != NULL)) {
                        c_node = c_node->parent->right;
                        c_axis = (c_axis + 1) % dim_s;
                        action = 0;
                    }
                    //if the current node was the right child and the query belongs in the right branch of the parent, go left
                    else if ((c_node->parent->right == c_node) && (c_node->parent->data[c_axis] < query_data[sample_coord + c_axis]) && (c_node->parent->left != NULL)) {
                        c_node = c_node->parent->left;
                        c_axis = (c_axis + 1) % dim_s;
                        action = 0;
                    }
                    else {
                        c_node = c_node->parent;
                        action = 3;
                    }
                }
                else {
                    c_node = c_node->parent;
                    action = 3;
                }
            }
        }
    }
}

/*
function that finds the spatial k nearest neighbors and aggregates the corresponding genes
if n_genes > 50, PCA applied to reduce dimensions
N - number of data points
dim_s - number of spatial dimensions
data - array that stores the data points, size - N x (dim_s + 1), the last entry is the gene label
n_genes - number of unique genes
k -> number of nearest neighbors to find
output:
    counts_out - array that stores the aggregated gene counts, size - N x max(50, n_genes)
*/
void knn_smoothing(float* data, int dim_s, int N, int n_genes, int k, float* counts_out) {
    kdt_node* arr_node = new kdt_node[N]; //array that stores tree nodes
    float* norm_r = new float[N]; //array that stores norm of tree nodes
    int* sample_id = new int[N];
    int* data_count = new int[N]();
    float* query_data = new float[(dim_s + 1) * N];

    float* heap_dist = new float[N * k];
    int* heap_genes = new int[N * k];
    int* arr_len = new int[N]();

    float* counts = new float[N * n_genes]();

    euclidean_norm(data, N, dim_s, 1, norm_r);
    get_sample_id(N, sample_id);

    auto start = high_resolution_clock::now();
    kdt_node* root = build_kdtree(data, dim_s, 1, norm_r, sample_id, data_count, N, arr_node);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    copy(data, data + (dim_s + 1) * N, query_data);

    #pragma omp parallel for schedule(static, 1000) num_threads(8) 
    for (int n = 0; n < N; ++n) {
        find_neighbors(root, query_data, dim_s, n, k, heap_dist, heap_genes, arr_len);
    }

    int glabel;
    for (int n = 0; n < N; ++n) {
        for (int i = 0; i < k; ++i) {
            glabel = heap_genes[n * k + i];
            counts[n * n_genes + glabel] += 1;
        }
    }

    if (n_genes > 50) {
        compute_PCA(counts, N, n_genes, 50, counts_out);
    }
    else {
        compute_PCA(counts, N, n_genes, n_genes, counts_out);
        //copy(counts, counts + N * n_genes, counts_out);
    }

    delete[] arr_node;
    delete[] norm_r;
    delete[] sample_id;
    delete[] data_count;
    delete[] query_data;
    delete[] heap_dist;
    delete[] heap_genes;
    delete[] arr_len;
    delete[] counts;
}

float* preprocess_data(float* coords, float* genes, int N, int dim_s, int n_genes, int k) {
    float* data = create_dataset(coords, genes, dim_s, 1, N);

    int dim_r = 50;
    if (n_genes < 50) {
        dim_r = n_genes;
    }
    float* counts_out = new float[N * dim_r];

    knn_smoothing(data, dim_s, N, n_genes, k, counts_out);

    delete[] data;

    data = create_dataset(coords, counts_out, dim_s, dim_r, N);

    delete[] counts_out;
    return data;
}

/*
function that does a single iteration of the mean shift algorithm for a single data point
inputs:
    root: root of the kdtree
    query_data: array that stores the data point
    dim_s: number of spatial dimensions
    dim_r: number of gene dimensions
    norm_qr: array that stores the norm of the range part of query points
    h_s: spatial bandwidth
    h_r: range bandwidth
    kernel_s: kernel type for spatial dimensions; allowable values -> 0 (uniform/disk), 1 (epanechnikov)
    kernel_r: kernel type for range dimensions; allowable values -> 0 (uniform/disk), 1 (epanechnikov)
    n: the query point for which to do the iteration
    min_s_range: array that stores the minimum value of the allowed range for each spatial dimension
    max_s_range: array that stores the maximum value of the allowed range for each spatial dimension
    flows: array that stores the flow vectors
    height: height of the flow image
    width: width of the flow image
    use_flows: flag to indicate whether to use flow vectors or not
    alpha: weight given to the flow vector; allowable values between 0 and 1
outputs:
    modes: array that stores the modes of the query point
    weight_sum: placeholder array that stores the sum of weights of the query point
*/
void ms_spacerange_iter(kdt_node* root, float* query_data, int dim_s, int dim_r, float* norm_qr, float h_s, float h_r, int kernel_s, int kernel_r, float* modes, float* weight_sum, int n, float* min_s_range, float* max_s_range, float* flows, int height, int width, int use_flows, float alpha) {

    kdt_node* c_node;
    int c_axis;
    int action;
    float dist2_s;
    float num, den, dist_r;
    float w_s, w_r, w;
    int sample_coord;
    float hs2 = h_s * h_s;
    float tmp_a;

    int xval, yval;
    float flow_x, flow_y;

    //start of iter
    sample_coord = (dim_s + dim_r) * n;

    c_node = root;
    c_axis = 0;

    action = 0;


    //Take an action, which defines your next action
    while (c_node) {
        if (action == 0) {
            /*Search the left branch until the left branch goes out of range OR you reach Null
            if it happens, go to action-1*/
            if ((c_node->data[c_axis] > min_s_range[dim_s * n + c_axis]) && (c_node->left != NULL)) {

                c_node = c_node->left;
                c_axis = (c_axis + 1) % dim_s;
            }
            else {
                action = 1;
            }
        }
        else if (action == 1) {
            //Check if current node is within range
            dist2_s = 0;
            for (int j = 0; j < dim_s; ++j) {
                tmp_a = (query_data[sample_coord + j] - c_node->data[j]);
                dist2_s += tmp_a * tmp_a;
            }
            
            if (dist2_s <= hs2) {
                Eigen::Map<Eigen::VectorXf> v1(&query_data[sample_coord + dim_s], dim_r);
                Eigen::Map<Eigen::VectorXf> v2(&c_node->data[dim_s], dim_r);

                num = v1.dot(v2);

                den = norm_qr[n] * c_node->norm_r[0];
                dist_r = 1 - num / den;

                if (dist_r <= h_r) {
                    //calculations += 1;
                    if (kernel_s == 0) {
                        w_s = 1;
                    }
                    else { //epanechnikov
                        w_s = hs2 - dist2_s;
                    }

                    if (kernel_r == 0) {
                        w_r = 1;
                    }
                    else {
                        w_r = h_r * h_r - dist_r * dist_r;
                    }

                    w = w_s * w_r;

                    for (int j = 0; j < (dim_s + dim_r); ++j) {
                        modes[sample_coord + j] += w * c_node->data_count[0] * c_node->data[j];
                    }
                    weight_sum[n] += w * c_node->data_count[0];
                }
            }

            /*If the right branch exists and is within range - go right, and set action = 0 to go left again
            Else - set action = 2*/
            if ((c_node->data[c_axis] < max_s_range[dim_s * n + c_axis]) && (c_node->right != NULL)) {
                c_node = c_node->right;
                c_axis = (c_axis + 1) % dim_s;
                action = 0;
            }
            else {
                action = 2;
            }
        }
        else {
            c_axis = (c_axis + dim_s - 1) % dim_s;

            //If the current node has no parent, terminate the loop
            if (c_node->parent == NULL) {
                c_node = NULL;
            }
            else {
                //if the current node was the left child, set action=1 to check the parent and right branch
                if (c_node->parent->left == c_node) {
                    action = 1;
                }
                else { //else action = 2, to travel further up because the parents of right children have already been checked
                    action = 2;
                }
                c_node = c_node->parent;
            }
        }
    }

    for (int j = 0; j < (dim_s + dim_r);++j) {
        modes[sample_coord + j] = modes[sample_coord + j] / weight_sum[n];
    }

    if (use_flows == 1) {
        xval = (int)query_data[sample_coord + 0];
        yval = (int)query_data[sample_coord + 1];

        flow_x = flows[height * width * xval + height * yval + 1];
        flow_y = flows[height * width * xval + height * yval + 0];

        if (flow_x * flow_x + flow_y * flow_y > 0.01) {
            modes[sample_coord + 0] = query_data[sample_coord + 0] + alpha * (modes[sample_coord + 0] - query_data[sample_coord + 0]) + (1 - alpha) * flow_x;
            modes[sample_coord + 1] = query_data[sample_coord + 1] + alpha * (modes[sample_coord + 1] - query_data[sample_coord + 1]) + (1 - alpha) * flow_y;
        }
    }
    return;
}

float* meanshift_spacerange(float* data, int dim_s, int dim_r, int len, int max_iter, float h_s, float h_r, int kernel_s, int kernel_r, int blurring, float* flows, int height, int width, int use_flows, float alpha) {
    kdt_node* arr_node = new kdt_node[len]; //array that stores tree nodes
    float* norm_r = new float[len]; //array that stores norm of tree nodes
    float* norm_qr = new float[len]; //array that stores norm of query nodes
    int* sample_id = new int[len];
    float* query_data = new float[(dim_s + dim_r) * len];
    float* modes = new float[(dim_s + dim_r) * len];
    float* wsum = new float[len];

    float* min_s_range = new float[dim_s * len];
    float* max_s_range = new float[dim_s * len];

    sort_node* arr_sorted = new sort_node[len];
    float* unique_query_data = new float[(dim_s + dim_r) * len];
    int* unique_inv = new int[len];
    int* unique_counts = new int[len]();
    int len_unique;

    euclidean_norm(data, len, dim_s, dim_r, norm_r);
    get_sample_id(len, sample_id);

    len_unique = find_unique(data, dim_s, dim_r, sample_id, len, arr_sorted, unique_query_data, unique_inv, unique_counts);
    euclidean_norm(unique_query_data, len_unique, dim_s, dim_r, norm_r);
    get_sample_id(len_unique, sample_id);
    kdt_node* root = build_kdtree(unique_query_data, dim_s, dim_r, norm_r, sample_id, unique_counts, len_unique, arr_node);

    for (int iter = 0; iter < max_iter; ++iter) {
        //if (PyErr_CheckSignals() != 0) //needed for checking ctrl+c
        //    throw py::error_already_set(); //needed for checking ctrl+c
        auto start = high_resolution_clock::now();
        for (int i = 0; i < len_unique; ++i) {
            for (int j = 0; j < dim_s;++j) {
                min_s_range[dim_s * i + j] = unique_query_data[(dim_s + dim_r) * i + j] - h_s;
                max_s_range[dim_s * i + j] = unique_query_data[(dim_s + dim_r) * i + j] + h_s;
            }
        }
        for (int i = 0; i < len;++i) {
            wsum[i] = 0;
        }
        for (int i = 0; i < (dim_s + dim_r) * len; ++i) {
            modes[i] = 0;
        }
        euclidean_norm(unique_query_data, len_unique, dim_s, dim_r, norm_qr);

        #pragma omp parallel for schedule(static, 1000) num_threads(8) 
        for (int n = 0; n < len_unique; ++n) {
            ms_spacerange_iter(root, unique_query_data, dim_s, dim_r, norm_qr, h_s, h_r, kernel_s, kernel_r, modes, wsum, n, min_s_range, max_s_range, flows, height, width, use_flows, alpha);
        }
        map_unique_inv(modes, unique_inv, query_data, dim_s, dim_r, len);
        
        if (blurring) {
            if (iter != max_iter - 1) {
                len_unique = find_unique(query_data, dim_s, dim_r, sample_id, len, arr_sorted, unique_query_data, unique_inv, unique_counts);
                euclidean_norm(unique_query_data, len_unique, dim_s, dim_r, norm_r);
                get_sample_id(len_unique, sample_id);
                root = build_kdtree(unique_query_data, dim_s, dim_r, norm_r, sample_id, unique_counts, len_unique, arr_node);
            }
        }

        if (iter == max_iter - 1) {
            copy(query_data, query_data + (dim_s + dim_r) * len, modes);
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "iter - " << iter << " completed, time taken - " << duration.count() / 1e6 << " seconds" << endl;
    }

    delete[] arr_node;
    delete[] norm_r;
    delete[] norm_qr;
    delete[] sample_id;
    delete[] query_data;
    delete[] wsum;
    delete[] min_s_range;
    delete[] max_s_range;
    delete[] arr_sorted;
    delete[] unique_query_data;
    delete[] unique_inv;
    delete[] unique_counts;

    return modes;
}

float* meanshift(float* coords, float* genes, int N, int dim_s, int n_genes, int k, int max_iter, float h_s, float h_r, int kernel_s, int kernel_r, int blurring, float* flows, int height, int width, int use_flows, float alpha) {
    auto start = high_resolution_clock::now();
    float* data = preprocess_data(coords, genes, N, dim_s, n_genes, k);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Preprocessing done: " << duration.count() / 1e6 << " seconds" << endl;

    int dim_r = 50;
    if (n_genes < 50) {
        dim_r = n_genes;
    }

    float* modes = meanshift_spacerange(data, dim_s, dim_r, N, max_iter, h_s, h_r, kernel_s, kernel_r, blurring, flows, height, width, use_flows, alpha);
    modes = meanshift_spacerange(modes, dim_s, dim_r, N, 1, h_s / 4, 4, 0, 0, 1, flows, 1, 1, 0, 1);
    return modes;
}

float* density_estimate(float* coords, int* seg, int N, float h) {
    float* density = new float[N]();

    kdt_node* arr_node = new kdt_node[N]; //array that stores tree nodes
    float* norm_r = new float[N]; //array that stores norm of tree nodes
    int* sample_id = new int[N];
    int* counts = new int[N]();
    
    euclidean_norm(coords, N, 2, 0, norm_r);
    get_sample_id(N, sample_id);

    kdt_node* root = build_kdtree(coords, 2, 0, norm_r, sample_id, counts, N, arr_node);

    kdt_node* c_node;
    int c_axis;
    int action;
    float dist2_s;
    float w_s;
    int sample_coord;
    float hs2 = h * h;
    float tmp_a;

    float* query_data = coords;

    //start of iter
    for (int n = 0; n < N; n++) {

        sample_coord = 2 * n;

        c_node = root;
        c_axis = 0;

        action = 0;


        //Take an action, which defines your next action
        while (c_node) {
            if (action == 0) {
                /*Search the left branch until the left branch goes out of range OR you reach Null
                if it happens, go to action-1*/
                if ((c_node->data[c_axis] > (query_data[2 * n + c_axis] - h)) && (c_node->left != NULL)) {

                    c_node = c_node->left;
                    c_axis = (c_axis + 1) % 2;
                }
                else {
                    action = 1;
                }
            }
            else if (action == 1) {
                //Check if current node is within range
                dist2_s = 0;
                for (int j = 0; j < 2; ++j) {
                    tmp_a = (query_data[sample_coord + j] - c_node->data[j]);
                    dist2_s += tmp_a * tmp_a;
                }

                if (dist2_s <= hs2) {
                    if (seg[c_node->sample_id[0]] == seg[n]) {
                        w_s = hs2 - dist2_s;

                        density[n] += w_s;
                    }
                }

                /*If the right branch exists and is within range - go right, and set action = 0 to go left again
                Else - set action = 2*/
                if ((c_node->data[c_axis] < (query_data[2 * n + c_axis] + h)) && (c_node->right != NULL)) {
                    c_node = c_node->right;
                    c_axis = (c_axis + 1) % 2;
                    action = 0;
                }
                else {
                    action = 2;
                }
            }
            else {
                c_axis = (c_axis + 2 - 1) % 2;

                //If the current node has no parent, terminate the loop
                if (c_node->parent == NULL) {
                    c_node = NULL;
                }
                else {
                    //if the current node was the left child, set action=1 to check the parent and right branch
                    if (c_node->parent->left == c_node) {
                        action = 1;
                    }
                    else { //else action = 2, to travel further up because the parents of right children have already been checked
                        action = 2;
                    }
                    c_node = c_node->parent;
                }
            }
        }
    }

    delete[] arr_node;
    delete[] norm_r;
    delete[] sample_id;
    delete[] counts;

    
    return density;
}

int main()
{
}
