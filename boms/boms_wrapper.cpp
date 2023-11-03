//wrapper for cpp functions

#include <tuple>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "meanshift.hpp"

using namespace std;

namespace py = pybind11;

tuple<float*, int, int> convert_to_c(py::array arr_in) {
    py::buffer_info buf = arr_in.request();
    float* ptr = (float*)buf.ptr;

    int N = buf.shape[0];
    int dim = buf.shape[1];

    float* data = new float[N * dim];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < dim; ++j) {
            data[dim * i + j] = ptr[dim * i + j];
        }
    }
    return { data, N, dim };
}

tuple<float*, int, int, int> convert_to_c_3d(py::array arr_in) {
    py::buffer_info buf = arr_in.request();
    float* ptr = (float*)buf.ptr;

    int dim1 = buf.shape[0];
    int dim2 = buf.shape[1];
    int dim3 = buf.shape[2];

    float* data = new float[dim1 * dim2 * dim3];
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            for (int k = 0; k < dim3; ++k) {
                data[dim1 * dim2 * k + dim1 * j + i] = ptr[dim2 * dim3 * i + dim3 * j + k];
            }
        }
    }
    return { data, dim1, dim2, dim3 };
}

py::array convert_to_py(float* data, int N, int dim) {
    auto result = py::array_t<float>(N * dim);
    py::buffer_info buf_result = result.request();

    float* result_ptr = (float*)buf_result.ptr;

    for (int i = 0; i < N * dim; ++i) {
        result_ptr[i] = data[i];
    }
    return result;
}

py::array meanshift_cpp(py::array coords_np, py::array genes_np, int n_genes, int k, int max_iter, float h_s, float h_r, int use_flows, py::array flows_np, float alpha) {
    float* coords;
    float* genes;
    int N, dim_s, dim_r;

    float* flows;
    int height, width, depth;

    tie(coords, N, dim_s) = convert_to_c(coords_np);
    tie(genes, N, dim_r) = convert_to_c(genes_np);
    tie(flows, height, width, depth) = convert_to_c_3d(flows_np);

    dim_r = 50;
    if (n_genes < 50) {
        dim_r = n_genes;
    }

    //float* meanshift(float* coords, float* genes, int N, int dim_s, int n_genes, int k, int max_iter, float h_s, float h_r, int kernel_s, int kernel_r, int blurring);
    float* modes = meanshift(coords, genes, N, dim_s, n_genes, k, max_iter, h_s, h_r, 1, 0, 1, flows, height, width, use_flows, alpha);

    py::array result = convert_to_py(modes, N, dim_s + dim_r);

    return result;
}

py::array smooth_ge_cpp(py::array coords_np, py::array genes_np, int n_genes, int k) {
    float* coords;
    float* genes;
    int N, dim_s, dim_r;

    tie(coords, N, dim_s) = convert_to_c(coords_np);
    tie(genes, N, dim_r) = convert_to_c(genes_np);

    dim_r = 50;
    if (n_genes < 50) {
        dim_r = n_genes;
    }

    float* data = preprocess_data(coords, genes, N, dim_s, n_genes, k);

    py::array result = convert_to_py(data, N, dim_s + dim_r);

    return result;
}

py::array density_estimator_cpp(py::array coords_np, py::array seg_np, float h) {

    py::buffer_info buf = seg_np.request();
    int* ptr = (int*)buf.ptr;

    int N = buf.shape[0];

    int* seg = new int[N];
    for (int i = 0; i < N; ++i) {
        seg[i] = ptr[i];
    }

    float* coords;
    int dim_s;

    tie(coords, N, dim_s) = convert_to_c(coords_np);

    float* density = density_estimate(coords, seg, N, h);

    py::array result = convert_to_py(density, N, 1);

    return result;
}

PYBIND11_MODULE(boms_wrapper, m) {
    m.doc() = "BOMS implementation in cpp"; // Optional module docstring
    m.def("meanshift_cpp", &meanshift_cpp);
    m.def("density_estimator_cpp", &density_estimator_cpp);
    m.def("smooth_ge_cpp", &smooth_ge_cpp);
}
