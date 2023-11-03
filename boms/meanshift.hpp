float* meanshift(float* coords, float* genes, int N, int dim_s, int n_genes, int k, int max_iter, float h_s, float h_r, int kernel_s, int kernel_r, int blurring, float* flows, int height, int width, int use_flows, float alpha);

float* preprocess_data(float* coords, float* genes, int N, int dim_s, int n_genes, int k);

float* density_estimate(float* coords, int* seg, int N, float h);