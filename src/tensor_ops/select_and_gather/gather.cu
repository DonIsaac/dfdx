__device__ unsigned int get_strided_index(
    unsigned int idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int strided_i = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

__device__ unsigned int get_gathered_index(
    const unsigned int index,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    const size_t *idx,
    const size_t idx_num_dims,
    const size_t *idx_dims,
    const size_t *idx_strides,
    const size_t out_num_dims
) {
    unsigned int ax;

    if (out_num_dims > inp_num_dims) {
        ax = 0;
    } else {
        ax = idx_num_dims - 1;
    }

    unsigned int elem_size = 1; // the size of each indexed element
    unsigned int row_len = inp_dims[ax]; // the size of the indexed dimension

    for (unsigned int d = 0; d < inp_num_dims - ax - 1; d++) {
        unsigned int dim_idx = inp_num_dims - 1 - d;
        elem_size *= inp_dims[dim_idx];
    }

    // location to find the index for the replaced dimension in "idx"
    unsigned int idx_idx = get_strided_index(index / elem_size, idx_num_dims, idx_dims, idx_strides);

    // indices for dimensions before, at, and after the indexed dimension
    unsigned int idx_before = index / (elem_size * row_len);
    unsigned int idx_mid = idx[idx_idx];
    unsigned int idx_after = index % elem_size;

    // recombine
    unsigned int new_idx = (idx_before * row_len + idx_mid) * elem_size + idx_after;
    return get_strided_index(new_idx, inp_num_dims, inp_dims, inp_strides);
}

extern "C" __global__ void gather_forward(
    const size_t numel,
    const float *inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    const size_t *idx,
    const size_t idx_num_dims,
    const size_t *idx_dims,
    const size_t *idx_strides,
    float *out,
    const size_t out_num_dims
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    unsigned int out_i = i;
    unsigned int inp_i =
        get_gathered_index(i, inp_num_dims, inp_dims, inp_strides, idx, idx_num_dims, idx_dims, idx_strides, out_num_dims);

    out[out_i] = inp[inp_i];
}

extern "C" __global__ void gather_backward(
    const size_t numel,
    float *grad_inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    const size_t *idx,
    const size_t idx_num_dims,
    const size_t *idx_dims,
    const size_t *idx_strides,
    const float *grad_out,
    const size_t out_num_dims
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    unsigned int out_i = i;
    unsigned int inp_i =
        get_gathered_index(i, inp_num_dims, inp_dims, inp_strides, idx, idx_num_dims, idx_dims, idx_strides, out_num_dims);

    atomicAdd(grad_inp + inp_i, grad_out[out_i]);
}