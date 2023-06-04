alias usize = u32;

struct BinaryKernelMeta {
    numel: usize,
    num_dims: usize,
    info: array<usize>
}

@group(0) @binding(0)
var<storage, read> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

@group(0) @binding(2)
var<storage, read> kernel_meta: BinaryKernelMeta;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn badd_fwd_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    // let dims = &kernel_meta.info;
    // let lhs_strides = &kernel_meta.info[kernel_meta.num_dims];
    // let rhs_strides = &kernel_meta.info[2 * kernel_meta.num_dims];

    // for (var i: usize = global_id.x; i < kernel_meta.numel; i += 1u) {
    //     var tmp_i: usize = i;
    //     var lhs_i: usize = 0u;
    //     var rhs_i: usize = 0u;
    //     for (var d: usize = kernel_meta.num_dims - 1u; d >= 0u; d -= 1u) {
    //         // var i_dim: usize = tmp_i % dims[d];
    //         // lhs_i += i_dim * lhs_strides[d];
    //         // rhs_i += i_dim * rhs_strides[d];
    //         // tmp_i /= dims[d];
    //         var i_dim: usize = tmp_i % kernel_meta.info[d];
    //         lhs_i += i_dim * kernel_meta.info[kernel_meta.num_dims + d];
    //         rhs_i += i_dim * kernel_meta.info[2u * kernel_meta.num_dims + d];
    //     }
    //     // var x: f32 = lhs ? lhs[lhs_i] : out[i];
    //     // var y: f32 = rhs ? rhs[rhs_i] : out[i];
    //     // select is (falseBranch, trueBranch, cond)
    //     // var x: f32 = select(out[i], lhs[lhs_i], lhs);
    //     // var y: f32 = select(out[i], rhs[rhs_i], rhs);
    //     var x: f32 = lhs[lhs_i];
    //     var y: f32 = rhs[rhs_i];
    //     output[i] = x + y;
    // }
    let i = global_id.x;
    output[i] = lhs[i] + rhs[i];
}
