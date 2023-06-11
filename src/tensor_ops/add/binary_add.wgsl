// __TYPENAME__ is set via string replacement when op is registered
alias T = __TYPENAME__;
alias usize = u32;

struct BinaryKernelMeta {
    numel: usize,
    num_dims: usize,
    info: array<usize>
}

@group(0) @binding(0)
var<storage, read> lhs: array<T>;

@group(0) @binding(1)
var<storage, read> rhs: array<T>;

@group(0) @binding(2)
var<storage, read> kernel_meta: BinaryKernelMeta;

@group(0) @binding(3)
var<storage, read_write> output: array<T>;

@compute @workgroup_size(1, 1, 1)
fn badd_fwd___TYPENAME__(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let i = global_id.x;
    output[i] = lhs[i] + rhs[i];
}
