#ifndef SPMV_MUL_KERNELS
#define SPMV_MUL_KERNELS

__global__ void
replicate0(int tot_size, char* flags_d) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tot_size){
        flags_d[idx] = 0;
    }

}
//tot size int
//mat shp sc_d  for [3 2 4] exclusive scan [0 3 5]
__global__ void
mkFlags(int mat_rows, int* mat_shp_sc_d, char* flags_d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < mat_rows) {
        int tempval0;
        if (idx == 0){
            tempval0 = 0;
        }
        else {
            tempval0 = mat_shp_sc_d[idx-1];
        }
        if (tempval0 >= 0){
            flags_d[tempval0] = 1;
        }
    }
    //flags = false array, matshape = scan exclusive
    // TODO: fill in your implementation here ...
}

// vctsize = 2076
//block_size = 256
// matrix row num = 11033
__global__ void
mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    // TODO: fill in your implementation here ...
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tot_size) {
        tmp_pairs[idx] = mat_vals[idx] * vct[mat_inds[idx]];
    }
    
}

__global__ void
select_last_in_sgm(int mat_rows, int* mat_shp_sc_d, float* tmp_scan, float* res_vct_d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //0 indexed
    if (idx < mat_rows){    
        int lastindex = mat_shp_sc_d[idx]-1;
        if (lastindex >= 0){
            res_vct_d[idx] = tmp_scan[lastindex];
        }
    }
    

        
    // TODO: fill in your implementation here ...
}

#endif // SPMV_MUL_KERNELS
