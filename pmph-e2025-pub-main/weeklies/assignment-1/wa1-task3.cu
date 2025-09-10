#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "helper.h"

#define GPU_RUNS 300

__global__ void addVec(float* X, float* Y, float* C, unsigned int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        C[i] = X[i] + Y[i];
    }
}

int main(int argc, char** argv) {
    unsigned int N;
    
    { // reading the number of elements 
      /* if (argc != 2) { 
        printf("Num Args is: %d instead of 1. Exiting!\n", argc); 
        exit(1);
      } */
    
      N = 753411; // 1190 gb/sec
      //N = 32768; 2^1562  //62.66 gb/sec
      //N = 524288; //2^19 878 gb/sec
      //N = 524288*2; //2^20 1400 gb/sec
      //N = 524288*2*2; //2^21 1706 gb/sec 
      //N = 524288*2*2*2; //2^22 816 gb/sec
      //N = atoi(argv[1]);
      printf("N is: %d\n", N);

      const unsigned int maxN = 500000000;
      if(N > maxN) {
          printf("N is too big; maximal value is %d. Exiting!\n", maxN);
          exit(2);
      }
    }

    // use the first CUDA device:
    cudaSetDevice(0);
    unsigned int mem_size = N*sizeof(float);
    // initialize the memory
    float* h_x = (float*)malloc(mem_size);
    float* h_y = (float*)malloc(mem_size);
    float* h_out1 = (float*)malloc(mem_size);
    float* h_out2 = (float*)malloc(mem_size);
    for (unsigned int i = 0; i < N; i++) {
            h_x[i] = (float)52145.42;
            h_y[i] = (float)125.4267;
            h_out1[i] = (float)0.0;
            h_out2[i] = (float)0.0;
        }
    
    //sequential
    double elapsed; struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);
    for (int i = 0; i < N; i++) {
        h_out2[i] = h_x[i] + h_y[i];
    }
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (1.0 * (t_diff.tv_sec*1e6+t_diff.tv_usec));
    double gigabytespersecsequential = (2.0 * N * 4.0) / (elapsed * 1000.0);
    printf("The sequential took %f microseconds. GB/sec: %f \n", elapsed, gigabytespersecsequential);
        
    // allocate device memory
    float* d_x;
    float* d_y;
    float* d_out;
    cudaMalloc((void**)&d_x,  mem_size);
    cudaMalloc((void**)&d_y,  mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    // copy host memory to device
    cudaMemcpy(d_x, h_x, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, mem_size, cudaMemcpyHostToDevice);

    unsigned int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // a small number of dry runs
    for(int r = 0; r < 1; r++) {
        addVec<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_out, N);
    }
  
    { // execute the kernel a number of times;
      // to measure performance use a large N, e.g., 200000000,
      // and increase GPU_RUNS to 100 or more. 
    
        double elapsed; struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);
        // Prefetch the x and y arrays to the GPU
        /* cudaMemPrefetchAsync(d_in, N*sizeof(float), 0, 0);
        cudaMemPrefetchAsync(d_out, N*sizeof(float), 0, 0);
        int blockSize = 256
        int numBlocks = (N + blockSize -1) /blockSize */
        for(int r = 0; r < GPU_RUNS; r++) {
            addVec<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_out, N);
        }
        cudaDeviceSynchronize();
        // ^ `cudaDeviceSynchronize` is needed for runtime
        //     measurements, since CUDA kernels are executed
        //     asynchronously, i.e., the CPU does not wait
        //     for the kernel to finish.
        //   However, `cudaDeviceSynchronize` is expensive
        //     so we need to amortize it across many runs;
        //     hence, when measuring performance use a big
        //     N and increase GPU_RUNS to 100 or more.
        //   Sure, it would be better by using CUDA events, but
        //     the current procedure is simple & works well enough.
        //   Please note that the execution of multiple
        //     kernels in Cuda executes correctly without such
        //     explicit synchronization; we need this only for
        //     runtime measurement.
        
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (1.0 * (t_diff.tv_sec*1e6+t_diff.tv_usec)) / GPU_RUNS;
        double gigabytespersec = (2.0 * N * 4.0) / (elapsed * 1000.0);
        printf("The kernel took on average %f microseconds. GB/sec: %f \n", elapsed, gigabytespersec);
        
    }
       
    // check for errors
    gpuAssert( cudaPeekAtLastError() );

    // copy result from ddevice to host
    cudaMemcpy(h_out1, d_out, mem_size, cudaMemcpyDeviceToHost);

    // print result
    //for(unsigned int i=0; i<N; ++i) printf("%.6f\n", h_out[i]);

    for(unsigned int i=0; i<N; ++i) {
        //h_out2 is CPU, h_out1 is gpu
        if (fabs(h_out2[i] - h_out1[i]) < 0.000001){ 
            float actualgpu   = h_out1[i];
            float actualcpu = h_out2[i]; 
        if( actualgpu != actualcpu ) {
            printf("Invalid result at index %d, actual: %f, expected: %f. \n", i, actualgpu, actualcpu);
            exit(3);
        }
    }
    }
    printf("Successful Validation.\n");
    
    
    
    // clean-up memory
    free(h_x);
    free(h_y);
    free(h_out1);
    free(h_out2);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);
}
