#include <cub/cub.cuh>
#include "scan.hpp"

#define BLOCK_SIZE 1024
#define SHARED_MEMORY 49152

// compute an exclusive sum scan of d_in in O(log_2 n) steps
void ex_sum_scan(int *d_out, int *d_in, int n)
{
    void *d_tmp = NULL;   // pointer to temporary storage used by the the scan
    size_t tmp_bytes = 0; // the number of bytes of temporary storage needed

    // determine how many bytes of temporary storage are needed
    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_in, d_out, n);

    // allocate temporary storage
    cudaMalloc(&d_tmp, tmp_bytes);

    // compute the exclusive prefix sum of d_in into d_out using d_tmp as
    // temporary storage
    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_in, d_out, n);

    // feee the temporary storage
    cudaFree(d_tmp);
}

void my_ex_sum_scan(int *d_out, int *d_in, long n, long limit)
{
    int *d_temp = NULL, *d_block_sum = NULL;
    long grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_temp, n * sizeof(int));
    cudaMalloc(&d_block_sum, grid_size * sizeof(int));
    cudaMemset(d_block_sum, 0, sizeof(unsigned int) * grid_size);
    cudaMemcpy(d_temp, d_in, limit * sizeof(int), cudaMemcpyDeviceToDevice);

    // Get partial exclusive sum result
    parallel_exclusive_scan<<<grid_size, BLOCK_SIZE, SHARED_MEMORY>>>(d_out, d_temp, d_block_sum, n, limit);

    // Perform exclusive sum for each BLOCK_SIZE block
    if (grid_size <= BLOCK_SIZE)
    {
        int *d_dummy_block_sum = NULL;
        cudaMalloc(&d_dummy_block_sum, sizeof(int));
        parallel_exclusive_scan<<<1, BLOCK_SIZE, SHARED_MEMORY>>>(d_block_sum, d_block_sum, d_dummy_block_sum, BLOCK_SIZE, BLOCK_SIZE);
        cudaFree(d_dummy_block_sum);
    }
    else
    {
        int *d_in_block_sum = NULL;
        cudaMalloc(&d_in_block_sum, grid_size * sizeof(int));
        cudaMemcpy(d_in_block_sum, d_block_sum, grid_size * sizeof(int), cudaMemcpyDeviceToDevice);
        my_ex_sum_scan(d_block_sum, d_in_block_sum, grid_size, grid_size);
        cudaFree(d_in_block_sum);
    }

    // Gather the partial scan result and restore into the entire results
    parallel_add_block_sum<<<grid_size, BLOCK_SIZE>>>(d_out, d_out, d_block_sum, limit);

    // Free Device (GPU) Memory
    cudaFree(d_temp);
}

// Reference: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
__global__ void parallel_exclusive_scan(int *d_out, int *d_in, int *d_block_sum, long n, long limit)
{
    extern __shared__ int temp[];

    long offset = 1;
    long thread_id = threadIdx.x;
    long global_id = blockIdx.x * blockDim.x + threadIdx.x;
    temp[thread_id] = temp[blockDim.x + thread_id] = 0;

    __syncthreads();
    if (global_id < n)
        temp[thread_id] = d_in[global_id];

    // Build sum in place up the tree (up-sweep)
    for (int d = BLOCK_SIZE >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (thread_id < d)
        {
            int ai = offset * (2 * thread_id + 1) - 1;
            int bi = offset * (2 * thread_id + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (thread_id == 0)
    {
        d_block_sum[blockIdx.x] = temp[BLOCK_SIZE - 1];
        temp[BLOCK_SIZE - 1] = 0;
    }

    // Traverse down tree and build the exclusive sum result (down-sweep)
    for (int d = 1; d < BLOCK_SIZE; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (thread_id < d)
        {
            int ai = offset * (2 * thread_id + 1) - 1;
            int bi = offset * (2 * thread_id + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    // Copy the result from the shared memory to the device memory
    __syncthreads();
    if (global_id < limit)
        d_out[global_id] = temp[thread_id];
}

// Reference: https://github.com/mark-poscablo/gpu-prefix-sum
__global__ void parallel_add_block_sum(int *d_out, int *d_in, int *d_block_sum, long n)
{
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d_out[i] = d_in[i] + d_block_sum[blockIdx.x];
}

void debug_print(const char *what, int *d_array, long n)
{
    int *h_temp = new int[n];
    cudaMemcpy(h_temp, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("*** %s ***\n", what);
    for (long i = 0; i < n; i++)
    {
        printf("%5d ", h_temp[i]);
    }
    printf("\n\n");
    free(h_temp);
}
