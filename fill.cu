#include <chrono>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include "scan.hpp"
#include "alloc.hpp"

#define DEFAULT_N 100
#define H 10
#define NBLOCKS(n, block_size) ((n + block_size - 1) / block_size)

using namespace std;

// global variables
int debug_dump = 0;
int verification = 0;

// compute a random vector of integers on the host, with about 1/H of the values
// negative (negative values represent holes).
void init_input(int *array, long n)
{
    for (long i = 0; i < n; i++)
    {
        int value = rand() % 10000;
        if (value % H == 0)
            value = -value;
        array[i] = value;
    }
}

// Reference: https://www.geeksforgeeks.org/smallest-power-of-2-greater-than-or-equal-to-n/
// Find a number which is greater than or equal to n and is the smallest power of 2.
long next_power_of_2(long n)
{
    unsigned long count = 0;
    if (n && !(n & (n - 1)))
        return n;

    while (n != 0)
    {
        n >>= 1;
        ++count;
    }

    return 1 << count;
}

// a function to dump a host array
void dump(const char *what, int *array, long n)
{
    if (debug_dump == 0)
        return;

    printf("*** %s ***\n", what);
    for (long i = 0; i < n; i++)
    {
        printf("%5d ", array[i]);
    }
    printf("\n\n");
}

// a function to print a host long number
void print(const char *what, long value)
{
    if (debug_dump == 0)
        return;
    printf("%s %ld\n", what, value);
}

// a function to dump a device array
void dump_device(const char *what, int *d_array, long n)
{
    HOST_ALLOCATE(h_temp, n, int);
    cudaMemcpy(h_temp, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);
    dump(what, h_temp, n);
    HOST_FREE(h_temp);
}

// a comparison function for qsort to sort values into descending order
int less(const void *left, const void *right)
{
    int *ileft = (int *)left;
    int *iright = (int *)right;
    return *iright - *ileft;
}

// sort a sequence of values in descending order, which puts all of the
// negative values at the end
void sort(int *h_in, long n)
{
    qsort(h_in, n, sizeof(int), less);
}

// verify that h_out contains the positive values of h_in
void verify(int *h_in, int *h_out, long n, long n_holes)
{
    if (verification == 0)
        return;

    long non_holes = n - n_holes;
    int verified = 0;

    // sort h_in in descending order so that the positive values are at the front
    sort(h_in, n);

    // sort h_out in descending order
    sort(h_out, non_holes);

    // if any value of h_out is not equal to the corresponding value in h_in, the
    // hole compaction is wrong
    for (long i = 0; i < non_holes; i++)
    {
        if (h_in[i] != h_out[i])
        {
            printf("verification failed: h_out[%d] (%d) != h_in[%d] (%d)\n", i, h_out[i], i, h_in[i]);
            return;
        }
        else
        {
            verified++;
        }
    }
    if (verified == non_holes)
        printf("verification succeeded; %d non holes!\n", non_holes);
}

// a serial hole counting algorithm
long serial_count_holes(int *in, long n)
{
    long n_holes = 0;
    for (long i = 0; i < n; i++)
    {
        if (in[i] < 0)
            n_holes++;
    }
    return n_holes;
}

// a serial hole filling algorithm
long serial_fill_holes(int *output, int *input, long n)
{
    long n_holes = 0;

    long right = n - 1; // right cursor in the input vector
    long left = 0;      // left cursor in the input and output vectors
    for (; left <= right; left++)
    {
        if (input[left] >= 0)
        {
            output[left] = input[left];
        }
        else
        {
            n_holes++; // count a hole in the prefix that needs filling
            while (right > left && input[right] < 0)
            {
                right--;
                n_holes++; // count a hole in the suffix backfill
            }
            if (right <= left)
                break;
            output[left] = input[right--]; // fill a hole at the left cursor
        }
    }

    return n_holes;
}

// a parallel hole filling algorithm
__global__ void
parallel_find_holes(int *in, int *out, long n)
{
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = in[i] < 0 ? 1 : 0;
}

// a parallel generate backfilling numbers algorithm
__global__ void
parallel_generate_backfilling_numbers(int *in, int *out, int *ex_scan, long offset, long n_holes)
{
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_holes && in[offset + i] >= 0)
    {
        long index = i - ex_scan[offset + i] + ex_scan[offset];
        out[index] = in[offset + i];
    }
}

// a parallel hole filling algorithm
__global__ void
paralel_fill_holes(int *in, int *out, int *nums, int *is_hole, int *ex_scan, long n)
{
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = is_hole[i] ? nums[ex_scan[i]] : in[i];
}

// argument processing
void getargs(int argc, char **argv, long *N)
{
    int opt;

    while ((opt = getopt(argc, argv, "dvn:")) != -1)
    {
        switch (opt)
        {
        case 'd':
            debug_dump = 1;
            break;

        case 'v':
            verification = 1;
            break;

        case 'n':
            *N = atol(optarg);
            break;

        default: /* '?' */
            fprintf(stderr, "Usage: %s [-n size] [-d] [-v]\n", argv[0]);
            exit(-1);
        }
    }
}

int main(int argc, char **argv)
{
    // Initialization and get input arguments
    long N = DEFAULT_N;
    long n_holes, number_to_fill, perfect_size;
    getargs(argc, argv, &N);
    perfect_size = next_power_of_2(N);
    printf("fill: N = %ld, debug_dump = %d\n", N, debug_dump);

    // Allocate memory on host (CPU)
    HOST_ALLOCATE(h_input, N, int);
    HOST_ALLOCATE(h_output, N, int);

    // Allocate memory on device (GPU)
    DEVICE_ALLOCATE(d_hole, N, int);
    DEVICE_ALLOCATE(d_scan, N, int);
    DEVICE_ALLOCATE(d_nums, N, int);
    DEVICE_ALLOCATE(d_input, N, int);
    DEVICE_ALLOCATE(d_output, perfect_size, int);

    // Randomly generate an integer array with the length of N and copy to device (GPU)
    init_input(h_input, N);
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    dump_device("d_input", d_input, N);

    // Fill hole in serial and count execution time
    chrono::steady_clock clock;
    auto cpu_start = clock.now();
    n_holes = serial_fill_holes(h_output, h_input, N);
    auto cpu_end = clock.now();
    auto time_span = static_cast<chrono::duration<double>>(cpu_end - cpu_start);
    print("serial count holes returns n_holes =", n_holes);
    printf("Serial Backfilling finished in: %.2f ms\n", time_span.count() * 1000);

    // Initialize variables for counting execution time on GPU
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);
    cudaEventRecord(gpu_start);

    // Fill hole in parallel
    parallel_find_holes<<<NBLOCKS(N, 1024), 1024>>>(d_input, d_hole, N);

    // Cuda version of exclusive scan
    ex_sum_scan(d_scan, d_hole, N);

    // Self-implemented version of exclusive scan
    // my_ex_sum_scan(d_scan, d_hole, perfect_size, N);

    // Calculate n_holes (H) and number_to_fill (N-H)
    cudaMemcpy(&n_holes, d_scan + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    n_holes += h_input[N - 1] < 0;
    number_to_fill = N - n_holes;

    // Generate positive integers for backfilling and compact the input array
    parallel_generate_backfilling_numbers<<<NBLOCKS(n_holes, 1024), 1024>>>(d_input, d_nums, d_scan, number_to_fill, n_holes);
    paralel_fill_holes<<<NBLOCKS(number_to_fill, 1024), 1024>>>(d_input, d_output, d_nums, d_hole, d_scan, number_to_fill);

    // count execution time on GPU
    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, gpu_start, gpu_end);
    printf("Parallel Backfilling finished in: %.2f ms\n", milliseconds);

    // Debug messages
    dump_device("d_hole", d_hole, N);
    dump_device("d_scan", d_scan, N);
    dump_device("d_nums", d_nums, n_holes);
    dump_device("d_output", d_output, number_to_fill);
    print("parallel count holes returns n_holes =", n_holes);

    // Copy backfilling result to CPU and verify the result
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    verify(h_input, h_output, N, n_holes);

    // Free host (CPU) memory
    HOST_FREE(h_input);
    HOST_FREE(h_output);

    // Free device (GPU) memory
    DEVICE_FREE(d_hole);
    DEVICE_FREE(d_scan);
    DEVICE_FREE(d_nums);
    DEVICE_FREE(d_input);
    DEVICE_FREE(d_output);

    return 0;
}
