void ex_sum_scan(int *d_out, int *d_in, int n);
void debug_print(const char *what, int *d_array, long n);
void my_ex_sum_scan(int *d_out, int *d_in, long n, long limit);
__global__ void parallel_add_block_sum(int *d_out, int *d_in, int *d_block_sum, long n);
__global__ void parallel_exclusive_scan(int *d_out, int *d_in, int *d_block_sum, long n, long limit);
