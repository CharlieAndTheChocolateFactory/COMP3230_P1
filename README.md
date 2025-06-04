# LLaMA2 Multi-Threaded Implementation

A parallelized implementation of the LLaMA2 transformer model using POSIX threads for optimized matrix-vector multiplication operations in attention mechanisms and feed-forward networks.

## Author Information
- **Name**: Wong Chor Sing
- **UID**: 3035790769
- **Development Platform**: VSCode on macOS
- **Course**: COMP3230 Operating Systems
- **Assignment**: Programming Assignment 2

## Overview

This project implements a multi-threaded version of the LLaMA2 transformer model, specifically optimizing the matrix-vector multiplication operations that are the most computationally intensive parts of the transformer architecture. The implementation uses POSIX threads with mutex locks and conditional variables for efficient thread synchronization.

### Key Optimizations
- **Parallel Matrix-Vector Multiplication**: The bottleneck operations in attention and feed-forward networks
- **Thread Pool Architecture**: Persistent worker threads that avoid creation/destruction overhead
- **Row-wise Work Distribution**: Each thread processes a subset of matrix rows independently
- **Synchronization**: Producer-consumer pattern with conditional variables for efficient coordination

## Features

- ✅ **Multi-threaded Matrix-Vector Multiplication**: Parallelizes computation across multiple threads
- ✅ **Thread Pool Management**: Creates and manages a pool of persistent worker threads
- ✅ **Efficient Synchronization**: Uses mutex locks and conditional variables for thread coordination
- ✅ **Resource Monitoring**: Tracks CPU usage (user and system time) for performance analysis
- ✅ **Memory Management**: Proper allocation and cleanup of thread-related resources
- ✅ **Load Balancing**: Even distribution of work across available threads

## System Requirements

### Dependencies
- **Compiler**: GCC with pthread support
- **Libraries**: 
  - `pthread` (POSIX threads)
  - `libm` (Math library)
- **System**: POSIX-compliant (Linux/macOS)

### Model Files
Download the required model and tokenizer files:

```bash
# Download LLaMA2 model (required)
wget -O model.bin https://huggingface.co/huangs0/llama2.c/resolve/main/model.bin

# Download tokenizer (required)
wget -O tokenizer.bin https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin
```

**Important**: These files must be in the same directory as your executable.

## Installation & Compilation

```bash
# Compile the program
gcc -o llama2_3035790769 llama2_3035790769.c utilities.c -O2 -pthread -lm
```

### Compilation Notes
- **`-pthread`**: Required to link the pthread library
- **`-O2`**: Optimization flag for better performance
- **`-lm`**: Links the math library
- Ensure both `llama2_3035790769.c` and `utilities.c` are present

## Usage

```bash
./llama2_3035790769 <seed> <thr_count>
```

### Parameters
- **`<seed>`**: Random seed for text generation (positive integer)
- **`<thr_count>`**: Number of threads to use for parallel computation (1-16 recommended)

### Examples
```bash
# Generate text with seed 42 using 4 threads
./llama2_3035790769 42 4

# Generate text with seed 123 using 8 threads
./llama2_3035790769 123 8

# Single-threaded execution for comparison
./llama2_3035790769 100 1
```

## Architecture Overview

### Core Components

#### 1. Thread Pool Management
```c
int init_mat_vec_mul(int thr_count)    // Initialize thread pool
int close_mat_vec_mul()                // Cleanup and termination
```

#### 2. Matrix-Vector Multiplication
```c
void mat_vec_mul(float* out, float* vec, float* mat, int col, int row)
void* thr_func(void* arg)              // Worker thread function
```

#### 3. Synchronization Primitives
- **Mutex Lock**: `pthread_mutex_t lock` for thread-safe operations
- **Conditional Variables**: 
  - `condStart`: Wake up worker threads
  - `condEnd`: Signal task completion
- **Thread State**: `threadRunning[]` array for work coordination

### Data Structures

```c
struct matric_arg {
    float* out;    // Output vector
    float* vec;    // Input vector  
    float* mat;    // Input matrix
    int col;       // Number of columns
    int row;       // Number of rows
};
```

### Workflow

1. **Initialization Phase**
   - Main thread creates worker threads
   - Worker threads immediately sleep waiting for tasks
   - Synchronization primitives are initialized

2. **Computation Phase**
   - Main thread assigns matrix-vector multiplication parameters
   - Worker threads are awakened via `pthread_cond_broadcast()`
   - Each thread processes assigned row range: `[start, end)`
   - Parallel computation: `out[i] = Σ(mat[i][j] * vec[j])`

3. **Synchronization Phase**
   - Worker threads signal completion via conditional variables
   - Main thread waits for all workers to finish
   - Results are collected and control returns to transformer

4. **Termination Phase**
   - Special termination signal sent to worker threads
   - Resource usage statistics collected and displayed
   - Memory cleanup and thread joining

## Performance Analysis

### Matrix-Vector Multiplication Optimization
The transformer model performs multiple matrix-vector multiplications:

**Attention Mechanism:**
- Query projection: `q = W_q @ x`
- Key projection: `k = W_k @ x` 
- Value projection: `v = W_v @ x`
- Output projection: `out = W_o @ attention_result`

**Feed-Forward Network:**
- Gate projection: `h1 = W_1 @ x`
- Up projection: `h2 = W_3 @ x`
- Down projection: `out = W_2 @ (silu(h1) * h2)`

### Threading Strategy
- **Row-wise parallelization**: Each thread computes a subset of output rows
- **Load balancing**: Rows evenly distributed across threads
- **Memory locality**: Each thread accesses contiguous memory regions

## Sample Output

```
Once upon a time, in a distant galaxy, there lived a young astronaut named Sarah...

length: 256, time: 2.3456 s, achieved tok/s: 109.12
Thread 0 has completed - user: 0.5234 s, system: 0.0123 s
Thread 1 has completed - user: 0.5201 s, system: 0.0134 s
Thread 2 has completed - user: 0.5189 s, system: 0.0145 s
Thread 3 has completed - user: 0.5167 s, system: 0.0156 s
main thread - user: 2.1456 s, system: 0.0234 s
```

### Output Explanation
- **Generated Text**: Model output based on the provided seed
- **Performance Metrics**: 
  - `length`: Number of tokens generated
  - `time`: Total execution time
  - `tok/s`: Tokens per second (throughput)
- **Resource Usage**: CPU time breakdown for each thread and main thread

## Performance Tuning

### Optimal Thread Count
- **General Rule**: Use number of CPU cores (4-8 for most systems)
- **Testing**: Try different thread counts to find optimal performance
- **Diminishing Returns**: More threads may cause synchronization overhead

### Performance Tips
1. **Monitor Resource Usage**: Check if threads are balanced
2. **System Load**: Avoid oversubscription on busy systems
3. **Memory**: Ensure sufficient RAM for model and threads
4. **Compiler Optimization**: Use `-O2` or `-O3` for better performance

## Troubleshooting

### Common Issues

1. **Compilation Errors**
   ```bash
   # Missing pthread
   error: undefined reference to `pthread_create`
   # Solution: Add -pthread flag
   ```

2. **Missing Model Files**
   ```bash
   # File not found errors
   # Solution: Download model.bin and tokenizer.bin to same directory
   ```

3. **Runtime Errors**
   ```bash
   # Invalid arguments
   Usage: ./compiled <seed> <thr_count>
   # Solution: Provide both seed and thread count
   ```

4. **Performance Issues**
   - **Too many threads**: Try reducing thread count
   - **Memory issues**: Check available RAM
   - **System load**: Close other applications

### Debugging Tips
- Start with single thread (`thr_count = 1`) to verify correctness
- Use small thread counts (2-4) for debugging
- Monitor system resources with `top` or `htop`

## Implementation Details

### Thread Safety
- All shared data access protected by mutex locks
- Conditional variables prevent busy waiting
- Each thread has private computation space

### Memory Management
- Dynamic allocation for thread-related structures
- Proper cleanup in `close_mat_vec_mul()`
- No memory leaks in normal execution

### Error Handling
- Argument validation in main function
- Resource cleanup on exit
- Graceful thread termination

## Technical Notes

### Synchronization Pattern
The implementation uses a **producer-consumer pattern**:
- **Producer**: Main thread assigns work
- **Consumer**: Worker threads process assignments
- **Coordination**: Conditional variables for efficient waiting

### Resource Usage Tracking
Uses `getrusage()` system call to collect:
- **User time**: CPU time spent in user mode
- **System time**: CPU time spent in kernel mode
- **Per-thread statistics**: Individual thread resource usage

## License & Academic Use

This implementation is developed for COMP3230 Operating Systems coursework. The code demonstrates:
- Multi-threading concepts and implementation
- Synchronization primitives usage
- Performance optimization techniques
- System resource monitoring

**Note**: This is educational code for academic purposes.
