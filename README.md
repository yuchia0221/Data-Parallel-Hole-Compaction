## Data-parallel Hole Compaction with CUDA

#### Table of Contents

-   [Folder Structure](#folder-structure)
-   [Commands](#commands)

---

### Folder Structure

    .
    ├── result                       # Experiment Resultsresults
    │   ├── number.txt           # Execution result given input size number with Cuda built-in exclusive scan
    │   ├── my-number.txt        # Execution result given input size number with self-implemented version of exclusive scan
    │   └── compare-number.txt   # Comparision of two versions of exclusive scan
    ├── fill.cu                      # Parallelized Version
    ├── scan.cu                      # Two versions of the exclusive scan algorithm
    ├── scan.hpp                     # Header File for exclusive scan
    ├── alloc.hpp                    # Utility functions for allocating memory on Host and Device
    ├── Makefile                     # Recipes for building and running your program
    └── README.md

---

## Commands

Makefile:

> a Makefile that includes recipes for building and running your program.

```bash
make                # builds your code
make view           # views HPCToolkit result
make run-hpc        # creates a HPCToolkit database for performance measurements
make clean          # removes all executable files
make clean-hpc      # removes all HPCToolkit-related files
make run            # run experiments with Cuda implementaion given input size 400M without debug messages or verfication
make run-d          # run experiments with Cuda implementaion given input size 20 with debug messages but without verfication
make run-v          # run experiments with Cuda implementaion given input size 400M with verfication but without debug messages
make run-dv         # run experiments with Cuda implementaion given input size 20 with debug messages and verfication
```
