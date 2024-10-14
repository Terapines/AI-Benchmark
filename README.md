# AI-Benchmark
RISCV C and Triton AI-Benchmark


### Directory Description:

1. **Root Directory**：

   - `build.sh`: Compiles the kernel and driver into the build directory.
   - `copy_remote_back.sh`: Copies reports and other files back to the local machine.
   - `copy_to_remote.sh`: Copies the compiled ELF files to the remote RISC-V server.
   - `report.sh`: Generates performance reports.
   - `run.sh`: Runs configurations based on different shapes and generates runtime statistics.
   - `plot_autuning.py`: Used to plot the autuning result, help finding the best autuning block shape
   - `plot_benchmark.py`: Used to plot the benchmark result.

   **NOTE**: Need fixed the environment variable in the script.

2. **`include`**: Contains header files used to declare functions and data structures for different modules.

   - `kernel`: Contains header files related to C operators. Triton kernel header files are automatically generated into the build directory.
      - `support`: Directory for header files of common functions.
        - `benchmark.h`: Header file related to performance testing. Currently not used.
        - `omp.h`: Header file related to OpenMP parallel processing.
        - `support.h`: Common functions.


3. **`src`**: Source code directory.

   - `c`: C language kernel.

   - `main`: Test programs.

     - `*.cfg`: Configuration files for shapes. Used to evaluate kernel performance under different input and output shapes.
     - `*.cpp`: Test programs.

   - `support`: Directory for source files of common functions.

   - `triton`: Triton language kernel.
4. **`patch`**: Triton-CPU patch file.
   * 0001: Support RISC-V cross-compile
   * 0002: Support autotuning
   * 0003: Support discrete memory access lowering to vector.gather

### Autotuning
Based on triton-cpu with the 0001-patch applied, the 0002-autotuning patch is added.

The autotuning script only compiles the Triton kernel. If you need to compare data with the C kernel, you need to disable the Triton kernel compilation in the build script, first run build.sh, and then run autotuning.sh.

**NOTE**: Patch files are located in the patch directory.

### Procedure to add a kernel to the AI-Benchmark：
1. Add a kernel implementation (e.g., C++ kernel) in the src/c/ directory to compare with the Triton kernel.
   1. Add multi-threading pragmas.
2. Add a kernel.py file in the src/triton/ directory. This file implements the Triton kernel.
   1. Pay attention to the format added by autotuning.
   2. Autotuning needs to remove the assignment of BLOCK_SIZE when calling kernel[grid].
3. Add a kernel.cpp file in the src/main/ directory (refer to existing kernels in the benchmark for specifics) to call and compare the kernel with the Triton kernel and evaluate performance.
   1. Conditionally compile Triton and C kernels.
   2. Use the macros defined in support.h for printing functions.
   3. The block size used during grid computation has been generated into the build/include/*launcher.h files, following the naming convention: {kernel}_{BLOCK_SIZE}.

### Version Information:
1. **zcc**
   - `ZCC-Pro-3.2.4-Ubuntu22.tar.bz2, md5sum: 784cde1fa77cec77a256e8565ff1328e`

2. **triton-cpu**
   - `https://github.com/northstreet12/triton-cpu.git`
   - `branch: main, commit: bfb302ffc5fde3b9efe040cb452ddac0454dbb98`
