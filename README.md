# AI-Benchmark
RISCV C and Triton AI-Benchmark



### 目录说明：

1. **根目录**：

   - `build.sh`：编译kernel和driver到build 目录
   - `copy_remote_back.sh`：将报表等copy回本地。
   - `copy_to_remote.sh`：将编译生成的 ELF 文件拷贝到远程 RISC-V 服务器。
   - `report.sh`：生成性能报告。
   - `run.sh`：根据不同的shape配置，运行并且生成运行时间的统计。

2. **`include` 目录**：包含头文件，用于声明不同模块的函数和数据结构。

   - `kernel`：包含C算子相关的头文件。Triton算子的头文件会自动生成到build目录
   
      - `support`： 通用的函数的头文件目录。
        - `benchmark.h`：性能测试相关的头文件。暂时没有用到
        - `omp.h`：OpenMP 并行处理相关的头文件。
        - `support.h`：通用的函数。
   

3. **`src` 目录**：源代码目录。

   - `c`： C语言kernel
   
   - `main` : 测试程序
     
     - `*.cfg`：shape的配置文件。用于统计不同shape的输入输出下算子的性能
     - `*.cpp`：测试程序。
     
   - `support`：通用的函数的源文件目录

   - `triton` Triton语言kernel
     
     - `layernorm` 目前用于保存Triton算子的LLVM IR 文件和launcher文件。 后续直接生成到build目录
       
       - `*.llir`：Triton kernel 编译后的LLVM IR 文件。
       - `*_launcher.cpp`和`*_launcher.h`：launcher文件
       
     - `*.py` ：Triton  算子的实现文件。
     
       
