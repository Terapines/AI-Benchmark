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

### Autotuning
triton-cpu 已经打过0001-patch的基础上，在加上0002-autotuning patch

autotuning脚本只编译了triton kernel, 如果需要对比C kernel的数据，需要关闭build脚本中的triton kernel编译，先运行build.sh， 然后运行autotuning.sh

**NOTE**: patch文件在patch目录下

### 添加一个 kernel 到 benchmark 的流程：
1. 在 src/c/ 目录下添加与 triton-cpu 对比的 kernel 实现（比如 c++ kernel）
   1. 添加多线程的pragma
2. 在 src/triton/ 目录下添加 kernel.py 文件，该文件中实现 triton-cpu kernel 并调用该 kernel
   1. 需要注意autotuning添加的格式
   2. autotuning需要删除 kernel[grid] 调用的时候的 BLOCK_SIZE 的赋值
3. 在 src/main/ 目录下添加 kernel.cpp 文件（具体可参考 benchmark 已有的 kernel），用来调用对比 kernel 和 triton-cpu kernel，并评估性能
   1. 条件编译triton和c kernel
   2. 打印函数使用support.h中定义的宏
   3. grid计算时使用的block size已经生成到了build/include/*launcher.h 文件中， 命名规则是: {kernel}_{BLOCK_SIZE}

### 版本说明：
1. **zcc**
   - `ZCC-Pro-3.2.4-Ubuntu22.tar.bz2, md5sum: 784cde1fa77cec77a256e8565ff1328e`

2. **triton-cpu**
   - `https://github.com/northstreet12/triton-cpu.git`
   - `branch: main, commit: bfb302ffc5fde3b9efe040cb452ddac0454dbb98`
