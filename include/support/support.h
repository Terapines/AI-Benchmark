#include <string>
#include <vector>
#include <memory>

unsigned int next_power_of_2(unsigned int n);
int check_tensor(float *a, float *b, int n, const char *label);

std::vector<int> splitStringToInts(const std::string &str,
                                   char delimiter = 'x');

bool getBoolEnv(const std::string &env);

std::optional<int64_t> getIntEnv(const std::string &env);

std::unique_ptr<uint32_t[][3]> get_all_grids(uint32_t gridX, uint32_t gridY,
                                             uint32_t gridZ);

#define PRINT_KERNEL_RUNNING_TIME(Kernel, Value)                               \
  std::cerr << "Running " << Kernel << " Time: " << Value << " s" << std::endl;

const std::string TRITON_KERNEL = "Triton Kernel";
const std::string C_KERNEL = "C Kernel";