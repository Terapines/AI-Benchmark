#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <iostream>

unsigned int next_power_of_2(unsigned int n);

template <typename T = float>
bool check_tensor(T *a, T *b, int n, const char *label) {
  bool ok = true;

  int j = 0;
  for (int i = 0; i < n; i++) {
    // Layer-norm error may larger than 1e-5
    if (fabs(a[i] - b[i]) > 1e-4) {
      ok = false;
      if (j++ < 32) {
        std::cout << i << " : " << a[i] << " vs " << b[i] << std::endl;
      }
      // break;
    }
  }
  std::string ACC = ok ? "OK" : "NOT OK";
  printf("%s %s\n", label, ACC.c_str());
  return ok;
}

std::vector<int> splitStringToInts(const std::string &str,
                                   char delimiter = 'x');

bool getBoolEnv(const std::string &env);

std::optional<int64_t> getIntEnv(const std::string &env);

std::optional<std::string> getStringEnv(const std::string &env);

// Data base
std::string getDB(const std::string &Shape);

std::unique_ptr<uint32_t[][3]> get_all_grids(uint32_t gridX, uint32_t gridY,
                                             uint32_t gridZ);

#define PRINT_KERNEL_RUNNING_TIME(Kernel, Value)                               \
  std::cerr << "Running " << Kernel << " Time: " << Value << " s" << std::endl;

const std::string TRITON_KERNEL = "Triton Kernel";
const std::string C_KERNEL = "C Kernel";