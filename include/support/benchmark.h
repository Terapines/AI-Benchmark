
#include <chrono>
#include <functional>
#include <vector>
#include <math.h>

class PerfCounter {
private:
  std::chrono::_V2::system_clock::time_point Time;

public:
  void startCounter() { Time = std::chrono::high_resolution_clock::now(); }

  float getElapsedTime(const PerfCounter &End) {
    std::chrono::duration<double> duration = End.Time - this->Time;
    return duration.count();
  }

  PerfCounter() : Time() {}
};

template <typename... Args>
std::vector<float> do_bench(std::function<void(Args...)> &fn, int warmup = 25,
                            int rep = 100, Args... args) {
  fn(args...);
  // synchronize();

  // Currently, a typical L3 cache size for high-end server CPUs are ~400MB.
  size_t cache_size = 512 * 1024 * 1024;

  std::vector<int> cache(cache_size / sizeof(int), 0);

  PerfCounter start_event;
  PerfCounter end_event;

  /// NOTE: Whether need consider cache flush time
  start_event.startCounter();
  for (int i = 0; i < 5; ++i) {
    // Flush cache
    std::fill(cache.begin(), cache.end(), 0);
    fn(args...);
  }
  end_event.startCounter();

  float estimate_ms = start_event.getElapsedTime(end_event) / 5;

  int n_warmup = std::max(1, int(ceil(warmup / estimate_ms)));
  int n_repeat = std::max(1, int(ceil(rep / estimate_ms)));

  std::vector<PerfCounter> start_events(n_repeat);
  std::vector<PerfCounter> end_events(n_repeat);

  // Warm-up
  for (int i = 0; i < n_warmup; ++i) {
    fn(args...);
  }

  // Benchmark
  std::vector<float> times(n_repeat);
  for (int i = 0; i < n_repeat; ++i) {
    // Flush cache
    std::fill(cache.begin(), cache.end(), 0);

    start_events[i].startCounter();
    fn(args...);
    end_events[i].startCounter();

    // synchronize();
    times[i] = start_events[i].getElapsedTime(end_events[i]);
  }

  return times;
}
