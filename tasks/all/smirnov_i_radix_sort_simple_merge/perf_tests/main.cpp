#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/smirnov_i_radix_sort_simple_merge/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(smirnov_i_radix_sort_simple_merge_all, test_pipeline_run) {
  constexpr int kCount = 10000000;
  boost::mpi::communicator world;
  // Create data
  std::vector<int> in(kCount, 0);
  std::vector<int> exp_out(kCount, 0);
  std::vector<int> out(kCount);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = static_cast<int>(i);
    exp_out[i] = static_cast<int>(i);
  }

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto test_task_all = std::make_shared<smirnov_i_radix_sort_simple_merge_all::TestTaskALL>(task_data_all);
  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (world.rank() == 0) {
    ASSERT_EQ(exp_out, out);
  }
}

TEST(smirnov_i_radix_sort_simple_merge_all, test_task_run) {
  constexpr int kCount = 10000000;
  boost::mpi::communicator world;
  // Create data
  std::vector<int> in(kCount, 0);
  std::vector<int> exp_out(kCount, 0);
  std::vector<int> out(kCount);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = static_cast<int>(i);
    exp_out[i] = static_cast<int>(i);
  }

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto test_task_all = std::make_shared<smirnov_i_radix_sort_simple_merge_all::TestTaskALL>(task_data_all);
  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (world.rank() == 0) {
    ASSERT_EQ(exp_out, out);
  }
}