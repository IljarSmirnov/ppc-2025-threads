#include "all/smirnov_i_radix_sort_simple_merge/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"
#include "mpi.h"

void smirnov_i_radix_sort_simple_merge_all::TestTaskALL::DistributeData(int rank, int size, int n, std::vector<int> &sendcounts,
                                                                        std::vector<int> &displs, std::vector<int> &local_data,
                                                                        const std::vector<int> &data) {
  for (int i = 0; i < size; i++) {
    sendcounts[i] = n / size + (i < n % size ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
  }
  local_data.resize(sendcounts[rank]);
  MPI_Scatterv(data.data(), sendcounts.data(), displs.data(), MPI_INT, local_data.data(), sendcounts[rank], MPI_INT, 0,
               MPI_COMM_WORLD);
}
void smirnov_i_radix_sort_simple_merge_all::TestTaskALL::ProcessThreads(int max_th, std::deque<std::vector<int>> &firstdq,
                                                                        std::deque<std::vector<int>> &seconddq,
                                                                        std::vector<std::thread> &threads) {
  bool flag = static_cast<int>(firstdq.size()) != 1;
  while (flag) {
    int pairs = (static_cast<int>(firstdq.size()) + 1) / 2;
    threads.clear();
    threads.reserve(pairs);
    for (int i = 0; i < pairs; i++) {
      threads.emplace_back(&smirnov_i_radix_sort_simple_merge_all::TestTaskALL::Merging, std::ref(firstdq),
                           std::ref(seconddq), std::ref(mtx));
    }
    for (auto &th : threads) {
      th.join();
    }
    if (static_cast<int>(firstdq.size()) == 1) {
      seconddq.push_back(std::move(firstdq.front()));
      firstdq.pop_front();
    }
    std::swap(firstdq, seconddq);
    flag = static_cast<int>(firstdq.size()) != 1;
    if (firstdq.empty() && seconddq.empty()) {
      flag = false;
    }
  }
}
void smirnov_i_radix_sort_simple_merge_all::TestTaskALL::CollectData(int rank, int size, std::deque<std::vector<int>> &globdq_a,
                                                                     std::vector<int> &local_res) {
  for (int i = 0; i < size; i++) {
    std::vector<int> local_sorted;
    if (i == 0) {
      local_sorted = local_res;
    } else {
      int res_size = 0;
      MPI_Recv(&res_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      local_sorted.clear();
      local_sorted.resize(res_size);
      MPI_Recv(local_sorted.data(), res_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (!local_sorted.empty()) {
      globdq_a.push_back(std::move(local_sorted));
    }
  }
}
std::vector<int> smirnov_i_radix_sort_simple_merge_all::TestTaskALL::Merge(std::vector<int> &mas1,
                                                                           std::vector<int> &mas2) {
  std::vector<int> res;
  res.reserve(mas1.size() + mas2.size());
  int p1 = 0;
  int p2 = 0;
  while (static_cast<int>(mas1.size()) != p1 && static_cast<int>(mas2.size()) != p2) {
    if (mas1[p1] < mas2[p2]) {
      res.push_back(mas1[p1]);
      p1++;
    } else if (mas2[p2] < mas1[p1]) {
      res.push_back(mas2[p2]);
      p2++;
    } else {
      res.push_back(mas1[p1]);
      res.push_back(mas2[p2]);
      p1++;
      p2++;
    }
  }
  while (static_cast<int>(mas1.size()) != p1) {
    res.push_back(mas1[p1]);
    p1++;
  }
  while (static_cast<int>(mas2.size()) != p2) {
    res.push_back(mas2[p2]);
    p2++;
  }
  return res;
}
void smirnov_i_radix_sort_simple_merge_all::TestTaskALL::RadixSort(std::vector<int> &mas) {
  if (mas.empty()) {
    return;
  }
  int longest = *std::ranges::max_element(mas.begin(), mas.end());
  int len = std::ceil(std::log10(longest + 1));
  std::vector<int> sorting(mas.size());
  int base = 1;
  for (int j = 0; j < len; j++, base *= 10) {
    std::vector<int> counting(10, 0);
    for (size_t i = 0; i < mas.size(); i++) {
      counting[mas[i] / base % 10]++;
    }
    std::partial_sum(counting.begin(), counting.end(), counting.begin());
    for (int i = static_cast<int>(mas.size() - 1); i >= 0; i--) {
      int pos = counting[mas[i] / base % 10] - 1;
      sorting[pos] = mas[i];
      counting[mas[i] / base % 10]--;
    }
    std::swap(mas, sorting);
  }
}
std::vector<int> smirnov_i_radix_sort_simple_merge_all::TestTaskALL::Sorting(int id, std::vector<int> &mas,
                                                                             int max_th) {
  int start = static_cast<int>(id * mas.size() / max_th);
  int end = static_cast<int>(std::min((id + 1) * mas.size() / max_th, mas.size()));
  std::vector<int> local_mas(end - start);
  std::copy(mas.begin() + start, mas.begin() + end, local_mas.data());
  RadixSort(local_mas);
  return local_mas;
}
void smirnov_i_radix_sort_simple_merge_all::TestTaskALL::Merging(std::deque<std::vector<int>> &firstdq,
                                                                 std::deque<std::vector<int>> &seconddq,
                                                                 std::mutex &mtx) {
  std::vector<int> mas1{};
  std::vector<int> mas2{};
  std::vector<int> merge_mas{};
  {
    std::lock_guard<std::mutex> lock(mtx);
    if (static_cast<int>(firstdq.size()) >= 2) {
      mas1 = std::move(firstdq.front());
      firstdq.pop_front();
      mas2 = std::move(firstdq.front());
      firstdq.pop_front();
    } else {
      return;
    }
  }
  if (!mas1.empty() && !mas2.empty()) {
    merge_mas = Merge(mas1, mas2);
  }
  if (!merge_mas.empty()) {
    std::lock_guard<std::mutex> lock(mtx);
    seconddq.push_back(std::move(merge_mas));
  }
}

bool smirnov_i_radix_sort_simple_merge_all::TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    unsigned int input_size = task_data->inputs_count[0];
    auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    mas_ = std::vector<int>(in_ptr, in_ptr + input_size);

    unsigned int output_size = task_data->outputs_count[0];
    output_ = std::vector<int>(output_size, 0);
  }
  return true;
}
bool smirnov_i_radix_sort_simple_merge_all::TestTaskALL::ValidationImpl() {
  bool is_valid = true;
  if (world_.rank() == 0) {
    is_valid = task_data->inputs_count[0] == task_data->outputs_count[0];
  }
  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return is_valid;
}

bool smirnov_i_radix_sort_simple_merge_all::TestTaskALL::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> sendcounts(size);
  std::vector<int> displs(size, 0);
  int offset = 0;
  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(mas_.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // for (int i = 0; i < size; i++) {
  //   sendcounts[i] = n / size + (i < n % size ? 1 : 0);
  //   displs[i] = offset;
  //   offset += sendcounts[i];
  // }
  // std::vector<int> local_mas(sendcounts[rank]);
  // MPI_Scatterv(mas_.data(), sendcounts.data(), displs.data(), MPI_INT, local_mas.data(), sendcounts[rank], MPI_INT,
  // 0, MPI_COMM_WORLD);

  DistributeData(rank, size, n, sendcounts, displs, local_data, data);

  const int max_th = ppc::util::GetPPCNumThreads();
  std::mutex mtxfirstdq;
  std::mutex mtx;
  std::vector<std::future<std::vector<int>>> ths(max_th);
  for (int i = 0; i < max_th; i++) {
    ths[i] = std::async(std::launch::async, &smirnov_i_radix_sort_simple_merge_all::TestTaskALL::Sorting, i,
                        std::ref(local_mas), max_th);
  }
  std::deque<std::vector<int>> firstdq;
  std::deque<std::vector<int>> seconddq;
  for (int i = 0; i < max_th; i++) {
    std::vector<int> local_th_mas = ths[i].get();
    if (!local_th_mas.empty()) {
      std::lock_guard<std::mutex> lock(mtxfirstdq);
      firstdq.push_back(std::move(local_th_mas));
    }
  }
  ProcessThreads(max_th, firstdq, seconddq, threads);
  // bool flag = static_cast<int>(firstdq.size()) != 1;
  // std::vector<std::thread> threads{};
  // while (flag) {
  //   int pairs = (static_cast<int>(firstdq.size()) + 1) / 2;
  //   threads.clear();
  //   threads.reserve(pairs);
  //   for (int i = 0; i < pairs; i++) {
  //     threads.emplace_back(&smirnov_i_radix_sort_simple_merge_all::TestTaskALL::Merging, std::ref(firstdq),
  //                          std::ref(seconddq), std::ref(mtx));
  //   }
  //   for (auto &th : threads) {
  //     th.join();
  //   }
  //   if (static_cast<int>(firstdq.size()) == 1) {
  //     seconddq.push_back(std::move(firstdq.front()));
  //     firstdq.pop_front();
  //   }
  //   std::swap(firstdq, seconddq);
  //   flag = static_cast<int>(firstdq.size()) != 1;
  //   if (firstdq.empty() && seconddq.empty()) {
  //     flag = false;
  //   }
  // }
  std::vector<int> local_res;
  if (!firstdq.empty()) {
    local_res = std::move(firstdq.front());
  }

  std::deque<std::vector<int>> globdq_a;
  if (rank == 0) {
    CollectData(rank, size, globdq_a, local_res);
  } else {
    int send_size = local_res.size();
    MPI_Send(&send_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(local_res.data(), send_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
  // std::deque<std::vector<int>> globdq_a;
  // if (rank == 0) {
  //   std::vector<int> local_sorted;
  //   for (int i = 0; i < size; i++) {
  //     if (i == 0) {
  //       local_sorted = local_res;
  //     } else {
  //       int res_size = 0;
  //       MPI_Recv(&res_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //       local_sorted.clear();
  //       local_sorted.resize(res_size);
  //       MPI_Recv(local_sorted.data(), res_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //     }
  //     if (!local_sorted.empty()) {
  //       globdq_a.push_back(std::move(local_sorted));
  //     }
  //   }
  // } else {
  //   int send_size = static_cast<int>(local_res.size());
  //   MPI_Send(&send_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  //   MPI_Send(local_res.data(), send_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
  // }

  if (rank == 0) {
    flag = static_cast<int>(globdq_a.size()) != 1;
    std::vector<std::thread> ts{};
    std::deque<std::vector<int>> globdq_b;
    while (flag) {
      int pairs = (static_cast<int>(globdq_a.size()) + 1) / 2;
      ts.clear();
      ts.reserve(pairs);
      for (int i = 0; i < pairs; i++) {
        ts.emplace_back(&smirnov_i_radix_sort_simple_merge_all::TestTaskALL::Merging, std::ref(globdq_a),
                        std::ref(globdq_b), std::ref(mtx));
      }
      for (auto &th : ts) {
        th.join();
      }
      if (static_cast<int>(globdq_a.size()) == 1) {
        globdq_b.push_back(std::move(globdq_a.front()));
        globdq_a.pop_front();
      }
      std::swap(globdq_a, globdq_b);
      flag = static_cast<int>(globdq_a.size()) != 1;
    }
    output_ = std::move(globdq_a.front());
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool smirnov_i_radix_sort_simple_merge_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (size_t i = 0; i < output_.size(); i++) {
      reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
    }
  }
  return true;
}