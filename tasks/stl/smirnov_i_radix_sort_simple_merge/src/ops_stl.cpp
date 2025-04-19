#include "stl/smirnov_i_radix_sort_simple_merge/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <future>
#include <mutex>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

std::vector<int> smirnov_i_radix_sort_simple_merge_stl::TestTaskSTL::Merge(std::vector<int>& mas1,
                                                                           std::vector<int>& mas2) {
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
void smirnov_i_radix_sort_simple_merge_stl::TestTaskSTL::RadixSort(std::vector<int>& mas) {
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
std::vector<int> smirnov_i_radix_sort_simple_merge_stl::TestTaskSTL::Sorting(int id, std::vector<int> &mas_,
                                                                             int max_th)
{
  int start = static_cast<int>(id * mas_.size() / max_th);
  int end = static_cast<int>(std::min((id + 1) * mas_.size() / max_th, mas_.size()));
  std::vector<int> local_mas_(end - start);
  std::copy(mas_.begin() + start, mas_.begin() + end, local_mas_.data());
  radix(local_mas_);
  return local_mas_;
}
void smirnov_i_radix_sort_simple_merge_stl::TestTaskSTL::Merging(std::deque<std::vector<int>> &firstdq,
                                                                 std::deque<std::vector<int>> &seconddq,
                                                                 std::mutex &mtx)
{
  std::vector<int> mas1{};
  std::vector<int> mas2{};
  std::vector<int> merge_mas{};
  mtx.lock();
  if (static_cast<int>(firstdq.size()) >= 2)
  {
    mas1 = std::move(firstdq.front());
    firstdq.pop_front();
    mas2 = std::move(firstdq.front());
    firstdq.pop_front();
  }
  else
  {
    mtx.unlock();
    return;
  }
  mtx.unlock();
  if (!mas1.empty() && !mas2.empty())
  {
    merge_mas = Merge(mas1, mas2);
  }
  if (!merge_mas.empty())
  {
    mtx.lock();
    seconddq.push_back(std::move(merge_mas));
    mtx.unlock();
  }
}
bool smirnov_i_radix_sort_simple_merge_stl::TestTaskSTL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  mas_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);
  return true;
}
bool smirnov_i_radix_sort_simple_merge_stl::TestTaskSTL::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}
bool smirnov_i_radix_sort_simple_merge_stl::TestTaskSTL::RunImpl() {
  int max_th = std::thread::hardware_concurrency();
  std::mutex mtxfirstdq;
  std::mutex mtx;
  bool flag;
  std::vector<std::future<std::vector<int>>> ths(max_th);
  for (int i = 0; i < max_th; i++)
  {
    ths[i] = std::async(std::launch::async, Sorting, i, std::ref(mas_), max_th);
  }
  std::deque<std::vector<int>> firstdq;
  std::deque<std::vector<int>> seconddq;
  for (int i = 0; i < max_th; i++)
  {
    std::vector<int> local_mas = ths[i].get();
    if (!local_mas.empty())
    {
      std::lock_guard<std::mutex> lock(mtxfirstdq);
      firstdq.push_back(std::move(local_mas));
    }
  }
  flag = static_cast<int>(firstdq.size()) != 1;
  std::vector<std::thread> threads(max_th);
  while (flag)
  {
    for (int i = 0; i < max_th; i++)
    {
      threads[i] = std::thread(Merging, std::ref(firstdq), std::ref(seconddq), std::ref(mtx));
    }
    for (auto &th : threads)
    {
      th.join();
    }
    if (static_cast<int>(firstdq.size()) == 1)
    {
      seconddq.push_back(std::move(firstdq.front()));
      firstdq.pop_front();
    }
    std::swap(firstdq, seconddq);
    flag = static_cast<int>(firstdq.size()) != 1;
  }
  output_ = std::move(firstdq.front());
  return true;
}

bool smirnov_i_radix_sort_simple_merge_stl::TestTaskSTL::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}