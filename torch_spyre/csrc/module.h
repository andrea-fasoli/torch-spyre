/*
 * Copyright 2025 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

#include <flex/runtime.hpp>
#include <memory>

namespace spyre {

struct SharedOwnerCtx {
  flex::DeviceMemoryAllocationPtr owner;
  size_t vf_offset = 0;  // allocation offset of reserved memory Block. VF only.
  signed char device_id;
};

struct MemoryRegion {
  // Contiguous interval of memory within a Segment (either free or occupied). VF only.

  size_t start;
  size_t end;  // one past last byte
  bool is_free;
  void* ctx_ptr;  // SharedOwnerCtx pointer for occupied regions, nullptr for free

  MemoryRegion() : start(0), end(0), is_free(true), ctx_ptr(nullptr) {}
  MemoryRegion(size_t s, size_t e, bool free = true, void* ctx = nullptr)
    : start(s), end(e), is_free(free), ctx_ptr(ctx) {}

  size_t size() const { return end - start; }

  bool operator<(const MemoryRegion& other) const {
    return start < other.start;  // for std::set ordering
  }
};

struct SegmentInfo {
  // One contiguous allocation on Spyre, via TryAllocate. VF only.
  // Allocated memory is subdivided into MemoryRegions (free or occupied).

  unsigned long segment_id;  // same as alloc_idx. Type: AIUMsg::V1::AllocationIndex = senlib::v2::LittleEndian<unsigned long>
  flex::DeviceMemoryAllocationPtr data;  // in common across all ShareOwnerCtx associated with the same Segment

  size_t total_size;
  size_t free_size;

  std::set<MemoryRegion> regions;  // all memory regions (free and occupied), ordered by start offset
  std::unordered_map<void*, MemoryRegion*> ctx_to_region;  // quick lookup from context to occupied region
  std::multiset<size_t> free_sizes;  // track sizes of all free regions for quick lookup

  SegmentInfo(unsigned long idx, size_t sz)
    : segment_id(idx), total_size(sz), free_size(sz) {}
};

class GlobalRuntime {
 public:
  static void set(const std::shared_ptr<flex::Runtime>& runtime) {
    instance() = runtime;
  }
  static void reset() {
    instance().reset();  // sets the shared_ptr to nullptr
  }

  static const std::shared_ptr<flex::Runtime>& get() {
    return instance();
  }

 private:
  GlobalRuntime() = delete;
  ~GlobalRuntime() = delete;

  static std::shared_ptr<flex::Runtime>& instance() {
    static std::shared_ptr<flex::Runtime> s;
    return s;
  }
};
bool get_downcast_warn_enabled();

}  // namespace spyre
