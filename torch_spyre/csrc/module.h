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
  size_t vf_offset = 0;  // VF only. Allocation offset of reserved Block within a Segment.
  signed char device_id;
};

struct BlockInfo {
  size_t offset_init;
  size_t offset_end;

  BlockInfo() : offset_init(0), offset_end(0) {}
  BlockInfo(size_t x, size_t y)
    : offset_init(x), offset_end(y) {}
};

struct FreeInterval {
  size_t start;
  size_t end;  // one past last byte

  bool operator<(const FreeInterval& other) const {
    return start < other.start;   // for std::set ordering
  }
};

struct SegmentInfo {
  unsigned long segment_id;  // VF only. Same as alloc_idx. Type: AIUMsg::V1::AllocationIndex = senlib::v2::LittleEndian<unsigned long>
  flex::DeviceMemoryAllocationPtr data;  // needed for deallocation? CHECK

  size_t total_size;
  size_t free_size;

  std::unordered_map<void*, BlockInfo> blocks;  // map ShareOwnerCtx ptr -> BlockInfo
  std::set<FreeInterval> free_intervals;  // track available memory
  std::multiset<size_t> free_interval_sizes;  // track sizes of all free intervals

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
