/*
 * Copyright 2026 The Torch-Spyre Authors.
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

#include <c10/core/Allocator.h>
#include <flex/runtime.hpp>
#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>

#include "module.h"

namespace spyre {

struct MemoryBlock {
  // Contiguous interval of memory within a Segment (either free or occupied).
  // VF only.

  size_t start;  // block starting offset
  size_t end;    // block ending offset (one past last byte)
  bool is_free;  // block represents memory occupied or free

  MemoryBlock() : start(0), end(0), is_free(true) {}
  MemoryBlock(size_t s, size_t e, bool free = true, void* ctx = nullptr)
      : start(s), end(e), is_free(free) {}

  size_t size() const {
    return end - start;
  }

  bool operator<(const MemoryBlock& other) const {
    return start < other.start;  // for std::set ordering
  }
};

struct MemorySegment {
  // One contiguous allocation on Spyre, via TryAllocate. VF only.
  // Allocated memory is subdivided into MemoryBlocks (free or occupied).

  size_t segment_id;  // same as alloc_idx. Type: AIUMsg::V1::AllocationIndex =
                      // senlib::v2::LittleEndian<unsigned long>
  flex::DeviceMemoryAllocationPtr
      data;  // in common across all ShareOwnerCtx associated with the same
             // MemorySegment

  size_t total_size;  // total allocated memory for this Segment
  size_t free_size;   // total available memory

  std::set<MemoryBlock>
      blocks;  // all memory blocks (free and occupied), ordered by start offset
  std::unordered_map<SharedOwnerCtx*, MemoryBlock*>
      ctx_to_block;  // quick lookup from context to occupied block
  std::multiset<size_t>
      free_sizes;  // track sizes of all free blocks for quick lookup

  MemorySegment(size_t seg_id, size_t sz)
      : segment_id(seg_id), total_size(sz), free_size(sz) {}
};

// Forward declarations for derived allocator classes
struct PFSpyreAllocator;
struct VFSpyreAllocator;

// Base allocator class, no longer final to allow inheritance
struct SpyreAllocator : public at::Allocator {
 protected:
  bool alloc_debug = false;  // control debug printouts

  static constexpr size_t MAX_SEGMENTS = 12;      // NOTE: limit to be defined
  static constexpr size_t MIN_ALLOC_BYTES = 128;  // Spyre requirement

  flex::DeviceMemoryAllocatorPtr getAllocator(unsigned int dev_id);

  static bool is_pf_mode();
  static bool is_alloc_debug();

  SpyreAllocator();

 public:
  virtual ~SpyreAllocator() = default;

  at::DeleterFnPtr raw_deleter() const override;
  void copy_data(void* dest, const void* src,
                 std::size_t count) const override;

  // Factory method to create the appropriate allocator
  static SpyreAllocator& instance();
};

// PF (Physical Function) Allocator - simple direct allocation
struct PFSpyreAllocator final : public SpyreAllocator {
 private:
  static void ReportAndDelete(void* ctx_void);

 public:
  PFSpyreAllocator();
  at::DataPtr allocate(size_t nbytes) override;
};

// VF (Virtual Function) Allocator - complex segment-based allocation
struct VFSpyreAllocator final : public SpyreAllocator {
 private:
  // Segments and Blocks storage/handling
  std::vector<MemorySegment> segments;
  std::unordered_map<SharedOwnerCtx*, MemorySegment*> block_to_segment;

  // Dynamic allocation control
  bool segments_locked;
  std::vector<size_t> fallback_sizes;
  size_t max_segments;

  // Mutex to protect shared state in VF mode
  mutable std::mutex allocator_mutex;

  // Static atomic pointer to this instance for ReportAndDelete
  static std::atomic<VFSpyreAllocator*> instance_ptr;

  struct AllocationInfo {
    MemorySegment* segment;
    MemoryBlock block;
    bool found;
  };

  static void ReportAndDelete(void* ctx_void);
  bool allocateNewSegment(flex::DeviceMemoryAllocatorPtr allocator);
  size_t setMinSpyreAllocation(size_t nbytes) const;
  AllocationInfo findFreeBlock(size_t nbytes,
                               flex::DeviceMemoryAllocatorPtr allocator);
  MemoryBlock* allocateInSegment(MemorySegment* seg, MemoryBlock block,
                                 size_t nbytes);
  void deallocateBlock(MemorySegment& seg, SharedOwnerCtx* ctx);
  void logSegmentState(const MemorySegment& seg, const char* context,
                       bool include_blocks = false);
  void logAllSegments(const char* context, bool include_blocks = false);

 public:
  VFSpyreAllocator(size_t max_seg = MAX_SEGMENTS);
  ~VFSpyreAllocator() override;
  at::DataPtr allocate(size_t nbytes) override;
};

}  // namespace spyre
