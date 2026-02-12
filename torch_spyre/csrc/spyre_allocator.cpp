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

#include "spyre_allocator.h"

#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

#include "logging.h"
#include "module.h"

namespace spyre {

// ============================================================================
// SpyreAllocator - Base Class Implementation
// ============================================================================

flex::DeviceMemoryAllocatorPtr SpyreAllocator::getAllocator(
    unsigned int dev_id) {
  return GlobalRuntime::get()
      ->GetDeviceHandle(dev_id)
      ->GetDeviceMemoryAllocator();
}

bool SpyreAllocator::is_pf_mode() {
  const char* fmode_envvar = std::getenv("FLEX_DEVICE");
  TORCH_CHECK(fmode_envvar != nullptr, "FLEX_DEVICE env var is not set!")

  std::string fmode = fmode_envvar;
  // Capitalize fmode to handle lowercase env var values (e.g., "vf" or "pf")
  std::transform(fmode.begin(), fmode.end(), fmode.begin(), ::toupper);
  if (fmode == "VF") {
    return false;
  } else if (fmode == "PF") {
    return true;
  } else {
    TORCH_CHECK(false, "Unsupported FLEX_DEVICE env var value: ", fmode);
  }
}

bool SpyreAllocator::is_alloc_debug() {
  const char* alloc_envvar = std::getenv("TORCH_SPYRE_ALLOC_DEBUG");
  if (alloc_envvar == nullptr) return false;

  std::string alloc_debug_str = alloc_envvar;
  return alloc_debug_str == "1";
}

SpyreAllocator::SpyreAllocator() {
  alloc_debug = is_alloc_debug();
}

at::DeleterFnPtr SpyreAllocator::raw_deleter() const {
  return nullptr;
}

void SpyreAllocator::copy_data(void* dest, const void* src,
                               std::size_t count) const {
  py::gil_scoped_acquire acquire;  // Python thread-safety mechanism
  DEBUGINFO("entering allocator->copy_data method");
  // do nothing -- look into when this is called
  // spyre_copy_from(reinterpret_cast<spyre_ptr_t>(dest),
  // reinterpret_cast<spyre_ptr_t>(src));
}

// ============================================================================
// PFSpyreAllocator - PF Mode Implementation
// ============================================================================

void PFSpyreAllocator::ReportAndDelete(void* ctx_void) {
  /* Called when DataPtr is being deallocated in PF mode. */
  if (!ctx_void) return;
  auto* ctx = static_cast<SharedOwnerCtx*>(ctx_void);
  delete ctx;
}

PFSpyreAllocator::PFSpyreAllocator() : SpyreAllocator() {}

at::DataPtr PFSpyreAllocator::allocate(size_t nbytes) {
  /* PF allocation implementation. Functionalities are preserved exactly from
   * earlier iteration of the code (PF-only).
   */

  c10::Device curr_device =
      c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
          ->getDevice();
  auto device_id = curr_device.index();
  DEBUGINFO("allocating", nbytes, "bytes on Spyre", curr_device, "(PF mode)");
  if (nbytes <= 0) return {nullptr, nullptr, &ReportAndDelete, curr_device};

  auto allocator = getAllocator(device_id);

  DEBUGINFO("PF allocation");
  flex::DeviceMemoryAllocationPtr data;      // a smart-pointer object
  allocator->TryAllocate(&data, nbytes, 0);  // allocation request to Spyre
  TORCH_CHECK(data, "Failed to allocate ", nbytes, " bytes on Spyre device.");

  // Instantiate object to live beyond SpyreAllocator scope.
  // vf_offset is set to 0 and not used in PF Mode
  auto* ctx = new SharedOwnerCtx{std::move(data), 0, device_id};
  void* ctx_void = static_cast<void*>(ctx);
  void* data_void = static_cast<void*>(ctx->owner.get());

  return at::DataPtr(data_void, ctx_void, &ReportAndDelete, curr_device);
}

// ============================================================================
// VFSpyreAllocator - VF Mode Implementation
// ============================================================================

// Define static member
std::atomic<VFSpyreAllocator*> VFSpyreAllocator::instance_ptr{nullptr};

void VFSpyreAllocator::ReportAndDelete(void* ctx_void) {
  /* Called when DataPtr is being deallocated in VF mode. */

  if (!ctx_void) return;

  auto* ctx = static_cast<SharedOwnerCtx*>(ctx_void);
  // Atomic read to initialize allocator
  VFSpyreAllocator* allocator = instance_ptr.load(std::memory_order_acquire);

  // Guard against dangling pointer if allocator was destroyed
  if (!allocator) {
    delete ctx;
    return;
  }

  std::lock_guard<std::mutex> lock(allocator->allocator_mutex);

  if (allocator->alloc_debug)
    allocator->logAllSegments("Pre deallocation", true);

  // Using lookup map for blocks into segments (O(1))
  auto seg_it = allocator->block_to_segment.find(ctx);
  if (seg_it != allocator->block_to_segment.end()) {
    allocator->deallocateBlock(*seg_it->second, ctx);
    allocator->block_to_segment.erase(seg_it);
  }

  if (allocator->alloc_debug)
    allocator->logAllSegments("Post deallocation", true);

  delete ctx;
}

bool VFSpyreAllocator::allocateNewSegment(
    flex::DeviceMemoryAllocatorPtr allocator) {
  /* Try to allocate a single new segment, attempting fallback sizes.
   * Returns true if successful, false if all attempts fail or max reached. */

  if (segments_locked) return false;

  // Check if we've reached maximum number of segments
  if (segments.size() >= max_segments) {
    DEBUGINFO("Reached maximum number of segments (", max_segments,
              "). Locking segments.");
    segments_locked = true;
    return false;
  }

  for (size_t attempt_size : fallback_sizes) {
    flex::DeviceMemoryAllocationPtr data;

    try {
      allocator->TryAllocate(&data, attempt_size, 0);
    }
    catch (const std::runtime_error& e) {
      // VF allocation failed - try next fallback size
      DEBUGINFO("TryAllocate failed for size", attempt_size, ":", e.what());
      continue;
    }

    if (data) {
      DEBUGINFO("Allocated new segment", segments.size() + 1, "of size",
                attempt_size, "bytes");
      segments.emplace_back(data->AllocIndex(), attempt_size);
      segments.back().data = data;

      // Initialize with one large free block covering entire segment
      segments.back().blocks.insert(MemoryBlock{0, attempt_size, true});
      segments.back().free_sizes.insert(attempt_size);

      // Check if we've now reached the maximum
      if (segments.size() >= max_segments) {
        DEBUGINFO("Reached maximum number of segments (", max_segments,
                  "). Locking segments.");
        segments_locked = true;
      }

      return true;
    }
  }

  // All allocation attempts failed - lock segments
  DEBUGINFO(
      "Failed to allocate new segment with all fallback sizes. Locking "
      "segments.");
  segments_locked = true;
  return false;
}

size_t VFSpyreAllocator::setMinSpyreAllocation(size_t nbytes) const {
  /* Adjust allocation according to Spyre requirement. */

  if (nbytes % MIN_ALLOC_BYTES != 0)
    return ((nbytes + MIN_ALLOC_BYTES - 1) / MIN_ALLOC_BYTES) *
           MIN_ALLOC_BYTES;
  return nbytes;
}

VFSpyreAllocator::AllocationInfo VFSpyreAllocator::findFreeBlock(
    size_t nbytes, flex::DeviceMemoryAllocatorPtr allocator) {
  /* Locate first memory block that can accommodate a block of size nbytes.
   * Until segments are locked, always attempt to allocate a new segment
   * first. Once locked, use load-balancing across existing segments. */

  // Check if requested size is reasonable for largest fallback size
  if (!fallback_sizes.empty()) {
    size_t max_segment_size = fallback_sizes[0];
    TORCH_CHECK(nbytes <= max_segment_size, "Requested allocation (", nbytes,
                " bytes) exceeds maximum segment size (", max_segment_size,
                " bytes)");
  }

  // If segments not locked, always try to allocate a new segment first
  if (!segments_locked) {
    if (allocateNewSegment(allocator)) {
      // Use the newly allocated segment (it's the last one)
      MemorySegment* new_seg = &segments.back();
      for (const MemoryBlock& r : new_seg->blocks) {
        if (r.is_free && r.size() >= nbytes) return {new_seg, r, true};
      }
    }
    // If allocation failed, segments are now locked, fall through to load
    // balancing
  }

  // Load-balancing: find segment with most free memory
  MemorySegment* best_seg = nullptr;
  size_t max_free_size = 0;

  for (MemorySegment& seg : segments) {
    if (seg.free_size < nbytes || seg.free_sizes.empty() ||
        *seg.free_sizes.rbegin() < nbytes)
      continue;

    // Track segment with most free memory
    if (seg.free_size > max_free_size) {
      max_free_size = seg.free_size;
      best_seg = &seg;
    }
  }

  if (best_seg == nullptr)
    return {nullptr, {}, false};  // not enough free memory

  // Find first-fit block in best segment
  for (const MemoryBlock& r : best_seg->blocks) {
    if (r.is_free && r.size() >= nbytes)
      return {best_seg, r, true};  // free block found
  }

  return {nullptr, {}, false};  // free block not found
}

MemoryBlock* VFSpyreAllocator::allocateInSegment(MemorySegment* seg,
                                                 MemoryBlock block,
                                                 size_t nbytes) {
  /* Given a predetermined Segment and a free memory block that accommodates
   * at least nbytes, mark this memory occupied, split block if needed, and
   * update total Segment free memory.
   */

  DEBUGINFO("VF block allocation");
  seg->blocks.erase(block);  // remove the free block

  // Update free sizes, removing a single value from it (not all)
  auto size_it = seg->free_sizes.find(block.size());
  seg->free_sizes.erase(size_it);

  // Insert occupied block
  MemoryBlock occupied{block.start, block.start + nbytes, false};
  auto [block_it, inserted] = seg->blocks.insert(occupied);

  // If there's remaining space, create a new free block
  if (block.size() > nbytes) {
    MemoryBlock remaining{block.start + nbytes, block.end, true};
    seg->blocks.insert(remaining);
    seg->free_sizes.insert(remaining.size());
  }

  seg->free_size -= nbytes;

  return const_cast<MemoryBlock*>(&(*block_it));
}

void VFSpyreAllocator::deallocateBlock(MemorySegment& seg,
                                       SharedOwnerCtx* ctx) {
  /* Deallocate a block from a segment and return its memory to the free pool.
   * Merges adjacent free blocks to reduce fragmentation. Updates segment's
   * free memory tracking and removes block from registry.
   */

  auto ctx_it = seg.ctx_to_block.find(ctx);
  if (ctx_it == seg.ctx_to_block.end()) return;

  DEBUGINFO("VF block deallocation");
  MemoryBlock* occupied_block = ctx_it->second;
  size_t freed_start = occupied_block->start;
  size_t freed_end = occupied_block->end;
  size_t freed_size = freed_end - freed_start;

  // Find the occupied block in the set and remove it
  auto block_it = seg.blocks.find(*occupied_block);
  if (block_it == seg.blocks.end()) return;
  seg.blocks.erase(block_it);

  // Merge with previous free block if adjacent
  // 1. find block (as iterator) *at or beyond* the position that was freed
  // 2. move to pointer to previous block (blocks are ordered by offset)
  // 3. if selected block is free and adjacent to freed block:
  //    - move new starting offset at the start of selected block
  //    - remove selected block size from free_sizes
  //    - remove selected block from set of all blocks
  auto freed_pos =
      seg.blocks.lower_bound(MemoryBlock{freed_start, freed_end, false});
  if (freed_pos != seg.blocks.begin()) {
    auto prev_it = std::prev(freed_pos);
    if (prev_it->is_free && prev_it->end == freed_start) {
      freed_start = prev_it->start;

      // iterator-based erasure to remove a single value
      auto prev_size_it = seg.free_sizes.find(prev_it->size());
      seg.free_sizes.erase(prev_size_it);

      seg.blocks.erase(prev_it);
    }
  }

  // Merge with next free block if adjacent
  // 1. reusing iterator that points to block *at or beyond* the freed one
  // 2. if selected block is free and adjacent to freed block:
  //    - move new ending offset at the _start_ of selected block
  //    - remove selected block size from free_sizes
  //    - remove selected block from set of all blocks
  if (freed_pos != seg.blocks.end() && freed_pos->is_free &&
      freed_pos->start == freed_end) {
    freed_end = freed_pos->end;
    auto next_size_it = seg.free_sizes.find(freed_pos->size());
    seg.free_sizes.erase(next_size_it);
    seg.blocks.erase(freed_pos);
  }

  // Insert the merged free block (with updated start/end)
  MemoryBlock new_free{freed_start, freed_end, true};
  seg.blocks.insert(new_free);
  seg.free_sizes.insert(new_free.size());
  seg.free_size += freed_size;
  seg.ctx_to_block.erase(ctx_it);
}

void VFSpyreAllocator::logSegmentState(const MemorySegment& seg,
                                       const char* context,
                                       bool include_blocks) {
  /* Log free and used memory in the specified Segment. */

  DEBUGINFO(context, "seg id", seg.segment_id, "free mem", seg.free_size);

  if (include_blocks) {
    // ctx_to_block only tracks *occupied* blocks (via their pointer)
    // and the corresponding SharedOwnerCtx pointer
    for (const auto& [soc_ptr, block_ptr] : seg.ctx_to_block) {
      DEBUGINFO("    occupied block: [", block_ptr->start, ",",
                block_ptr->end, ") size:", block_ptr->size(),
                "ctx:", soc_ptr);
    }

    // seg.blocks includes both free and occupied blocks
    for (const MemoryBlock& block : seg.blocks) {
      if (block.is_free) {
        DEBUGINFO("  free block: [", block.start, ",", block.end,
                  ") size:", block.size());
      }
    }
  }

  for (const size_t& sz : seg.free_sizes) DEBUGINFO("  free sz", sz);
}

void VFSpyreAllocator::logAllSegments(const char* context,
                                      bool include_blocks) {
  /* Log free and used memory of all Segments. */

  DEBUGINFO(context);
  for (const MemorySegment& seg : segments) {
    logSegmentState(seg, "", include_blocks);
  }
}

VFSpyreAllocator::VFSpyreAllocator(size_t max_seg)
    : SpyreAllocator(), segments_locked(false), max_segments(max_seg) {
  // Initialize fallback sizes: 12GB, 8GB, 4GB
  // NOTE: size selection to be defined
  fallback_sizes = {12ULL * 1024 * 1024 * 1024, 8ULL * 1024 * 1024 * 1024,
                    4ULL * 1024 * 1024 * 1024};
  instance_ptr.store(this, std::memory_order_release);  // atomic write
}

VFSpyreAllocator::~VFSpyreAllocator() {
  instance_ptr.store(nullptr, std::memory_order_release);  // atomic write
}

at::DataPtr VFSpyreAllocator::allocate(size_t nbytes) {
  /* VF allocation implementation with dynamic segment allocation.
   * Segments are allocated on-demand as needed. When a new segment cannot
   * be allocated, the vectors of segments is locked and the allocator
   * assigns blocks using load-balancing logic across existing segments,
   * selecting the one with most free memory, and first-fit logic within
   * each segment, occupying the first (= lowest offset) free block that
   * can fit the requested allocation.
   */

  c10::Device curr_device =
      c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
          ->getDevice();
  auto device_id = curr_device.index();
  DEBUGINFO("allocating", nbytes, "bytes on Spyre", curr_device, "(VF mode)");
  if (nbytes <= 0) return {nullptr, nullptr, &ReportAndDelete, curr_device};

  auto allocator = getAllocator(device_id);

  std::lock_guard<std::mutex> lock(allocator_mutex);

  size_t aligned_nbytes = setMinSpyreAllocation(nbytes);
  AllocationInfo alloc_info = findFreeBlock(aligned_nbytes, allocator);

  TORCH_CHECK(
      alloc_info.found, "Unable to find enough free memory for allocation. ",
      segments_locked
          ? "All segments are full and no new segments could be allocated."
          : "Failed to allocate memory.");

  MemoryBlock* new_block =
      allocateInSegment(alloc_info.segment, alloc_info.block, aligned_nbytes);

  flex::DeviceMemoryAllocationPtr data = alloc_info.segment->data;
  TORCH_CHECK(data, "Failed to allocate ", aligned_nbytes,
              " bytes on Spyre device.");

  if (alloc_debug)
    logSegmentState(*alloc_info.segment, "After block allocation");

  // Instantiating object to live beyond SpyreAllocator scope.
  // Note: changed to share the data pointer, not move it. This is needed
  // so that the segment retains its reference when a block is freed.
  // With std::move(data), the segment would lose its reference to the
  // Spyre allocation after the first block allocation
  auto* ctx = new SharedOwnerCtx{data, new_block->start, device_id};
  void* ctx_void = static_cast<void*>(ctx);
  void* data_void = static_cast<void*>(ctx->owner.get());

  alloc_info.segment->ctx_to_block[ctx] = const_cast<MemoryBlock*>(new_block);
  block_to_segment[ctx] = alloc_info.segment;

  return at::DataPtr(data_void, ctx_void, &ReportAndDelete, curr_device);
}

// ============================================================================
// Factory Method Implementation
// ============================================================================

SpyreAllocator& SpyreAllocator::instance() {
  static std::unique_ptr<SpyreAllocator> allocator;
  static std::once_flag init_flag;

  std::call_once(init_flag, [&allocator]() {
    if (is_pf_mode()) {
      allocator.reset(new PFSpyreAllocator());
    } else {
      allocator.reset(new VFSpyreAllocator());
    }
  });

  return *allocator;
}

}  // namespace spyre

// Register our custom allocator
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &spyre::SpyreAllocator::instance());
