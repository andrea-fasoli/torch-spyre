# SpyreAllocator VF Mode: Design Summary

## Overview

`SpyreAllocator` is a custom memory allocator managing device memory allocation for tensors that execute on Spyre hardware, supporting both Physical Function (PF) and Virtual Function (VF) modes.

## Core Functionalities

### 1. Dual-Mode Memory Management

The new implementation adds VF Mode support, allowing the allocator to operate in two distinct modes. Mode selection is based on the `FLEX_DEVICE` environment variable. The selection takes place once, at the time `SpyreAllocator` is constructed.

#### **PF (Physical Function) Mode**
- This mode has been left mostly functionally-unaltered from previous implementations. The only I/O difference is a new `vf_offset` attribute in SharedOwnerCtx. `vf_offset` is set to 0 for PF Mode and should be ignored
- **Functionalities**
   - Each tensor allocation directly requests memory from Spyre hardware via `TryAllocate()`
   - Allocation on demand, deallocation when tensor is destroyed
   - Each allocation is independent

#### **VF (Virtual Function) Mode**
Implemented in PR #355, updated with adaptive allocation strategy
- **Functionalities**
   - Allocates memory segments on-demand (one at a time) as allocation requests arrive
   - Uses fallback sizes (12GB → 8GB → 4GB) when primary allocation fails
   - Locks segment vector when allocation fails or maximum segments reached (default: 12 segments)
   - After locking, uses load-balancing across existing segments
   - Divides segments into blocks for individual tensor allocations
   - Freed blocks return to the pool for future allocations
   - Merges adjacent free blocks to reduce fragmentation
- **Additional details**
   - Called via `allocate` method (a request for `nbytes` allocation)
   - Returns `at::DataPtr` smart pointer comprising raw memory buffer, context/metadata pointer needed by the deleter, cleanup function, and device identifier
   - Context is `SharedOwnerCtx` object, augmented with `vf_offset` (block offset within segment)
   - All blocks within the same segment share the same `flex::DeviceMemoryAllocationPtr data` pointer
   - Segments are allocated via `TryAllocate()` with fallback size handling in `allocateNewSegment()`

### 2. Memory Segment Architecture (VF Mode)

**MemorySegment Structure:**
- Represents one contiguous allocation on Spyre hardware
- Tracks total size, free size, and allocation index
- Maintains ordered set of `MemoryBlock` objects (both free and occupied)
- Uses `std::multiset` to track free block sizes for efficient lookup
- Maps `SharedOwnerCtx` pointers to blocks for O(1) deallocation

**MemoryBlock Structure:**
- Represents contiguous memory intervals within a segment, either free or occupied
- Stores start/end offsets and free/occupied status
- Ordered by start offset for efficient merging operations

### 3. Allocation Strategy (VF Mode)

**Adaptive Segment Allocation with Fallback Sizes:**

The allocator now uses an adaptive on-demand allocation strategy instead of pre-allocating all segments upfront:

1. **On-Demand Segment Allocation**: When `allocate()` is called and segments are not yet locked:
   - Attempts to allocate a new segment first (before checking existing segments)
   - Tries fallback sizes in order: 12GB → 8GB → 4GB
   - If allocation succeeds, uses the newly created segment
   - If all fallback sizes fail, locks the segment vector and switches to load-balancing mode

2. **Segment Locking**: Once the allocator cannot allocate a new segment (either due to allocation failure or reaching `max_segments`), it locks the segment vector and transitions to load-balancing across existing segments

3. **Load-Balanced Block Assignment** (after segments are locked):
   - **Segment selection**: Choose segment with most free memory
   - **Block selection**: Find first free block (lowest offset) that fits the request
   - **Block splitting**: If selected block is larger than needed, split into occupied + free blocks

4. **Alignment**: All allocations aligned to 128-byte boundaries (Spyre hardware requirement)

### 4. VF Mode Implementation Details

**Key Data Structures:**
- `segments`: Vector of `MemorySegment` objects, grows dynamically as segments are allocated
- `block_to_segment`: Hash map for O(1) lookup from `SharedOwnerCtx*` to its containing segment
- `segments_locked`: Boolean flag indicating whether new segment allocation is disabled
- `fallback_sizes`: Vector of fallback sizes to try when allocating new segments (12GB, 8GB, 4GB)
- `max_segments`: Maximum number of segments allowed (default: 12, defined as `MAX_SEGMENTS`)

**Allocation Flow (`allocate` method):**
1. Acquire mutex lock for thread safety
2. Align requested bytes to 128-byte boundary
3. Call `findFreeBlock()` to locate suitable memory:
   - If segments not locked: attempt to allocate new segment via `allocateNewSegment()`
   - If new segment allocated successfully: use it immediately
   - If allocation fails or segments locked: use load-balancing to find best existing segment
4. Call `allocateInSegment()` to reserve the block:
   - Remove free block from segment's block set
   - Create occupied block for requested size
   - Split remaining space into new free block if needed
   - Update segment's free memory tracking
5. Register block in `block_to_segment` map for fast deallocation
6. Return `at::DataPtr` with shared `flex::DeviceMemoryAllocationPtr` and block offset

**Segment Allocation (`allocateNewSegment` method):**
1. Check if segments are already locked or max reached → return false
2. Iterate through fallback sizes (12GB, 8GB, 4GB):
   - Call `TryAllocate()` with current size
   - If successful: create new `MemorySegment`, initialize with one large free block, return true
   - If failed: try next fallback size
3. If all sizes fail: lock segments and return false

### 5. Deallocation and Defragmentation (VF Mode)

**Deallocation Flow (`ReportAndDelete` static method):**
1. Acquire mutex lock for thread safety
2. Look up segment containing the block via `block_to_segment` map (O(1))
3. Call `deallocateBlock()` to free the memory:
   - Locate the occupied block in segment's block set
   - Remove occupied block
   - Merge with previous adjacent free block if exists
   - Merge with next adjacent free block if exists
   - Insert merged free block back into segment
   - Update segment's free memory tracking and free sizes multiset
4. Remove entry from `block_to_segment` map
5. Delete `SharedOwnerCtx` object

**Coalescing Strategy:**
- When a block is freed, automatically merges with adjacent free blocks (both previous and next)
- Uses `std::set::lower_bound()` iterator for efficient neighbor lookup
- Coalescing reduces fragmentation by creating larger contiguous free blocks
- Updates free block sizes multiset and segment metadata
- Thread-safe via mutex protection
- No block relocation is performed (blocks stay at their original offsets)

## VF Mode Design Considerations

### Adaptive Segment Allocation Strategy

**Adaptive On-demand Implementation:**
- Segments are allocated on-demand, one at a time, as memory requests arrive
- When a new segment cannot be allocated (allocation fails or `max_segments` reached), the allocator locks the segment vector and switches to load-balancing mode
- This approach is more flexible and multi-tenant friendly compared to pre-allocating all segments upfront

**Fallback Sizes:**
- Segment max size and fallbacks are hardcoded: 12, 8, 4 GB. This is an empyrical choice and should be discussed.
- If all fallback sizes fail, segments are locked and no further allocation attempts are made
- In multi-tenant scenarios, we could consider unlocking the segments vector under certain conditions (to be decided) to re-attempt expanding the segment vector.

**Spyre Memory Coverage:**
- With 12 maximum segments of at most 12 GB, theoretical maximum is 144 GB.
- In practice, observed allocation of 9 x 12 GB = 108 GB. All fallbacks failed at the next requests. Fundamental reason for this behavior remains unclear.
- Alternatively, upon increasing max segment size to 14 GB (with 12, 8, 4 GB fallbacks), observed allocation of 7 x 14 + 1 x 12 GB = 110 GB.
- Finally, upon increasing max segment size to 16 GB (with 12, 8, 4 GB fallbacks), observed allocation of 6 x 16 + 1 x 12 GB = 108 GB.

### Alignment to 128 bytes

- Supposedly, a Spyre hardware requirement (value unconfirmed)
- `nbytes` alignment implemented in `setMinSpyreAllocation`
- Unclear if `allocate` can ever receive an `nbytes` request that doesn't satisfy this requirement

### Extensibility

- Further updating the allocation design only needs to alter `SpyreAllocator` (in `spyre_mem.cpp`) and core structures `MemoryBlock`, `MemorySegment` (in `spyre_allocator.h`), and `SharedOwnerCtx` (in `module.h`)
- Current VF Mode implementation preserves `allocate` I/O interface used by PF Mode
- Key extension points:
  - `allocateNewSegment()`: Modify fallback size logic or allocation strategy
  - `findFreeBlock()`: Change segment selection or block search algorithms
  - `allocateInSegment()`: Adjust block splitting or alignment logic
  - `deallocateBlock()`: Modify coalescing or defragmentation behavior

## Design Strengths and Limitations

### Strengths

1. **Dual-mode support**: Supports both PF (preserved) and VF allocation strategies
2. **Adaptive allocation**: On-demand segment allocation is more flexible and multi-tenant friendly than pre-allocation
3. **Fallback mechanism**: Gracefully handles allocation failures by trying smaller segment sizes (12GB → 8GB → 4GB)
4. **Load balancing**: After segments are locked, distributes allocations across segments based on available memory
5. **Predictable allocation**: First-fit strategy within best segment provides deterministic behavior
6. **Defragmentation**: Reduces fragmentation by coalescing adjacent freed memory blocks
7. **Thread-safe**: Mutex protection ensures correctness in multi-threaded environments
8. **Debugging support**: Optional debug logging via `TORCH_SPYRE_ALLOC_DEBUG` environment variable
9. **Efficient lookups**: Uses `block_to_segment` map for O(1) deallocation lookups

### Limitations

1. **Fixed maximum segments**: Hard limit on number of segments (default 8) may be restrictive in some scenarios
2. **No segment shrinking**: Once allocated, segments remain allocated for the lifetime of the allocator
3. **First-fit vs best-fit**: First-fit block assignment could be improved to best-fit for better space utilization
4. **Internal fragmentation**: Freed blocks are merged but not relocated, potentially leading to fragmentation over time despite coalescing; relocation overhead may be substantial though
5. **Single allocator instance**: Single SpyreAllocator instance for all devices may need rework in multi-device scenarios
6. **Synchronous operations**: All allocations/deallocations are blocking; could lead to delays under high allocation rate; may be possible to improve by limiting mutex scope
7. **OOM handling**: Limited handling of out-of-memory errors beyond TORCH_CHECK validation; no retry or recovery mechanisms
8. **Binary logging**: Logging is all-or-nothing; finer-grained logging control would be beneficial
9. **Fallback size selection**: Fallback sizes (12GB, 8GB, 4GB) are hardcoded; could benefit from configuration options
