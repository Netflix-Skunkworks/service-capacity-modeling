# Ralph Status: Review Cass Memory Model

## Current Phase: Phase 2 - Verify working set formula
## Status: COMPLETE

## Phase 2 Findings

### Working set formula verification — ALL CORRECT

| Check | Expected | Actual (cassandra.py) | Status |
|---|---|---|---|
| Observed path uses raw disk | `current_capacity.disk_utilization_gib.mid` | Line 306: `raw_disk_per_node = current_capacity.disk_utilization_gib.mid` | CORRECT |
| Fallback when disk_util=0 | `disk_used_gib / node_count` | Lines 307-310: `raw_disk_per_node = disk_used_gib / cluster_instance_count.mid` | CORRECT |
| Negative page cache guard | `max(0, ...)` | Line 303: `page_cache_per_node = max(0, ram_gib - memory_utilization_gib)` | CORRECT |
| Working set cap at 1.0 | `min(1.0, ...)` | Line 312 (observed): `min(1.0, page_cache / raw_disk)` | CORRECT |
| Division-by-zero guard | `max(1, ...)` | Line 311: `raw_disk_per_node = max(1, raw_disk_per_node)` | CORRECT |

### Theoretical fallback (line 315)
- `min(working_set, rps_working_set)` — both inputs bounded:
  - `working_set` from `working_set_from_drive_and_slo()` → CDF output ∈ [0, 1]
  - `rps_working_set = min(1.0, disk_rps / max_rps_to_disk)` (line 262)
- Result always ∈ [0, 1] ✅

## Phase 1 Findings

### Buffer integration verification — ALL CORRECT

| Buffer Intent | Expected Result | Actual Behavior | Status |
|---|---|---|---|
| No buffer | `base_needed_memory` | `scale(1) * base_needed_memory * 1.0` | CORRECT |
| preserve | `existing_page_cache` | Early return `existing_capacity` | CORRECT |
| scale_up (floor=1) | `max(base_needed_memory, existing_page_cache)` | `max(requirement, 1 * existing_capacity)` | CORRECT |
| scale_down (ceiling=1) | `min(base_needed_memory, existing_page_cache)` | `min(requirement, 1 * existing_capacity)` | CORRECT |

### Pattern consistency with CPU
- Cassandra memory follows same `DerivedBuffers.calculate_requirement(current_usage, existing_capacity)` pattern as `RequirementFromCurrentCapacity.cpu()` (common.py:942)
- `desired_buffer_ratio` intentionally omitted (working set already captures memory need)
- Fallback path (no current_capacity) correctly skips buffer policy

## Progress
- [x] Phase 1: Verify buffer integration correctness
- [x] Phase 2: Verify working set formula
- [ ] Phase 3: Verify write_buffer_gib behavior
- [ ] Phase 4: Run tests
- [ ] Phase 5: Check for remaining issues
