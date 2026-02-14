---
description: Rules and guidelines for developing and fixing the USD parser in Genesis
---

# USD Parser Rules

## Overview

This document describes the rules and guidelines for developing and fixing the USD parser in Genesis.

**Before starting:** Read the [USD Parser Specification](../../genesis/utils/usd/UsdParserSpec.md) to understand the mathematical foundations, transform handling, and architectural constraints.

## Core Rules

### 1. Scope Restriction: No Simulator-Side Modifications

**CRITICAL:** You MUST NOT modify any code outside the `genesis/utils/usd` directory.

- **Allowed:** Changes to files in `genesis/utils/usd/` (e.g., `usd_rigid_entity.py`, `usd_utils.py`, `usd_geometry.py`)
- **Forbidden:** Changes to simulator code, entity definitions, or any code outside the USD parser directory
- **Rationale:** The USD parser is an isolated utility that converts USD files to Genesis format without affecting core simulator functionality

### 2. USD Schema Research: Always Check pxr-stubs First

**Before implementing parsing logic for any USD attribute:**

1. **Locate pxr-stubs:** Search in the `pxr-stubs` package for the relevant schema definition
   - Example: `UsdPhysics` attributes are in the `UsdPhysics` module of `pxr-stubs`
   
2. **Verify attribute properties:**
   - Default values (if any)
   - Whether the attribute is optional or required
   - Valid value ranges or types
   - Whether `HasValue()` or `IsValid()` checks are needed

3. **Example pattern:** See how existing code handles attributes:
   ```python
   # Check if attribute exists and has a value
   attr = prim.GetAttribute("someAttr")
   value = attr.Get() if attr.HasValue() else default_value
   ```

### 3. External Research: Search When pxr-stubs Are Insufficient

**If information is missing from pxr-stubs:**

1. Search the internet for USD schema documentation
2. Check official USD documentation: https://openusd.org/
3. Look for USD Physics schema references: https://openusd.org/release/api/usd_physics_page_front.html
4. Review existing parser code for similar patterns

### 4. Code Organization: Use Helper Functions for Complex Logic

**When parsing logic becomes complex:**

- **Extract helper functions** to `usd_utils.py` or create module-specific helpers
- **Keep main parsing functions** (e.g., in `usd_rigid_entity.py`) clean and readable
- **Follow existing patterns:** See `usd_utils.py` for examples like `usd_pos_to_numpy()`, `usd_quat_to_numpy()`, etc.

**Example structure:**
```python
# In usd_rigid_entity.py (main parsing logic)
def parse_entity(...):
    # High-level flow
    transform = _get_rigid_transform(prim)
    geometry = parse_prim_geoms(context, prim)
    # ...

# In usd_utils.py (helper functions)
def _get_rigid_transform(prim: Usd.Prim) -> np.ndarray:
    # Complex transform decomposition logic
    # ...
```

## Key Files Reference

- **Specification:** [UsdParserSpec.md](../../genesis/utils/usd/UsdParserSpec.md)
- **Main parser:** [usd_rigid_entity.py](../../genesis/utils/usd/usd_rigid_entity.py)
- **Utilities:** [usd_utils.py](../../genesis/utils/usd/usd_utils.py)
- **Geometry parsing:** [usd_geometry.py](../../genesis/utils/usd/usd_geometry.py)
- **Context management:** [usd_context.py](../../genesis/utils/usd/usd_context.py)
