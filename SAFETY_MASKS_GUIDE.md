# Safety Mask Functions in SingleAgentDIEnv

## Overview

The `SingleAgentDIEnv` wrapper now exposes all safety checking functions from the underlying `DoubleIntegrator` environment. This allows you to check safety properties for single-agent scenarios using the same robust algorithms.

## Available Safety Functions

### 1. **`safe_mask(state=None)`**
Checks if agent is in a **safe** state with sufficient clearance.

```python
is_safe = env.safe_mask()  # Check current state
is_safe = env.safe_mask(custom_state)  # Check custom state
```

**Safe means:**
- Agent maintains clearance of **2 × car_radius** from all obstacles
- Returns: `bool` (True if safe)

---

### 2. **`unsafe_mask(state=None)`**
Checks if agent is in an **unsafe** state (collision or dangerous heading).

```python
is_unsafe = env.unsafe_mask()  # Check current state
is_unsafe = env.unsafe_mask(custom_state)  # Check custom state
```

**Unsafe means EITHER:**
1. Currently colliding with obstacle
2. Heading toward obstacle within danger cone (predictive)

Uses the detailed LiDAR-based heading detection algorithm.

Returns: `bool` (True if unsafe)

---

### 3. **`collision_mask(state=None)`**
Checks if agent is **currently in collision**.

```python
is_colliding = env.collision_mask()  # Check current state
is_colliding = env.collision_mask(custom_state)  # Check custom state
```

**Collision means:**
- Agent center is within **car_radius** of obstacle boundary (physically touching)

Returns: `bool` (True if colliding)

---

### 4. **`finish_mask(state=None)`**
Checks if agent has **reached its goal**.

```python
reached_goal = env.finish_mask()  # Check current state
reached_goal = env.finish_mask(custom_state)  # Check custom state
```

**Goal reached means:**
- Distance from agent to goal < **2 × car_radius**

Returns: `bool` (True if goal reached)

---

### 5. **`get_cost(state=None)`**
Gets the **collision cost** value.

```python
cost = env.get_cost()  # Check current state
cost = env.get_cost(custom_state)  # Check custom state
```

**Cost:**
- `0.0` if no collision
- `1.0` if collision (for single agent)

Returns: `float`

---

## Trajectory Safety Analysis

### 6. **`check_trajectory_safety(states, check_type)`**
Check safety masks for an entire trajectory.

```python
states = np.array([[x1,y1,vx1,vy1], [x2,y2,vx2,vy2], ...])  # [T, 4]

# Check which states are safe
safe_flags = env.check_trajectory_safety(states, "safe")

# Check which states are unsafe
unsafe_flags = env.check_trajectory_safety(states, "unsafe")

# Check which states have collisions
collision_flags = env.check_trajectory_safety(states, "collision")

# Check which states reached goal
finish_flags = env.check_trajectory_safety(states, "finish")
```

**Args:**
- `states`: Trajectory [T, 4] or single state [4]
- `check_type`: One of `"safe"`, `"unsafe"`, `"collision"`, `"finish"`

**Returns:** `np.ndarray[bool]` of shape [T]

---

### 7. **`is_trajectory_safe(states, tolerance)`**
Check if entire trajectory satisfies safety requirement.

```python
states = np.array([[...], [...], ...])  # [T, 4]

# Strict: all states must be safe
is_safe = env.is_trajectory_safe(states, "strict")

# Collision-free: no actual collisions (predictive unsafe allowed)
is_safe = env.is_trajectory_safe(states, "collision_free")
```

**Args:**
- `states`: Trajectory [T, 4]
- `tolerance`: `"strict"` or `"collision_free"`

**Returns:** `bool`

---

### 8. **`get_first_unsafe_index(states)`**
Find the first timestep where trajectory becomes unsafe.

```python
states = np.array([[...], [...], ...])  # [T, 4]

first_unsafe = env.get_first_unsafe_index(states)
# Returns: int (index) or None if all safe
```

---

## Usage Examples

### Example 1: Check Current State
```python
from gcbfplus.env.double_integrator import DoubleIntegrator
from gcbfplus.env.sa_di_wrapper import SingleAgentDIEnv

# Setup
env_m = DoubleIntegrator(num_agents=1, area_size=6.0)
env = SingleAgentDIEnv(env_m)
obs = env.reset(seed=42)

# Check safety
print(f"Safe: {env.safe_mask()}")
print(f"Unsafe: {env.unsafe_mask()}")
print(f"Collision: {env.collision_mask()}")
print(f"Reached goal: {env.finish_mask()}")
print(f"Cost: {env.get_cost()}")
```

### Example 2: Check Custom State
```python
# Check if a specific state would be safe
test_state = np.array([3.0, 3.0, 0.1, 0.1])  # [x, y, vx, vy]

if env.safe_mask(test_state):
    print("This state is safe!")
else:
    print("This state is NOT safe")
```

### Example 3: Analyze Trajectory
```python
# Collect trajectory
states = []
for step in range(100):
    states.append(env.get_state())
    action = policy.get_action(obs)
    obs, reward, done, info = env.step(action)

states = np.array(states)  # [T, 4]

# Analyze
safe_flags = env.check_trajectory_safety(states, "unsafe")
print(f"Unsafe states: {np.sum(safe_flags)} / {len(states)}")

if env.is_trajectory_safe(states, "collision_free"):
    print("Trajectory is collision-free!")

first_unsafe = env.get_first_unsafe_index(states)
if first_unsafe is not None:
    print(f"First became unsafe at step {first_unsafe}")
```

### Example 4: Predictive Safety (MPC/Planning)
```python
# Check if a planned action sequence is safe
planned_actions = np.random.randn(10, 2)  # 10 steps

# Rollout in simulation
states, rewards, infos = env.rollout_sequence(planned_actions)

# Check safety
if env.is_trajectory_safe(states, "strict"):
    print("Plan is safe! Executing...")
    # Execute first action
else:
    first_unsafe = env.get_first_unsafe_index(states)
    print(f"Plan becomes unsafe at step {first_unsafe}")
    print("Replanning...")
```

---

## Key Differences Between Masks

| Function | Strictness | Use Case |
|----------|-----------|----------|
| `safe_mask()` | Most strict | Verify sufficient clearance maintained |
| `unsafe_mask()` | Medium | Predictive - heading toward danger? |
| `collision_mask()` | Least strict | Physical contact only |
| `finish_mask()` | N/A | Task completion |

**Safety hierarchy:**
```
safe_mask = False  →  unsafe_mask might be True
unsafe_mask = True  →  collision_mask might be True
collision_mask = True  →  definitely not safe
```

---

## Integration with DoubleIntegrator

All these functions call the underlying `DoubleIntegrator` methods:
- Uses the same LiDAR-based obstacle detection
- Same danger cone geometry calculations
- Same clearance thresholds
- Fully compatible with multi-agent algorithms (just using single agent)

The wrapper handles:
- Converting single state to graph representation
- Extracting single-agent result from array
- Type conversions (JAX → NumPy)
