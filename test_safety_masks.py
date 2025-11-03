"""
Test script demonstrating safety mask functions in SingleAgentDIEnv wrapper.

This shows how to use:
- safe_mask()
- unsafe_mask()
- collision_mask()
- finish_mask()
- check_trajectory_safety()
- is_trajectory_safe()
"""

import numpy as np
from gcbfplus.env.double_integrator import DoubleIntegrator
from gcbfplus.env.sa_di_wrapper import SingleAgentDIEnv


def main():
    # Create single-agent double integrator environment
    print("=" * 60)
    print("Creating Single-Agent Double Integrator Environment")
    print("=" * 60)
    
    env_m = DoubleIntegrator(
        num_agents=1,
        area_size=6.0,
        max_step=256,
        dt=0.03,
    )
    
    env = SingleAgentDIEnv(env_m, include_velocity=True, normalize_lidar=True)
    
    # Reset environment
    obs = env.reset(seed=42)
    
    print(f"\nEnvironment Parameters:")
    print(f"  Car radius: {env.car_radius}")
    print(f"  Comm radius: {env.comm_radius}")
    print(f"  Number of LiDAR rays: {env.n_rays}")
    print(f"  Time step: {env.dt}")
    print(f"  Area size: {env.area_size}")
    
    print(f"\nInitial State:")
    print(f"  Agent position: {obs['state'][:2]}")
    print(f"  Agent velocity: {obs['state'][2:]}")
    print(f"  Goal position: {obs['goal'][:2]}")
    print(f"  Distance to goal: {np.linalg.norm(obs['goal'][:2] - obs['state'][:2]):.3f}")
    
    # Check initial safety
    print("\n" + "=" * 60)
    print("Initial Safety Checks")
    print("=" * 60)
    
    is_safe = env.safe_mask()
    is_unsafe = env.unsafe_mask()
    is_collision = env.collision_mask()
    is_finished = env.finish_mask()
    cost = env.get_cost()
    
    print(f"  Safe mask: {is_safe} (clearance maintained)")
    print(f"  Unsafe mask: {is_unsafe} (collision or dangerous heading)")
    print(f"  Collision mask: {is_collision} (physically touching obstacle)")
    print(f"  Finish mask: {is_finished} (reached goal)")
    print(f"  Collision cost: {cost:.3f} (0.0 = no collision)")
    
    # Take some random actions and track safety
    print("\n" + "=" * 60)
    print("Taking Random Actions and Tracking Safety")
    print("=" * 60)
    
    states = [env.get_state()]
    actions = []
    
    for step in range(10):
        # Random action
        action = np.random.uniform(env.action_low, env.action_high)
        actions.append(action)
        
        obs, reward, done, info = env.step(action)
        states.append(env.get_state())
        
        # Check safety at this step
        is_safe = env.safe_mask()
        is_unsafe = env.unsafe_mask()
        is_collision = env.collision_mask()
        
        status = "✓ SAFE" if is_safe else "⚠ UNSAFE" if is_unsafe else "✗ COLLISION" if is_collision else "?"
        
        print(f"  Step {step+1}: {status} | Pos: [{obs['state'][0]:.2f}, {obs['state'][1]:.2f}] | Reward: {reward:.3f}")
        
        if is_collision:
            print(f"    ⚠️ COLLISION DETECTED!")
            break
    
    # Analyze trajectory safety
    print("\n" + "=" * 60)
    print("Trajectory Safety Analysis")
    print("=" * 60)
    
    states_array = np.array(states)
    
    # Check different safety criteria
    safe_flags = env.check_trajectory_safety(states_array, "safe")
    unsafe_flags = env.check_trajectory_safety(states_array, "unsafe")
    collision_flags = env.check_trajectory_safety(states_array, "collision")
    
    print(f"\nTrajectory length: {len(states_array)} states")
    print(f"  Safe states: {np.sum(safe_flags)} / {len(states_array)}")
    print(f"  Unsafe states: {np.sum(unsafe_flags)} / {len(states_array)}")
    print(f"  Collision states: {np.sum(collision_flags)} / {len(states_array)}")
    
    # Overall safety
    is_strictly_safe = env.is_trajectory_safe(states_array, "strict")
    is_collision_free = env.is_trajectory_safe(states_array, "collision_free")
    
    print(f"\nOverall trajectory safety:")
    print(f"  Strictly safe (no unsafe states): {is_strictly_safe}")
    print(f"  Collision-free (no collisions): {is_collision_free}")
    
    # Find first unsafe point
    first_unsafe = env.get_first_unsafe_index(states_array)
    if first_unsafe is not None:
        print(f"  First unsafe state at index: {first_unsafe}")
    else:
        print(f"  No unsafe states detected!")
    
    # Test custom state checking
    print("\n" + "=" * 60)
    print("Custom State Safety Checking")
    print("=" * 60)
    
    # Create some test states
    test_states = [
        np.array([3.0, 3.0, 0.0, 0.0]),  # Middle of arena, stationary
        np.array([0.1, 0.1, 0.2, 0.2]),  # Near corner, moving
        np.array([5.8, 5.8, 0.0, 0.0]),  # Near edge
    ]
    
    for i, state in enumerate(test_states):
        print(f"\nTest state {i+1}: pos=[{state[0]:.1f}, {state[1]:.1f}], vel=[{state[2]:.1f}, {state[3]:.1f}]")
        print(f"  Safe: {env.safe_mask(state)}")
        print(f"  Unsafe: {env.unsafe_mask(state)}")
        print(f"  Collision: {env.collision_mask(state)}")
        print(f"  Cost: {env.get_cost(state):.3f}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
