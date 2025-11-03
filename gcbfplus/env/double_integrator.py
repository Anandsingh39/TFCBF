"""
Double Integrator Multi-Agent Environment

This module implements a 2D planar environment where multiple agents with double integrator
dynamics navigate to goals while avoiding static obstacles and each other. The environment
uses a graph neural network representation with LiDAR-based obstacle sensing.

Physics Model:
    State: [x, y, vx, vy] - position and velocity in 2D
    Dynamics: ẍ = F/m, ÿ = F/m (force inputs divided by mass)
    
Graph Representation:
    - Agent nodes: current state
    - Goal nodes: target positions
    - Obstacle nodes: LiDAR hit points on obstacles
    - Edges: relative state differences between connected nodes
"""

import functools as ft
import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import sys
from typing import NamedTuple, Tuple, Optional
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, AgentState, Array, Cost, Done, Info, Reward, State
from ..utils.utils import merge01, jax_vmap
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Rectangle
from .plot import render_video
from .utils import get_lidar, inside_obstacles, lqr, get_node_goal_rng


class DoubleIntegrator(MultiAgentEnv):
    """
    Multi-agent environment with double integrator dynamics.
    
    This class implements a 2D navigation environment where agents must reach their goals
    while avoiding collisions with obstacles and other agents. The environment uses:
    - Graph neural network representation for agent interactions
    - LiDAR-based obstacle detection
    - LQR controller for reference actions
    - Safety masks based on collision zones and heading directions
    """
    # Node type identifiers for graph representation
    AGENT = 0  # Agent nodes in the graph
    GOAL = 1   # Goal nodes in the graph
    OBS = 2    # Obstacle (LiDAR hit) nodes in the graph

    class EnvState(NamedTuple):
        """
        Environment state container.
        
        Attributes:
            agent: Current state of all agents [num_agents, 4] where each row is [x, y, vx, vy]
            goal: Goal state for all agents [num_agents, 4]
            obstacle: Collection of rectangular obstacles in the environment
        """
        agent: AgentState
        goal: State
        obstacle: Obstacle

        @property
        def n_agent(self) -> int:
            """Returns the number of agents in the environment."""
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    # Default environment parameters
    PARAMS = {
        "car_radius": 0.05,        # Radius of each agent (for collision detection)
        "comm_radius": 0.5,        # Communication/sensing radius for agent interactions
        "n_rays": 32,              # Number of LiDAR rays per agent
        "obs_len_range": [0.1, 0.5],  # Range of obstacle dimensions [min, max]
        "n_obs": 32,                # Number of randomly generated obstacles
        "m": 0.1,                  # Mass of each agent (kg)
    }

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int = 256,
            max_travel: float = None,
            dt: float = 0.03,
            params: dict = None
    ):
        """
        Initialize the double integrator environment.
        
        Args:
            num_agents: Number of agents in the environment
            area_size: Size of the square environment (area_size x area_size)
            max_step: Maximum number of timesteps per episode
            max_travel: Maximum distance agents can travel (for goal placement)
            dt: Timestep duration in seconds
            params: Optional dictionary to override default PARAMS
        """
        super(DoubleIntegrator, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        
        # Build discrete-time linear system matrices for double integrator dynamics
        # Continuous dynamics: ẋ = v, v̇ = u/m
        # State: [x, y, vx, vy], Action: [fx, fy]
        
        # State transition matrix A (relates current state to next state without control)
        A = np.zeros((self.state_dim, self.state_dim), dtype=np.float32)
        A[0, 2] = 1.0  # x_next = x + vx * dt
        print("A[0, 2]:", A)

        A[1, 3] = 1.0  # y_next = y + vy * dt
        print("A[1, 3]:", A)
        self._A = A * self._dt + np.eye(self.state_dim)  # Discretize: I + A*dt
        
        print("selfA:", self._A)
        # Control input matrix B (relates control action to state change)
        # Only affects velocity components (rows 2,3) since force -> acceleration
        self._B = (
            np.array([[0.0, 0.0], [0.0, 0.0], [1.0 / self._params["m"], 0.0], [0.0, 1.0 / self._params["m"]]])
            * self._dt
        )
        print("self.B:", self._B)
        # LQR cost matrices for computing reference controller
        self._Q = np.eye(self.state_dim) * 5  # State deviation penalty
        self._R = np.eye(self.action_dim)     # Control effort penalty
        
        # Compute LQR gain matrix K for feedback control: u = -K(x - x_goal)
        self._K = jnp.array(lqr(self._A, self._B, self._Q, self._R))
        
        # Vectorized obstacle creation function
        self.create_obstacles = jax_vmap(Rectangle.create)

    @property
    def state_dim(self) -> int:
        """Dimensionality of agent state: [x, y, vx, vy]."""
        return 4  # x, y, vx, vy

    @property
    def node_dim(self) -> int:
        """Dimensionality of node features in graph: one-hot encoding of node type."""
        return 3  # indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        """Dimensionality of edge features in graph: relative state between nodes."""
        return 4  # x_rel, y_rel, vx_rel, vy_rel

    @property
    def action_dim(self) -> int:
        """Dimensionality of action space: forces in x and y directions."""
        return 2  # fx, fy

    def reset(self, key: Array) -> GraphsTuple:
        """
        Reset the environment to initial state.
        
        This function:
        1. Randomly generates rectangular obstacles in the environment
        2. Randomly places agents and goals, ensuring they don't overlap with obstacles
        3. Initializes all agents with zero velocity
        4. Constructs and returns the initial graph representation
        
        Args:
            key: JAX random key for reproducible randomness
            
        Returns:
            GraphsTuple representing the initial state with agents, goals, and obstacles
        """
        self._t = 0
        # Randomly generate rectangular obstacles
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0
        
        # Generate random obstacle positions
        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
        
        # Generate random obstacle dimensions (length and width)
        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (n_rng_obs, 2),
            minval=self._params["obs_len_range"][0],
            maxval=self._params["obs_len_range"][1],
        )
        
        # Generate random obstacle orientations
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
        
        # Create obstacle objects
        obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        # Randomly generate agent start positions and goal positions
        # Ensures minimum separation from obstacles and between agents/goals
        states, goals = get_node_goal_rng(
            key, self.area_size, 2, obstacles, self.num_agents, 4 * self.params["car_radius"], self.max_travel)

        print("states", states)
        print("goals", goals)

        # Append zero velocities to position-only states
        # states/goals are [num_agents, 2], extend to [num_agents, 4]
        states = jnp.concatenate([states, jnp.zeros((self.num_agents, 2))], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros((self.num_agents, 2))], axis=1)

        # Package into environment state
        env_states = self.EnvState(states, goals, obstacles)
        print("env_states:", env_states)

        # Convert to graph representation
        return self.get_graph(env_states)

    def agent_accel(self, action: Action) -> Action:
        """
        Convert force action to acceleration using Newton's second law: a = F/m.
        
        Args:
            action: Force inputs [num_agents, 2] in the form [fx, fy]
            
        Returns:
            Acceleration [num_agents, 2] in the form [ax, ay]
        """
        return action / self._params["m"]

    def agent_step_exact(self, agent_states: AgentState, action: Action) -> AgentState:
        """
        Compute exact next state using kinematic equations for constant acceleration.
        
        Uses: x_new = x + v*dt + 0.5*a*dt^2
              v_new = v + a*dt
        
        Args:
            agent_states: Current states [num_agents, 4] as [x, y, vx, vy]
            action: Force actions [num_agents, 2] as [fx, fy]
            
        Returns:
            Next states [num_agents, 4]
        """
        assert action.shape == (self.num_agents, self.action_dim)
        # [x, y, vx, vy]
        assert agent_states.shape == (self.num_agents, self.state_dim)
        
        # Convert force to acceleration
        n_accel = self.agent_accel(action)
        
        # Update position: x_new = x + v*dt + 0.5*a*dt^2
        n_pos_new = agent_states[:, :2] + agent_states[:, 2:] * self.dt + n_accel * self.dt**2 / 2
        
        # Update velocity: v_new = v + a*dt
        n_vel_new = agent_states[:, 2:] + n_accel * self.dt
        
        # Concatenate position and velocity
        n_state_agent_new = jnp.concatenate([n_pos_new, n_vel_new], axis=1)
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return n_state_agent_new

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        """
        Compute next state using forward Euler integration.
        
        Uses: x_new = x + ẋ*dt where ẋ is computed by agent_xdot()
        
        Args:
            agent_states: Current states [num_agents, 4] as [x, y, vx, vy]
            action: Force actions [num_agents, 2] as [fx, fy]
            
        Returns:
            Next states [num_agents, 4], clipped to state limits
        """
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        
        # Get state derivative
        x_dot = self.agent_xdot(agent_states, action)
        
        # Euler step: x_new = x + ẋ*dt
        n_state_agent_new = x_dot * self.dt + agent_states
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        
        # Clip to state bounds (velocity limits)
        return self.clip_state(n_state_agent_new)

    def agent_xdot(self, agent_states: AgentState, action: Action) -> AgentState:
        """
        Compute state derivative for double integrator dynamics.
        
        Dynamics: ẋ = vx, ẏ = vy, v̇x = fx/m, v̇y = fy/m
        
        Args:
            agent_states: Current states [num_agents, 4] as [x, y, vx, vy]
            action: Force actions [num_agents, 2] as [fx, fy]
            
        Returns:
            State derivatives [num_agents, 4] as [vx, vy, ax, ay]
        """
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        
        # Convert force to acceleration
        n_accel = self.agent_accel(action)
        
        # State derivative: [velocity, acceleration]
        x_dot = jnp.concatenate([agent_states[:, 2:], n_accel], axis=1)
        assert x_dot.shape == (self.num_agents, self.state_dim)
        return x_dot

    def step(
        self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        """
        Execute one environment step with the given actions.
        
        This function:
        1. Applies actions to update agent states
        2. Computes reward (negative deviation from reference controller)
        3. Computes cost (collision penalty)
        4. Constructs next graph representation
        
        Args:
            graph: Current graph state representation
            action: Actions for all agents [num_agents, 2] as [fx, fy]
            get_eval_info: If True, include additional evaluation metrics in info
            
        Returns:
            Tuple of (next_graph, reward, cost, done, info)
        """
        self._t += 1

        # Extract current state components from graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle
        
        # Ensure actions are within limits
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        # Simulate physics to get next agent states
        next_agent_states = self.agent_step_euler(agent_states, action)

        # Episode termination controlled externally (not by this function)
        done = jnp.array(False)

        # Compute reward: penalize deviation from LQR reference controller
        # Encourages smooth, optimal-like behavior
        reward = jnp.zeros(()).astype(jnp.float32)
        reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
        
        # Compute cost: collision penalty (agent-agent and agent-obstacle)
        cost = self.get_cost(graph)

        assert reward.shape == tuple()
        assert cost.shape == tuple()
        assert done.shape == tuple()

        # Package next state (goals and obstacles remain static)
        next_state = self.EnvState(next_agent_states, goal_states, obstacles)

        # Optional: include diagnostic information for evaluation
        info = {}
        if get_eval_info:
            agent_pos = agent_states[:, :2]
            info["inside_obstacles"] = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])

        return self.get_graph(next_state), reward, cost, done, info

    def get_cost(self, graph: EnvGraphsTuple) -> Cost:
        """
        Compute collision cost for the current state.
        
        Cost includes:
        1. Agent-agent collisions (distance < 2*car_radius)
        2. Agent-obstacle collisions (agent center inside obstacle + car_radius buffer)
        
        Args:
            graph: Current graph state
            
        Returns:
            Scalar cost in [0, 1] representing fraction of agents in collision
        """
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle

        # Check agent-agent collisions
        agent_pos = agent_states[:, :2]
        
        # Compute pairwise distances between all agents
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        
        # Ignore self-distances by adding large value on diagonal
        dist += jnp.eye(self.num_agents) * 1e6
        
        # Collision if distance < 2*radius (agents touching)
        collision = (self._params["car_radius"] * 2 > dist).any(axis=1)
        cost = collision.mean()

        # Check agent-obstacle collisions
        collision = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])
        cost += collision.mean()

        return cost

    def render_video(
            self,
            rollout: RolloutResult,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        """
        Render a video of the agent trajectories from a rollout.
        
        Args:
            rollout: Recorded trajectory data from an episode
            video_path: Where to save the output video file
            Ta_is_unsafe: Optional array indicating unsafe timesteps
            viz_opts: Visualization options dictionary
            dpi: Dots per inch for video resolution
            **kwargs: Additional arguments passed to render function
        """
        render_video(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=2,
            n_agent=self.num_agents,
            n_rays=self.params["n_rays"],
            r=self.params["car_radius"],
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            dpi=dpi,
            **kwargs
        )

    def edge_blocks(self, state: EnvState, lidar_data: State) -> list[EdgeBlock]:
        """
        Construct edge blocks for the graph neural network representation.
        
        Creates three types of edges:
        1. Agent-agent edges: connections between nearby agents (within comm_radius)
        2. Agent-goal edges: one-to-one connections from each agent to its goal
        3. Agent-obstacle edges: connections from agents to LiDAR hit points
        
        Edge features are relative state differences (pos + vel) between connected nodes.
        
        Args:
            state: Current environment state
            lidar_data: LiDAR hit point positions [n_rays * num_agents, 4]
            
        Returns:
            List of EdgeBlock objects defining graph connectivity and features
        """
        n_hits = self._params["n_rays"] * self.num_agents

        # ===== Agent-Agent Edges =====
        # Connect agents that are within communication radius of each other
        agent_pos = state.agent[:, :2]
        
        # Compute relative positions: pos_diff[i,j] = pos_j - pos_i (vector from i to j)
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        
        # Compute distances between all agent pairs
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        
        # Remove self-connections by making diagonal distances large
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        
        # Full state difference (position + velocity)
        state_diff = state.agent[:, None, :] - state.agent[None, :, :]
        
        # Create mask: True where agents are within comm_radius
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # ===== Agent-Goal Edges =====
        # One-to-one connection from each agent to its corresponding goal
        # Features are clipped to comm_radius to prevent excessively large edge features
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        agent_goal_mask = jnp.eye(self.num_agents)  # Diagonal mask for one-to-one
        
        # Relative state: agent -> goal
        agent_goal_feats = state.agent[:, None, :] - state.goal[None, :, :]
        
        # Clip spatial component to comm_radius to stabilize GNN
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        
        # Scaling coefficient: 1.0 if within radius, else comm_radius/distance
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :2].set(agent_goal_feats[:, :2] * coef)
        
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )

        # ===== Agent-Obstacle Edges =====
        # Connect each agent to its LiDAR hit points (obstacles in sensing range)
        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        
        for i in range(self.num_agents):
            # Get indices for this agent's LiDAR rays
            id_hits = jnp.arange(i * self._params["n_rays"], (i + 1) * self._params["n_rays"])
            
            # Relative position and state from agent to LiDAR hits
            lidar_pos = agent_pos[i, :] - lidar_data[id_hits, :2]
            lidar_feats = state.agent[i, :] - lidar_data[id_hits, :]
            
            # Distance to each LiDAR hit
            lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
            
            # Only connect to LiDAR points within sensing range
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
            
            agent_obs_mask = jnp.ones((1, self._params["n_rays"]))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges

    def control_affine_dyn(self, state: State) -> [Array, Array]:
        assert state.ndim == 2
        
        # Drift dynamics: position changes with velocity, no acceleration drift
        f = jnp.concatenate([state[:, 2:], jnp.zeros((state.shape[0], 2))], axis=1)
        
        # Control matrix: forces only affect acceleration (rows 2,3)
        g = jnp.concatenate([jnp.zeros((2, 2)), jnp.eye(2) / self._params['m']], axis=0)
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        
        assert f.shape == state.shape
        assert g.shape == (state.shape[0], self.state_dim, self.action_dim)
        return f, g

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        """
        Update edge features in graph with relative states.
        
        Recomputes edge features as relative state differences between sender and receiver
        nodes, with spatial components clipped to communication radius for stability.
        
        Args:
            graph: Graph with existing connectivity
            state: Updated node states [n_nodes, 4]
            
        Returns:
            Graph with updated edge features and states
        """
        assert graph.is_single
        assert state.ndim == 2

        # Compute relative state: receiver - sender
        edge_feats = state[graph.receivers] - state[graph.senders]
        
        # Clip spatial components to comm_radius
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        edge_feats = edge_feats.at[:, :2].set(edge_feats[:, :2] * coef)

        return graph._replace(edges=edge_feats, states=state)

    def get_graph(self, state: EnvState, adjacency: Array = None) -> GraphsTuple:
        """
        Construct graph representation from environment state.
        
        Graph structure:
        - Nodes: [agent_1, ..., agent_n, goal_1, ..., goal_n, lidar_hit_1, ..., lidar_hit_m]
        - Node features: one-hot encoding [is_obstacle, is_goal, is_agent]
        - Edges: defined by edge_blocks (agent-agent, agent-goal, agent-obstacle)
        - Edge features: relative states between connected nodes
        
        Args:
            state: Current environment state with agents, goals, and obstacles
            adjacency: Unused, kept for interface compatibility
            
        Returns:
            GraphsTuple with complete graph representation
        """
        # Calculate total number of nodes
        n_hits = self._params["n_rays"] * self.num_agents  # LiDAR hit points
        n_nodes = 2 * self.num_agents + n_hits
        
        # Create one-hot node features [n_nodes, 3]
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits, 3))
        node_feats = node_feats.at[: self.num_agents, 2].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 1].set(1)  # goal feats
        node_feats = node_feats.at[-n_hits:, 0].set(1)  # obs feats

        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(DoubleIntegrator.GOAL)
        node_type = node_type.at[-n_hits:].set(DoubleIntegrator.OBS)

        get_lidar_vmap = jax_vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
            )
        )
        lidar_data = merge01(get_lidar_vmap(state.agent[:, :2]))
        lidar_data = jnp.concatenate([lidar_data, jnp.zeros_like(lidar_data)], axis=-1)
        edge_blocks = self.edge_blocks(state, lidar_data)

        # create graph
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate([state.agent, state.goal, lidar_data], axis=0),
        ).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([-jnp.inf, -jnp.inf, -0.5, -0.5])
        upper_lim = jnp.array([jnp.inf, jnp.inf, 0.5, 0.5])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal = graph.type_states(type_idx=1, n_type=self.num_agents)
        error = goal - agent
        error_max = jnp.abs(error / jnp.linalg.norm(error, axis=-1, keepdims=True) * self._params["comm_radius"])
        error = jnp.clip(error, -error_max, error_max)
        return self.clip_action(error @ self._K.T)

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)
        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)

        next_graph = self.add_edge_feats(graph, next_states)
        return next_graph

    
    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, graph: GraphsTuple) -> Array:
        """
        Determine which agents are in safe states.
        
        An agent is considered safe if:
        1. It maintains clearance of 4*car_radius from all other agents
        2. It maintains clearance of 2*car_radius from all obstacles
        
        This conservative definition ensures agents have buffer space for maneuvering.
        
        Args:
            graph: Current graph state
            
        Returns:
            Boolean array [num_agents] where True indicates safe state
        """
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
    
        # ---- Single-agent optimization ----
        # Skip inter-agent collision checks if only one agent
        if self.num_agents == 1:
            # Safe if outside obstacles with 2*car_radius clearance
            safe_obs = jnp.logical_not(
                inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 2)
            )
            return safe_obs  # shape (1,)
    
        # ---- Multi-agent logic ----
        # Check inter-agent distances
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: vector from j to i
        dist = jnp.linalg.norm(pos_diff, axis=-1)  # Pairwise distances
        
        # Remove self-connections by adding large distance on diagonal
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)
        
        # Safe if distance > 4*car_radius from all other agents
        safe_agent = jnp.greater(dist, self._params["car_radius"] * 4)
        safe_agent = jnp.min(safe_agent, axis=1)  # All pairwise distances must be safe
    
        # Safe if outside obstacles with 2*car_radius clearance
        safe_obs = jnp.logical_not(
            inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 2)
        )
        
        # Safe only if both conditions satisfied
        safe_mask = jnp.logical_and(safe_agent, safe_obs)
        return safe_mask
    
    
    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        """
        Determine which agents are in unsafe states.
        
        An agent is unsafe if:
        1. COLLISION: Touching another agent (dist < 2*car_radius) OR inside an obstacle
        2. UNSAFE DIRECTION: Heading toward an agent/obstacle within a warning zone
        
        The unsafe direction check uses:
        - Warning zones: 3*car_radius for agents, 2*car_radius for obstacles
        - Heading cone: angle threshold based on geometry (tangent to circular obstacle)
        - Inner product: checks if velocity vector points toward the hazard
        
        Args:
            graph: Current graph state
            
        Returns:
            Boolean array [num_agents] where True indicates unsafe state
        """
        agent_state = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_state[:, :2]
    
        # ---- Single-agent optimization ----
        if self.num_agents == 1:
            # ===== PART 1: Direct Collision Detection =====
            # Check if agent is physically touching/inside an obstacle
            # Example: agent at [0.5, 0.5], car_radius=0.05
            #          If obstacle boundary is at distance < 0.05 from agent center → COLLISION
            unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])
            
            # ===== PART 2: Predictive Unsafe Direction Detection =====
            # Uses LiDAR to detect if agent is HEADING toward an obstacle (even if not colliding yet)
            
            # Step 1: Get LiDAR hit points
            # LiDAR shoots rays in all directions, returns where they hit obstacles
            # Example: 32 rays → 32 hit points around the agent
            n_hits = self._params["n_rays"] * self.num_agents  # = n_rays for single agent (e.g., 32)
            obs_pos = graph.type_states(type_idx=2, n_type=n_hits)[:, :2]  # LiDAR hit points [32, 2]
            
            # Step 2: Compute vectors from agent to each LiDAR hit point
            # Example: 
            #   agent_pos = [[0.5, 0.5]]  (shape: [1, 2])
            #   obs_pos = [[0.6, 0.5], [0.55, 0.6], ...]  (shape: [32, 2])
            #   obs_pos_diff[0, 0, :] = [0.6, 0.5] - [0.5, 0.5] = [0.1, 0.0]  → vector to 1st obstacle
            #   obs_pos_diff[0, 1, :] = [0.55, 0.6] - [0.5, 0.5] = [0.05, 0.1] → vector to 2nd obstacle
            obs_pos_diff = obs_pos[None, :, :] - agent_pos[:, None, :]  # [1, 32, 2]
            print("obs_pos_diff:", obs_pos_diff)
            # Step 3: Calculate distance to each obstacle point
            # Example:
            #   obs_dist[0, 0] = sqrt(0.1^2 + 0.0^2) = 0.1  (10 cm away)
            #   obs_dist[0, 1] = sqrt(0.05^2 + 0.1^2) = 0.112  (11.2 cm away)
            obs_dist = jnp.linalg.norm(obs_pos_diff, axis=-1)  # [1, 32]
            print("obs_dist:", obs_dist)
            # Step 4: Define warning zone (how close is "too close"?)
            # Example: car_radius = 0.05, so warn_dist = 2*0.05 = 0.1 m
            #   Any obstacle within 10 cm triggers warning
            obs_warn_dist = 2 * self._params["car_radius"]
            warn_zone_obs = jnp.less(obs_dist, obs_warn_dist)  # [1, 32] boolean array
            print("warn_zone_obs:", warn_zone_obs)  
            
            # Example result: [True, False, True, False, ...] where True = obstacle within 10 cm
    
            # Step 5: Normalize direction vectors (make them unit length)
            # Example:
            #   obs_pos_diff[0, 0] = [0.1, 0.0] → pos_vec[0, 0] = [1.0, 0.0] (pointing East)
            #   obs_pos_diff[0, 1] = [0.05, 0.1] → pos_vec[0, 1] = [0.447, 0.894] (pointing NE)
            pos_vec = obs_pos_diff / (jnp.linalg.norm(obs_pos_diff, axis=2, keepdims=True) + 1e-4)  # [1, 32, 2]
            print("pos_vec:", pos_vec)
            # Step 6: Get agent's heading direction (where is it moving?)
            # Example: agent_state[:, 2:] = [[0.2, 0.1]] (vx=0.2 m/s East, vy=0.1 m/s North)
            #   speed = sqrt(0.2^2 + 0.1^2) = 0.224 m/s
            #   heading = [0.2/0.224, 0.1/0.224] = [0.894, 0.447] (moving mostly East)
            speed_agent = jnp.linalg.norm(agent_state[:, 2:], axis=1, keepdims=True)  # [1, 1]
            heading_vec0 = (agent_state[:, 2:] / (speed_agent + 1e-4))[:, None, :]  # [1, 1, 2]
            heading_vec = jnp.repeat(heading_vec0, pos_vec.shape[1], axis=1)  # [1, 32, 2] broadcast
            
            # Step 7: Check alignment - is agent heading TOWARD each obstacle?
            # Inner product (dot product) = cos(angle between vectors)
            # Example:
            #   heading = [0.894, 0.447] (moving East-North)
            #   obstacle_dir[0] = [1.0, 0.0] (obstacle to the East)
            #   inner_prod = 0.894*1.0 + 0.447*0.0 = 0.894
            #   This is cos(26°) → agent heading 26° away from straight-toward-obstacle
            #
            #   obstacle_dir[1] = [0.447, 0.894] (obstacle to the North-East)
            #   inner_prod = 0.894*0.447 + 0.447*0.894 = 0.8 
            #   This is cos(36°) → pretty well aligned, heading toward it!
            inner_prod = jnp.sum(pos_vec * heading_vec, axis=2)  # [1, 32]
            print("inner_prod:", inner_prod)
            # Values close to 1.0 = heading straight toward obstacle
            # Values close to 0.0 = heading perpendicular
            # Values close to -1.0 = heading away from obstacle
    
            # Step 8: Compute "danger cone" angle threshold
            # Geometrically: what's the maximum angle where agent could still hit obstacle?
            # 
            # Visualization:
            #        obstacle (radius r)
            #           ___
            #          /   \
            #    θ → /       \  ← θ (danger cone half-angle)
            #       /    ·    \    
            #      |  center  |
            #       \    d   /     d = distance to obstacle
            #        \ ___ /
            #          ↓
            #        agent
            #
            # Math: tan(θ) = r / sqrt(d² - r²)  (from right triangle geometry)
            # Example: 
            #   car_radius = 0.05, distance = 0.1
            #   theta = arctan(0.05 / sqrt(0.1² - 0.05²)) = arctan(0.05/0.0866) = 30°
            #   So if agent heading is within ±30° of straight-toward-obstacle → will collide!
            unsafe_theta_obs = jnp.arctan2(
                self._params["car_radius"],
                jnp.sqrt(jnp.maximum(obs_dist**2 - self._params["car_radius"]**2, 1e-9))
            )  # [1, 32] angle in radians
            print("unsafe_theta_obs:", unsafe_theta_obs)
            # Step 9: Determine if agent is in unsafe direction
            # Unsafe if:
            #   1. Obstacle is in warning zone (< 10 cm away)
            #   AND
            #   2. Agent heading is within danger cone (inner_prod > cos(unsafe_theta))
            #
            # Example:
            #   unsafe_theta = 30° → cos(30°) = 0.866
            #   If inner_prod = 0.894 > 0.866 → UNSAFE! (heading too close to obstacle direction)
            #   If inner_prod = 0.5 < 0.866 → safe (heading away enough from obstacle)
            unsafe_dir = jnp.max(
                jnp.logical_and(warn_zone_obs, jnp.greater(inner_prod, jnp.cos(unsafe_theta_obs))), 
                axis=1
            )  # [1] - True if ANY ray shows unsafe direction
            print("unsafe_dir:", unsafe_dir)
            
            # Example final result:
            #   Ray 5: distance=0.08 (< 0.1, in warning zone), inner_prod=0.95 (> cos(35°)=0.819)
            #   → This ray is UNSAFE
            #   → unsafe_dir = True (at least one unsafe ray found)
    
            collision_mask = unsafe_obs  # [1]
            print("collision_mask:", collision_mask)
            print("Final unsafe mask:", jnp.logical_or(collision_mask, unsafe_dir))
            return jnp.logical_or(collision_mask, unsafe_dir)
            # Returns True if: currently colliding OR heading dangerously toward obstacle
    
        # ---- Multi-agent logic ----
        # Check agent-agent collisions
        agent_pos_diff = agent_pos[None, :, :] - agent_pos[:, None, :]
        agent_dist = jnp.linalg.norm(agent_pos_diff, axis=-1)
        agent_dist = agent_dist + jnp.eye(agent_dist.shape[1]) * (self._params["car_radius"] * 2 + 1)
        
        # Unsafe if touching another agent
        unsafe_agent = jnp.less(agent_dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)
    
        # Check agent-obstacle collisions
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])
        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)
    
        # Check unsafe heading directions
        agent_warn_dist = 3 * self._params["car_radius"]  # Warning zone for agents
        obs_warn_dist = 2 * self._params["car_radius"]    # Warning zone for obstacles
        
        # Get LiDAR obstacle positions
        obs_pos = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)[:, :2]
        obs_pos_diff = obs_pos[None, :, :] - agent_pos[:, None, :]
        obs_dist = jnp.linalg.norm(obs_pos_diff, axis=-1)
        
        # Combine agent and obstacle positions
        pos_diff = jnp.concatenate([agent_pos_diff, obs_pos_diff], axis=1)
        
        # Create warning zone mask
        warn_zone = jnp.concatenate(
            [jnp.less(agent_dist, agent_warn_dist), jnp.less(obs_dist, obs_warn_dist)], 
            axis=1
        )
        
        # Normalize position vectors
        pos_vec = (pos_diff / (jnp.linalg.norm(pos_diff, axis=2, keepdims=True) + 0.0001))
        
        # Compute heading vectors for all agents
        speed_agent = jnp.linalg.norm(agent_state[:, 2:], axis=1, keepdims=True)
        heading_vec0 = (agent_state[:, 2:] / (speed_agent + 0.0001))[:, None, :]
        heading_vec = heading_vec0.repeat(pos_vec.shape[1], axis=1)
        
        # Alignment with heading
        inner_prod = jnp.sum(pos_vec * heading_vec, axis=2)
        
        # Compute danger cone angles
        unsafe_theta_agent = jnp.arctan2(
            self._params['car_radius'] * 2,
            jnp.sqrt(jnp.maximum(agent_dist**2 - 4 * self._params['car_radius']**2, 1e-9))
        )
        unsafe_theta_obs = jnp.arctan2(
            self._params['car_radius'],
            jnp.sqrt(jnp.maximum(obs_dist**2 - self._params['car_radius']**2, 1e-9))
        )
        unsafe_theta = jnp.concatenate([unsafe_theta_agent, unsafe_theta_obs], axis=1)
        
        # Create mask to only check valid LiDAR connections (agent i with its own LiDAR hits)
        lidar_mask = jnp.ones((self._params["n_rays"],))
        lidar_mask = jax.scipy.linalg.block_diag(*[lidar_mask] * self.num_agents)
        valid_mask = jnp.concatenate([jnp.ones((self.num_agents, self.num_agents)), lidar_mask], axis=-1)
        warn_zone = jnp.logical_and(warn_zone, valid_mask)
        
        # Unsafe if heading toward any hazard within warning zone
        unsafe_dir = jnp.max(
            jnp.logical_and(warn_zone, jnp.greater(inner_prod, jnp.cos(unsafe_theta))), 
            axis=1
        )
    
        return jnp.logical_or(collision_mask, unsafe_dir)
    
    
    def collision_mask(self, graph: GraphsTuple) -> Array:
        """
        Check which agents are currently in collision.
        
        Collision occurs when:
        - Agent-agent: distance < 2*car_radius (agents touching)
        - Agent-obstacle: agent center inside obstacle boundary + car_radius buffer
        
        Args:
            graph: Current graph state
            
        Returns:
            Boolean array [num_agents] where True indicates collision
        """
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
    
        # ---- Single-agent optimization ----
        if self.num_agents == 1:
            unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])
            return unsafe_obs  # shape (1,)
    
        # ---- Multi-agent logic ----
        # Check inter-agent collisions
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # Remove self-connections
        
        unsafe_agent = jnp.less(dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)  # Any collision
    
        # Check obstacle collisions
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])
        
        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)
        return collision_mask
    
    
    def finish_mask(self, graph: GraphsTuple) -> Array:
        """
        Check which agents have reached their goals.
        
        An agent is considered to have reached its goal when the distance to the goal
        is less than 2*car_radius.
        
        Args:
            graph: Current graph state
            
        Returns:
            Boolean array [num_agents] where True indicates goal reached
        """
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        
        # Distance from each agent to its corresponding goal
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 2
        
        return reach  # works for both single-agent (shape (1,)) and multi-agent

    # @ft.partial(jax.jit, static_argnums=(0,))
    # def safe_mask(self, graph: GraphsTuple) -> Array:
    #     agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

    #     # agents are not colliding
    #     pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
    #     dist = jnp.linalg.norm(pos_diff, axis=-1)
    #     dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
    #     safe_agent = jnp.greater(dist, self._params["car_radius"] * 4)

    #     safe_agent = jnp.min(safe_agent, axis=1)

    #     safe_obs = jnp.logical_not(
    #         inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 2)
    #     )

    #     safe_mask = jnp.logical_and(safe_agent, safe_obs)

    #     return safe_mask

    # @ft.partial(jax.jit, static_argnums=(0,))
    # def unsafe_mask(self, graph: GraphsTuple) -> Array:
    #     agent_state = graph.type_states(type_idx=0, n_type=self.num_agents)
    #     agent_pos = agent_state[:, :2]

    #     # agents are colliding
    #     agent_pos_diff = agent_pos[None, :, :] - agent_pos[:, None, :]
    #     agent_dist = jnp.linalg.norm(agent_pos_diff, axis=-1)
    #     agent_dist = agent_dist + jnp.eye(agent_dist.shape[1]) * (self._params["car_radius"] * 2 + 1)
    #     unsafe_agent = jnp.less(agent_dist, self._params["car_radius"] * 2)
    #     unsafe_agent = jnp.max(unsafe_agent, axis=1)

    #     # agents are colliding with obstacles
    #     unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

    #     collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

    #     # unsafe direction
    #     agent_warn_dist = 3 * self._params["car_radius"]
    #     obs_warn_dist = 2 * self._params["car_radius"]
    #     obs_pos = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)[:, :2]
    #     obs_pos_diff = obs_pos[None, :, :] - agent_pos[:, None, :]
    #     obs_dist = jnp.linalg.norm(obs_pos_diff, axis=-1)
    #     pos_diff = jnp.concatenate([agent_pos_diff, obs_pos_diff], axis=1)
    #     warn_zone = jnp.concatenate([jnp.less(agent_dist, agent_warn_dist), jnp.less(obs_dist, obs_warn_dist)], axis=1)
    #     pos_vec = (pos_diff / (jnp.linalg.norm(pos_diff, axis=2, keepdims=True) + 0.0001))
    #     speed_agent = jnp.linalg.norm(agent_state[:, 2:], axis=1, keepdims=True)
    #     heading_vec0 = (agent_state[:, 2:] / (speed_agent + 0.0001))[:, None, :]
    #     heading_vec = heading_vec0.repeat(pos_vec.shape[1], axis=1)
    #     inner_prod = jnp.sum(pos_vec * heading_vec, axis=2)
    #     unsafe_theta_agent = jnp.arctan2(self._params['car_radius'] * 2,
    #                                      jnp.sqrt(agent_dist**2 - 4 * self._params['car_radius']**2))
    #     unsafe_theta_obs = jnp.arctan2(self._params['car_radius'],
    #                                    jnp.sqrt(obs_dist**2 - self._params['car_radius']**2))
    #     unsafe_theta = jnp.concatenate([unsafe_theta_agent, unsafe_theta_obs], axis=1)
    #     lidar_mask = jnp.ones((self._params["n_rays"],))
    #     lidar_mask = jax.scipy.linalg.block_diag(*[lidar_mask] * self.num_agents)
    #     valid_mask = jnp.concatenate([jnp.ones((self.num_agents, self.num_agents)), lidar_mask], axis=-1)
    #     warn_zone = jnp.logical_and(warn_zone, valid_mask)
    #     unsafe_dir = jnp.max(jnp.logical_and(warn_zone, jnp.greater(inner_prod, jnp.cos(unsafe_theta))), axis=1)

    #     return jnp.logical_or(collision_mask, unsafe_dir)  # | unsafe_stop

    # def collision_mask(self, graph: GraphsTuple) -> Array:
    #     agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

    #     # agents are colliding
    #     pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
    #     dist = jnp.linalg.norm(pos_diff, axis=-1)
    #     dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
    #     unsafe_agent = jnp.less(dist, self._params["car_radius"] * 2)
    #     unsafe_agent = jnp.max(unsafe_agent, axis=1)

    #     # agents are colliding with obstacles
    #     unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

    #     collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

    #     return collision_mask

    # def finish_mask(self, graph: GraphsTuple) -> Array:
    #     agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
    #     goal_pos = graph.env_states.goal[:, :2]
    #     reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 2
    #     return reach
