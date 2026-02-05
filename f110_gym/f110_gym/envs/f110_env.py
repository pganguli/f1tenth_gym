"""
Author: Hongrui Zheng
"""

# gym imports
import pathlib
import time
from typing import Any

# others
import numpy as np

# gl
import pyglet

import gymnasium as gym
from gymnasium import spaces

# base classes
from .base_classes import Integrator, Simulator

pyglet.options["debug_gl"] = False

# rendering constants
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800


class F110Env(gym.Env):
    """
    OpenAI/Gymnasium environment for F1TENTH

    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility

            map (str, default='vegas'): name of the map used for the environment. Currently, available environments include: 'berlin', 'vegas', 'skirk'. You could use a string of the absolute path to the yaml file of your custom map.

            map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'

            params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front
            C_Sr: Cornering stiffness coefficient, rear
            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=2): number of agents in the environment

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents

            lidar_dist (float, default=0): vertical distance between LiDAR and backshaft
    """

    metadata = {"render_modes": ["human", "human_fast"], "render_fps": 200}

    # rendering
    renderer = None
    current_obs = None
    render_callbacks = []

    def __init__(self, **kwargs):
        # kwargs extraction
        self.seed = kwargs.get("seed", 12345)
        self.render_mode = kwargs.get("render_mode", "human")
        self.render_fps = kwargs.get("render_fps", 200)
        self.map_name = kwargs.get("map", "vegas")

        # different default maps
        current_dir = pathlib.Path(__file__).parent.absolute()
        if self.map_name in ["berlin", "skirk", "levine", "vegas"]:
            self.map_path = str(current_dir / "maps" / f"{self.map_name}.yaml")
        else:
            self.map_path = self.map_name + ".yaml"

        self.map_ext = kwargs.get("map_ext", ".png")

        default_params = {
            "mu": 1.0489,
            "C_Sf": 4.718,
            "C_Sr": 5.4562,
            "lf": 0.15875,
            "lr": 0.17145,
            "h": 0.074,
            "m": 3.74,
            "I": 0.04712,
            "s_min": -0.4189,
            "s_max": 0.4189,
            "sv_min": -3.2,
            "sv_max": 3.2,
            "v_switch": 7.319,
            "a_max": 9.51,
            "v_min": -5.0,
            "v_max": 20.0,
            "width": 0.31,
            "length": 0.58,
        }
        self.params = kwargs.get("params", default_params)

        # simulation parameters
        self.num_agents = kwargs.get("num_agents", 2)
        self.timestep = kwargs.get("timestep", 0.01)

        # default ego index
        self.ego_idx = kwargs.get("ego_idx", 0)

        # default integrator
        self.integrator = kwargs.get("integrator", Integrator.RK4)

        # default LiDAR position
        self.lidar_dist = kwargs.get("lidar_dist", 0.0)

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents,), dtype=np.float64)
        # TODO: collision_idx not used yet
        # self.collision_idx = -1 * np.ones((self.num_agents, ))

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents,), dtype=np.float64)
        self.lap_counts = np.zeros((self.num_agents,), dtype=np.float64)
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,), dtype=np.float64)
        self.start_xs = np.zeros((self.num_agents,), dtype=np.float64)
        self.start_ys = np.zeros((self.num_agents,), dtype=np.float64)
        self.start_thetas = np.zeros((self.num_agents,), dtype=np.float64)
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(
            self.params,
            self.num_agents,
            self.seed,
            time_step=self.timestep,
            integrator=self.integrator,
            lidar_dist=self.lidar_dist,
        )
        self.sim.set_map(self.map_path, self.map_ext)

        # stateful observations for rendering
        self.render_obs = None

        # Gymnasium requires action_space and observation_space
        # Action space: (num_agents, 2) - [steering, velocity] per agent
        self.action_space = spaces.Box(
            low=np.array([[self.params["s_min"], self.params["v_min"]]] * self.num_agents, dtype=np.float64),
            high=np.array([[self.params["s_max"], self.params["v_max"]]] * self.num_agents, dtype=np.float64),
            dtype=np.float64,
        )

        # Observation space: Dict space with various observations
        # scans shape depends on lidar config, using 1080 beams as default
        num_beams = 1080
        self.observation_space = spaces.Dict({
            "ego_idx": spaces.Discrete(self.num_agents),
            "scans": spaces.Box(low=0.0, high=30.0, shape=(self.num_agents, num_beams), dtype=np.float64),
            "poses_x": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
            "poses_y": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
            "poses_theta": spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_agents,), dtype=np.float64),
            "linear_vels_x": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
            "linear_vels_y": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
            "ang_vels_z": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
            "collisions": spaces.Box(low=0.0, high=1.0, shape=(self.num_agents,), dtype=np.float64),
            "lap_times": spaces.Box(low=0.0, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
            "lap_counts": spaces.Box(low=0.0, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
        })

    def _check_done(self) -> tuple[bool, bool]:
        """
        Check if the current rollout is done

        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (np.ndarray): boolean mask of agents that have finished the lap
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t = 2
        right_t = 2

        poses_x = np.array(self.poses_x) - self.start_xs
        poses_y = np.array(self.poses_y) - self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :] ** 2 + temp_y**2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time

        done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)

        return bool(done), bool(self.toggle_list >= 4)

    def _update_state(self, obs_dict: dict[str, Any]) -> None:
        """
        Update the env's states according to observations

        Args:
            obs_dict (dict): dictionary of observation

        Returns:
            None
        """
        self.poses_x = obs_dict["poses_x"]
        self.poses_y = obs_dict["poses_y"]
        self.poses_theta = obs_dict["poses_theta"]
        self.collisions = obs_dict["collisions"]

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            terminated (bool): if the simulation is terminated
            truncated (bool): if the simulation is truncated
            info (dict): auxillary information dictionary
        """

        # call simulation step
        obs = self.sim.step(action)
        obs["lap_times"] = self.lap_times
        obs["lap_counts"] = self.lap_counts

        F110Env.current_obs = obs

        self.render_obs = {
            "ego_idx": obs["ego_idx"],
            "poses_x": obs["poses_x"],
            "poses_y": obs["poses_y"],
            "poses_theta": obs["poses_theta"],
            "lap_times": obs["lap_times"],
            "lap_counts": obs["lap_counts"],
            "scans": obs["scans"],
        }

        # times
        reward = self.timestep
        self.current_time = self.current_time + self.timestep

        # update data member
        self._update_state(obs)

        # check done
        done, toggle_list = self._check_done()
        info = {"checkpoint_done": toggle_list}
        
        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Reset the gym environment by given poses

        Args:
            seed: random seed for the environment
            options: dictionary of options, including 'poses'

        Returns:
            obs (dict): observation of the current step
            info (dict): auxillary information dictionary
        """
        super().reset(seed=seed)
        
        if options is not None:
            poses = options.get("poses")
        else:
            poses = None

        if poses is None:
             # Default poses if not provided (example fallback)
            poses = np.zeros((self.num_agents, 3))
            
        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,), dtype=np.float64)
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,), dtype=np.float64)

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [
                [
                    np.cos(-self.start_thetas[self.ego_idx]),
                    -np.sin(-self.start_thetas[self.ego_idx]),
                ],
                [
                    np.sin(-self.start_thetas[self.ego_idx]),
                    np.cos(-self.start_thetas[self.ego_idx]),
                ],
            ]
        )

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, _, _, _, info = self.step(action)

        self.render_obs = {
            "ego_idx": obs["ego_idx"],
            "poses_x": obs["poses_x"],
            "poses_y": obs["poses_y"],
            "poses_theta": obs["poses_theta"],
            "lap_times": obs["lap_times"],
            "lap_counts": obs["lap_counts"],
            "scans": obs["scans"],
        }

        return obs, info

    def update_map(self, map_path: str, map_ext: str) -> None:
        """
        Updates the map used by simulation

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file

        Returns:
            None
        """
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params: dict[str, Any], index: int = -1) -> None:
        """
        Updates the parameters used by simulation for vehicles

        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """

        F110Env.render_callbacks.append(callback_func)

    def render(self) -> None:
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """

        if F110Env.renderer is None:
            # first call, initialize everything
            from f110_gym.envs.rendering import EnvRenderer

            F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            F110Env.renderer.update_map(self.map_name, self.map_ext)

        F110Env.renderer.update_obs(self.render_obs)

        for render_callback in F110Env.render_callbacks:
            render_callback(F110Env.renderer)

        F110Env.renderer.dispatch_events()
        F110Env.renderer.on_draw()
        F110Env.renderer.flip()
        if self.render_mode == "human":
            time.sleep(1.0 / self.render_fps)
        elif self.render_mode == "human_fast":
            pass
