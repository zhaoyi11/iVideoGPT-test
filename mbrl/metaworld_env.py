from collections import OrderedDict, deque, namedtuple
from typing import Any, NamedTuple
import os
import abc
import re

import numpy as np
import gymnasium as gym

MAX_EPISODE_LENGTH_DMC = 1000
MAX_EPISODE_LENGTH_MW = 200
MAX_EPISODE_LENGTH_RS = 200

# 50 meta world tasks in total, 1 embodiment
_MetaWorld_TASK_SET = (
    'assembly', 'basketball', 'bin-picking', 'box-close', 'button-press', 
    'button-press-topdown', 'button-press-topdown-wall', 'button-press-wall', 'coffee-button', 'coffee-pull', 
    'coffee-push', 'dial-turn', 'disassemble', 'door-close', 'door-lock', 
    'door-open', 'door-unlock', 'drawer-close', 'drawer-open', 'faucet-open', 
    'faucet-close', 'hammer', 'hand-insert', 'handle-press-side', 'handle-press', 
    'handle-pull-side', 'handle-pull', 'lever-pull', 'peg-insert-side', 'peg-unplug-side',
    'pick-out-of-hole', 'pick-place-wall', 'pick-place', 'plate-slide-back-side', 'plate-slide-back', 
    'plate-slide-side', 'plate-slide', 'push-back', 'push-wall', 'push', 
    'reach', 'reach-wall', 'shelf-place', 'soccer', 'stick-push', 
    'stick-pull', 'sweep-into', 'sweep', 'window-close', 'window-open',
)
MetaWorld_TASK_SET = tuple([f'mw-{name}' for name in _MetaWorld_TASK_SET])

# 30 dm control tasks in total, 6 embodiments
_DMControl_TASK_SET = (
    'cartpole-balance', 'cartpole-swingup', 'cartpole-swingup-sparse', 'cartpole-swingup-hard', # 4
    'acrobot-swingup', 'acrobot-swingup-sparse', 'acrobot-swingup-hard', # 3
    'cheetah-run', 'cheetah-run-hard', 'cheetah-run-backwards', 'cheetah-jump', 'cheetah-run-back', 'cheetah-run-front', # 6
    'walker-stand', 'walker-walk', 'walker-walk-hard', 'walker-run', 'walker-run-hard', 'walker-walk-backwards', 'walker-run-backwards', 'walker-backflip', # 8
    # 'hopper-stand', 'hopper-hop', 'hopper-flip', 'hopper-hop-backwards', # 4
    'quadruped-stand', 'quadruped-walk', 'quadruped-run', 'quadruped-jump', 'quadruped-roll', 'quadruped-roll-fast', # 6
    'humanoid-stand', 'humanoid-walk', 'humanoid-run', # 3
)
DMControl_TASK_SET = tuple([f'dmc-{name}' for name in _DMControl_TASK_SET])

# 10 robosuite tasks in total, 5 embodiments
# TODO: check the camera view for each task
_RoboSuite_TASK_SET = (
    'kinova3-lift', 'kinova3-door', 'kinova3-pickplace',
    'iiwa-lift', 'iiwa-door', 'iiwa-pickplace',
    'omron-lift', 'omron-door',
    'b1z1f-door',
    'h1a-peg',
)
RoboSuite_TASK_SET = tuple([f'rs-{name}' for name in _RoboSuite_TASK_SET])

TASK_SET = (*DMControl_TASK_SET, *MetaWorld_TASK_SET, *RoboSuite_TASK_SET)
TASK_DICT = {k: i for i, k in enumerate(TASK_SET)}

# action dim
_DOMAIN_ACT_DIM = {'dmc-cartpole': 1, 'dmc-acrobot': 1, 'dmc-cheetah': 6, 'dmc-walker': 6, 'dmc-quadruped': 12, 'dmc-humanoid': 21, 
                  'dmc-hopper': 4, 'rs-kinova3': 7, 'rs-panda': 7, 'rs-iiwa': 7, 'rs-omron': 11, 'rs-h1a': 12, 'rs-b1z1f': 10, 'mw-': 4}
# get action dim for each task
TASK_ACT_DIM = {task: _DOMAIN_ACT_DIM.get(re.match(r'^([\w]+-[\w]+)', task).group(), 4) for task in TASK_SET} # the default action dim is 4 (for metaworld tasks)
MAX_ACTION_DIM = max(_DOMAIN_ACT_DIM.values())

observation_space = namedtuple('observation_space', ['image', 'state'])
specs = namedtuple('specs', ['shape', 'dtype', 'name']) 
Timestep = namedtuple('Timestep', ['observation', 'state', 'reward', 'action',
                                    'done', 'discount', 'is_first', 'is_last', 'info']) # done: whether an episode ends, is_last: whether an episode is truncated, discount: whether an episode is terminated (1. if not terminated, 0. if terminated)

class Env(abc.ABC):
    @abc.abstractmethod
    def reset():
        pass

    @abc.abstractmethod
    def step():
        pass

    @abc.abstractmethod
    def close():
        pass

    @property
    @abc.abstractmethod
    def obs_space():
        pass

    @property
    @abc.abstractmethod
    def action_space():
        pass

    @property
    @abc.abstractmethod
    def episode_length():
        pass

class MetaWorld(Env):
    def __init__(self, task_name, seed, obs_type='pixels', 
                    width=64, height=64, 
                    camera_name='corner2', render_mode='rgb_array'):
        import metaworld, copy
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        
        env_id = task_name.split('-', 1)[-1]
        task = f'{env_id}-v2-goal-observable'
        if not task_name.startswith('mw-') or task not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
            raise ValueError('Unknown task:', task_name)

        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed, width=width, height=height, camera_name=camera_name, render_mode=render_mode)
        # self._env = copy.deepcopy(env)
        self._env._freeze_rand_vec = False

        self._obs_type = obs_type
        self._camera_name = camera_name

    def reset(self):
        if self._camera_name == 'corner2':
            # from https://github.com/XuGW-Kevin/DrM/blob/main/metaworld_env.py 
            self._env.model.cam_pos[2] = [0.75, 0.075, 0.7] # bring the camera closer to the object
        
        state, info = self._env.reset()
        state = state.astype(np.float32)
        # reset viewer https://github.com/Farama-Foundation/Gymnasium/issues/736
        if self._env.mujoco_renderer.viewer is not None:
            self._env.mujoco_renderer.viewer.make_context_current()

        if self._obs_type == 'pixels':
            image = self._env.render()[::-1].copy() # flip the image, due to the difference in coordinate system of opengl and opencv
        else:
            image = None
        return Timestep(observation=image, state=state, reward=0.0,
                        action=np.zeros_like(self._env.action_space.sample()),
                         done=False, is_first=True, is_last=False, discount=1.0, info=info)

    def step(self, action):
        assert np.isfinite(action).all(), action
        state, reward, terminated, truncated, info = self._env.step(action)
        state = state.astype(np.float32)
        done = terminated or truncated
        if self._obs_type == 'pixels':
            image = self._env.render()[::-1].copy()
        else:
            image = None

        return Timestep(observation=image, state=state, reward=reward, action=action, 
                        done=done, is_first=False, is_last=truncated, discount=float(not terminated), info=info)

    def close(self):
        self._env.close()

    @property
    def obs_space(self):
        return observation_space(image=specs(shape=(self._env.height, self._env.width, 3), dtype=np.uint8, name='image'), 
                                state=specs(shape=(self._env.observation_space.shape[0],), dtype=np.float32, name='state'))

    @property
    def action_space(self):
        return specs(shape=(self._env.action_space.shape[0],), dtype=np.float32, name='action')

    @property
    def episode_length(self): # TODO:
        return self._env.episode_length

class RoboSuite(Env):
    def __init__(self, task_name, seed, obs_type='pixels',
                        width=64, height=64, camera_name='agentview', render_mode='rgb_array'):
        
        import robosuite as suite
        from robosuite.robots import register_robot_class
        from robosuite.controllers import load_composite_controller_config
        from robosuite.models.robots import GR1ArmsOnly
        from robosuite_models.robots import UR5eOmron
        from robosuite_models.robots import Arx5, Z1
        from robosuite_models.robots import SpotArm
        import warnings
        warnings.filterwarnings('ignore')

        @register_robot_class("LeggedRobot")
        class B1Z1Floating(Z1):
            """
            Variant of B1Z1 robot with floating base. Currently serves as placeholder class.
            """

            @property
            def default_base(self):
                return "B1Floating"

            @property
            def base_xpos_offset(self):
                return {
                    "bins": (-0.8, -0.1, 0.8),
                    "empty": (-0.8, 0, 0.8),
                    "table": lambda table_length: (-0.55 - table_length / 2, 0.0, 0.8),
                }
                
        task_map = {'lift': 'Lift', 'door': 'Door', 'pickplace': 'PickPlaceBread', 'stack': 'Stack', 'assembly': 'NutAssemblySquare', 'peg': 'TwoArmPegInHole'}
        robot_map = {'kinova3': 'Kinova3', 'panda': 'Panda', 'iiwa': 'IIWA', 'omron': 'UR5eOmron', 'h1a': 'H1ArmsOnly', 'b1z1f': 'B1Z1Floating'}

        env_id = task_name.split('-', 1)[-1]
        robot, task = env_id.split('-')

        if not task_name.startswith('rs-') or (task not in task_map.keys()) or (robot not in robot_map.keys()):
            raise ValueError('Unknown task:', task_name) 

        if obs_type == 'pixels':
            camera_kwargs = {
                'has_offscreen_renderer': True,
                'use_camera_obs': True,
                'camera_heights': height,
                'camera_widths': width,
                'camera_names': camera_name, # 'agentview', 'frontview'
            }
        else:
            camera_kwargs = {
                'has_offscreen_renderer': False,
                'use_camera_obs': False,
            }
        env = suite.make(
            env_name=task_map[task], robots=robot_map[robot], 
            controller_configs=load_composite_controller_config(controller='BASIC'), # BASIC controller: arms controlled using OSC, mobile base (if present) using JOINT_VELOCITY, other parts controlled using JOINT_POSITION 
            has_renderer=False,
            ignore_done=True, # ignore @horizon, https://robosuite.ai/docs/simulation/environment.html
            hard_reset=True, 
            use_object_obs=True,
            env_configuration='opposed',
            control_freq=20, reward_shaping=True, reward_scale=1.0,
            **camera_kwargs
        )

        self._env = env
        
        self._obs_type = obs_type
        self._camera_name = camera_name
        self._width = width
        self._height = height
        # state keys
        keys = ['object-state']
        for idx in range(len(env.robots)):
            keys.append(f'robot{idx}_proprio-state')
        self._keys = keys
        _ = self.reset()
        
    def reset(self):
        _state = self._env.reset()
        state = np.concatenate([_state[key] for key in self._keys], axis=-1)
        if self._obs_type == 'pixels':
            image = _state[f'{self._camera_name}_image'][::-1]
        else:
            image = None
        self._state_shape = state.shape
        return Timestep(observation=image, state=state, reward=0.0,
                        action=np.zeros((self._env.action_dim,), dtype=np.float32),
                        done=False, is_first=True, is_last=False, discount=1.0, info=None)

    def step(self, action):
        _state, reward, done, info = self._env.step(action) 
        state = np.concatenate([_state[key] for key in self._keys], axis=-1)
        if self._obs_type == 'pixels':
            image = _state[f'{self._camera_name}_image'][::-1]
        else:
            image = None
        # since we set ignore_done=True, which means the horizon is ignored, done state here is for termination (not truncation)
        return Timestep(observation=image, state=state, reward=reward, action=action,
                        done=done, is_first=False, is_last=False, discount=float(not done), info=info) 

    def close(self):
        self._env.close()

    @property
    def obs_space(self):
        return observation_space(image=specs(shape=(self._width, self._height, 3), dtype=np.uint8, name='image'), 
                                state=specs(shape=self._state_shape, dtype=np.float32, name='state'))

    @property
    def action_space(self):
        return specs(shape=(self._env.action_dim,), dtype=np.float32, name='action')

    @property
    def episode_length(self): # TODO:
        return self._env.episode_length

class DMControl(Env):
    def __init__(self, task_name, seed, obs_type='pixels', 
                    width=64, height=64,
                ):
        from dm_control import suite
        from dm_control.suite.wrappers import pixels
        import custom_dmc_tasks as cdmc

        
        domain, task = task_name.split('-', 1)[-1].split('-', 1)
        domain = dict(cup='ball_in_cup', point='point_mass').get(domain, domain)
        task = task.replace('-', '_')
        
        env = suite.load(
            domain, task,
            task_kwargs=dict(random=seed),
            environment_kwargs=dict(flat_observation=True),
            visualize_reward=False,
        )

        # pixel observation
        if obs_type == 'pixels':
            camera_id = dict(quadruped=2).get(domain, 0)
            render_kwargs = dict(width=width, height=height, camera_id=camera_id)
            env = pixels.Wrapper(env, 
                                pixels_only=False,
                                render_kwargs=render_kwargs)

        self._env = env
        self._obs_type = obs_type
        env._width = width
        env._height = height

    def reset(self):
        timestep = self._env.reset()
        if self._obs_type == 'pixels':
            image = timestep.observation['pixels']
        else:
            image = None
        return Timestep(observation=image, state=timestep.observation['observations'], reward=0., 
                        action=np.zeros(self._env.action_spec().shape, dtype=np.float32),
                        done=False, is_first=timestep.first(), is_last=timestep.last(), discount=1.0, info=None)

    def step(self, action):
        timestep = self._env.step(action)
        
        if self._obs_type == 'pixels':
            image = timestep.observation['pixels']
        else:
            image = None
        return Timestep(observation=image, state=timestep.observation['observations'], reward=timestep.reward, action=action, 
                        done=timestep.last(), is_first=timestep.first(), is_last=timestep.last(), discount=timestep.discount, info=None)

    def close(self):
        self._env.close()

    @property
    def obs_space(self):
        return observation_space(image=specs(shape=self._env.observation_spec()['pixels'].shape, dtype=np.uint8, name='image'), 
                                state=specs(shape=self._env.observation_spec()['observations'].shape, dtype=np.float32, name='state'))

    @property
    def action_space(self):
        return specs(shape=(self._env.action_spec().shape[0],), dtype=np.float32, name='action')

    @property
    def episode_length(self):
        return self._env._max_episode_steps

class TimeLimitWrapper(Env):
    """ Terminate an episode after a fixed number of steps. """
    def __init__(self, env, max_episode_steps):
        self._env = env
        self._max_episode_steps = max_episode_steps
        self.curr_steps = 0

    def reset(self):
        self.curr_steps = 0
        return self._env.reset()
    
    def step(self, action):
        timestep = self._env.step(action)
        self.curr_steps += 1
        truncated = (self.curr_steps >= self._max_episode_steps)
        terminated = (timestep.discount == 0.0)
        done = truncated or terminated
        return timestep._replace(done=done, is_last=truncated, discount=timestep.discount)

    def close(self):
        self._env.close()

    @property
    def obs_space(self):
        return self._env.obs_space
    
    @property
    def action_space(self):
        return self._env.action_space

    @property
    def episode_length(self):
        return self._max_episode_steps

class ActionRepeatWrapper(Env):
    """ Repeat the same action for a fixed number of times. """
    def __init__(self, env, repeat):
        self._env = env
        self._repeat = repeat
        assert hasattr(self._env, 'curr_steps'), 'The environment must have max_episode_steps attribute (Try to call TimeLimitWrapper first).'

    def reset(self):
        return self._env.reset()

    def step(self, action):
        reward = 0.
        done = False
        for _ in range(self._repeat):
            timestep = self._env.step(action)
            reward += (timestep.reward or 0.0) * float(not done)
            done = done or timestep.done
            if done:
                break
        return timestep._replace(reward=reward, done=done)

    def close(self):
        self._env.close()

    @property
    def obs_space(self):
        return self._env.obs_space

    @property   
    def action_space(self):
        return self._env.action_space

    @property
    def episode_length(self):
        return self._env.episode_length

class SmoothActionWrapper(Env):
    """ Smooth the action by averaging the action over a fixed window. """
    def __init__(self, env, window_size=5, penalty_ratio=1e-3):
        self._env = env
        self._window_size = window_size
        self._actions = deque(maxlen=window_size)
        self._penalty_ratio = penalty_ratio

    def reset(self):
        self._actions.clear()
        timestep = self._env.reset()
        self._actions.append(np.zeros_like(timestep.action))
        return timestep

    def step(self, action):
        """ Add an action penalty to prevent the action change too fast. """
        pre_action = np.mean(self._actions, axis=0)
        diff_action = action - pre_action
        penalty = -np.linalg.norm(diff_action)
        timestep = self._env.step(action)
        timestep = timestep._replace(reward=timestep.reward + self._penalty_ratio * penalty)
        self._actions.append(action)
        return timestep

    def close(self):
        self._env.close()

    @property
    def obs_space(self):
        return self._env.obs_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def episode_length(self):
        return self._env.episode_length

class FrameStackWrapper(Env):
    """ Stack consecutive frames as a single observation. """
    def __init__(self, env, num_stack=1):
        self._env = env
        self._num_stack = num_stack
        self._frames = deque(maxlen=num_stack)
    
    def reset(self):
        timestep = self._env.reset()
        if timestep.observation is not None:
            for _ in range(self._num_stack):
                self._frames.append(timestep.observation)
            return timestep._replace(observation=np.concatenate(list(self._frames), axis=-1))
        return timestep
    
    def step(self, action):
        timestep = self._env.step(action)
        if timestep.observation is not None:
            self._frames.append(timestep.observation)
            return timestep._replace(observation=np.concatenate(list(self._frames), axis=-1))
        return timestep

    def close(self):
        self._env.close()

    @property
    def obs_space(self):
        image_shape = self._env.obs_space.image.shape
        return observation_space(image=specs(shape=(image_shape[0], image_shape[1], image_shape[2] * self._num_stack), dtype=np.uint8, name='image'), 
                                state=self._env.obs_space.state)
    
    @property
    def action_space(self):
        return self._env.action_space

    @property
    def episode_length(self):
        return self._env.episode_length

class HardTaskWrapper(Env):
    """ Penalize actions and use sparse rewards for hard tasks. """
    def __init__(self, env, reward_threshold=0.0, action_penalty=0.0):
        self._env = env
        self._reward_threshold = reward_threshold
        self._action_penalty = action_penalty
    
    def reset(self):
        return self._env.reset()
    
    def step(self, action):
        timestep = self._env.step(action)
        reward = timestep.reward
        if reward < self._reward_threshold:
            reward = 0.
        reward = reward - self._action_penalty * np.linalg.norm(action)
        return timestep._replace(reward=reward)

    def close(self):
        self._env.close()

    @property
    def obs_space(self):
        return self._env.obs_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def episode_length(self):
        return self._env.episode_length


class FormatOutputWrapper(Env):
    """ Covert output to float32, and permute image to CHW. """
    def __init__(self, env):
        self._env = env

    def _format_obs(self, timestep):
        if timestep.observation is not None:
            image = np.transpose(timestep.observation, (2, 0, 1)) # uint8, HWC -> CHW
        else:
            image = None
        if timestep.state is not None:
            state = timestep.state.astype(np.float32)
        else:
            state = None
        return timestep._replace(observation=image, state=state)

    def reset(self):
        timestep = self._env.reset()
        return self._format_obs(timestep)

    def step(self, action):
        timestep = self._env.step(action)
        return self._format_obs(timestep)

    def close(self):
        self._env.close()
    
    @property
    def obs_space(self):
        obs_space = self._env.obs_space
        # permute image to CHW
        return obs_space._replace(image=specs(shape=(obs_space.image.shape[2], obs_space.image.shape[0], obs_space.image.shape[1]),
                                 dtype=np.uint8, name='image'))

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def episode_length(self):
        return self._env.episode_length

def make(task_name, seed, obs_type="pixels", action_repeat=1, frame_stack=1,  width=64, height=64, reward_threshold=0.0, action_penalty=0.0):
    """ Make a single environment. """
    _hard_task = False
    if task_name.endswith('-hard'):
        _task_name = task_name[:-5]
        _hard_task = True
    else:
        _task_name = task_name
        _hard_task = False

    if task_name.startswith('mw-'):
        env = MetaWorld(_task_name, seed, obs_type=obs_type, width=width, height=height)
        max_episode_frames = MAX_EPISODE_LENGTH_MW
    elif task_name.startswith('rs-'):
        env= RoboSuite(_task_name, seed, obs_type=obs_type, width=width, height=height)
        env = SmoothActionWrapper(env, window_size=5, penalty_ratio=1e-3) # encourage smooth actions for robosuite tasks
        max_episode_frames = MAX_EPISODE_LENGTH_RS
    elif task_name.startswith('dmc-'):
        env = DMControl(_task_name, seed, obs_type=obs_type, width=width, height=height)
        max_episode_frames = MAX_EPISODE_LENGTH_DMC
    else:
        raise ValueError('Unknown task:', task_name)

    # wrap the environment
    if _hard_task:
        env = HardTaskWrapper(env, reward_threshold=reward_threshold, action_penalty=action_penalty)

    env = TimeLimitWrapper(env, max_episode_frames)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FrameStackWrapper(env, frame_stack)
    env = FormatOutputWrapper(env)

    return env


def make_env_dict(cfg, seed):
    """ Make a dictionary of environments. """
    assert cfg.obs_type in ('pixels', 'state')
    # ! Note that metaworld tasks are not included for multitask co-training due to its rendering issue.

    if cfg.task == 'mt-4':
        tasks = ['dmc-walker-run', 'dmc-cartpole-swingup', 'dmc-cheetah-run', 'dmc-quadruped-walk']
    elif cfg.task == 'mt-10':
        tasks = ['dmc-acrobot-swingup', 'dmc-cheetah-run', 'dmc-quadruped-run', 'dmc-hopper-hop', 'dmc-walker-run',
                # 'mw-bin-picking', 'mw-box-close',  'mw-door-close', 'mw-drawer-close',
                'rs-panda-lift', 'rs-iiwa-door',   'omron-door', 'rs-b1z1f-door', 'rs-h1-peg']
    elif cfg.task == 'mt-all':
        tasks = (*(f'dmc-{name}' for name in DMControl_TASK_SET), 
            *(f'rs-{name}' for name in RoboSuite_TASK_SET)) 
    elif not isinstance(cfg.task, str):
        import omegaconf
        tasks = omegaconf.OmegaConf.to_object(cfg.task)
        assert 'mw-' not in '\t'.join(tasks), 'Metaworld tasks are not included for multitask co-training.'
    else: # single task
        tasks = [cfg.task]

    envs = {}
    for task in tasks:
        env = make(task, seed, obs_type=cfg.obs_type, action_repeat=cfg.action_repeat, frame_stack=cfg.frame_stack, 
                    width=cfg.img_size, height=cfg.img_size, reward_threshold=cfg.reward_threshold, action_penalty=cfg.action_penalty)
        envs[task] = env
    return envs

