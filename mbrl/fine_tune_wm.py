# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs
from tqdm import tqdm, trange
import time
import imageio
import cv2
import subprocess

import metaworld_env
import drq_utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_segment_replay_loader
from video import TrainVideoRecorder, VideoRecorder
from video_predictor import VideoPredictor
from drqv2 import DrQV2Agent

torch.backends.cudnn.benchmark = True


def make_agent(obs_shape, action_shape, cfg):
    cfg.obs_shape = obs_shape
    cfg.action_shape = action_shape
    return hydra.utils.instantiate(cfg)


def make_video_predictor(cfg):
    return VideoPredictor('cuda', cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        drq_utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.video_predictor = make_video_predictor(self.cfg.world_model)

        self.timer = drq_utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.cfg.task_name = "-".join(self.cfg.task_name.split("_"))

        self.train_env = metaworld_env.make(task_name=self.cfg.task_name, seed=self.cfg.seed, 
                                            frame_stack=self.cfg.frame_stack, action_repeat=self.cfg.action_repeat)
        self.eval_env = metaworld_env.make(task_name=self.cfg.task_name, seed=self.cfg.seed, 
                                           frame_stack=self.cfg.frame_stack, action_repeat=self.cfg.action_repeat)

        # self.train_env = metaworld_env.make(self.cfg.task_name, self.cfg.frame_stack,
        #                                self.cfg.action_repeat, self.cfg.seed, self.cfg.camera, self.cfg.duration, self.cfg.succ_bonus)
        # self.eval_env = metaworld_env.make(self.cfg.task_name, self.cfg.frame_stack,
        #                               self.cfg.action_repeat, self.cfg.seed, self.cfg.camera, self.cfg.duration, self.cfg.succ_bonus)
        # create replay buffer
        obs_shape = self.train_env.obs_space.image.shape
        action_shape = self.train_env.action_space.shape
        data_specs = (metaworld_env.specs(shape=obs_shape, dtype=np.uint8, name='observation'),
                    metaworld_env.specs(shape=action_shape, dtype=np.float32, name='action'), # action is set to None to allow different action shapes
                    metaworld_env.specs(shape=(1,), dtype=np.float32, name='reward'),
                    metaworld_env.specs(shape=(1,), dtype=np.float32, name='discount'))

        # for model training
        self.seg_replay_loader = make_segment_replay_loader(
            # self.work_dir / 'buffer', 
            Path('/data/metaworld_datasets'),
            self.cfg.replay_buffer_size,
            self.cfg.world_model.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount,
            self.cfg.gen_horizon + self.cfg.world_model.context_length, demo_path=None)
        self._seg_replay_iter = None


    @property
    def global_step(self):
        return self._global_step

    def validate(self, global_frame):
        if self._seg_replay_iter is None:
            self._seg_replay_iter = iter(self.seg_replay_loader)
        batch = next(self._seg_replay_iter)
        obs_gt = torch.cat([batch[0][:, :-2], batch[0][:, 1:-1], batch[0][:, 2:]], dim=2)
        action = batch[1][:, 2:]
        reward_gt = batch[2][:, 2:]
        policy = lambda obs, step: action[:, step].to(obs.device)
        start = time.time()
        obs_pred, _, reward_pred = self.video_predictor.rollout(
            obs_gt[:, 0], 
            policy,
            obs_gt.shape[1] - 1,
        )
        obs_mse = ((obs_pred[:, 1:] - (obs_gt[:, 1:]/255.).to(obs_pred.device)) ** 2).mean()
        reward_mse = ((reward_pred[:, 1:] - reward_gt[:, 1:].to(reward_pred.device)) ** 2).mean()
        print(f'validate time: {time.time() - start}')
        
        for i in range(obs_gt.shape[0]):
            gif_path = os.path.join(self.work_dir, 'validate_gif', f'val-sample-{global_frame}-{i}.gif')
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            frames = []
            for t in range(obs_gt.shape[1]):
                frame = (obs_gt[i, t, -3:].permute(1,2,0).detach().cpu().numpy()).astype(np.uint8)
                frame = np.ascontiguousarray(frame)
                frame_pred = (obs_pred[i, t, -3:].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
                frame_pred = np.ascontiguousarray(frame_pred)
                frame_error = np.abs(frame.astype(float) - frame_pred.astype(float)).astype(np.uint8)
                if t > 0:
                    cv2.putText(frame, f'{reward_gt[i, t].item():.2f}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    cv2.putText(frame_pred, f'{reward_pred[i, t].item():.2f}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                frames.append(np.concatenate([frame, frame_pred, frame_error], axis=1))
            imageio.mimsave(gif_path, frames, fps=4, loop=0)
            
        return {
            "val/obs_mse": obs_mse.item(),
            "val/reward_mse": reward_mse.item(),
        }

    def train(self):
        assert self.cfg.num_seed_frames == self.cfg.agent.num_expl_steps * self.cfg.action_repeat

        # predicates
        train_until_step = drq_utils.Until(self.cfg.num_train_steps,
                                           self.cfg.action_repeat,
                                           bar_name='train_step')

        # self.train_video_recorder.init(time_step.observation)

        while train_until_step(self.global_step):
            # save snapshot
            if self.cfg.save_snapshot and self.global_step % 10_000 == 0:
                self.video_predictor.save_snapshot(self.work_dir)
                print(f'saved snapshot at {self.global_step}')
            
            # validate
            if self.global_step % 5_000 == 0:
                metrics = self.validate(self.global_step)
                self.logger.log_metrics(metrics, self.global_step, ty='eval')

            # try to update the agent
            if self._seg_replay_iter is None:
                self._seg_replay_iter = iter(self.seg_replay_loader)
            batch = next(self._seg_replay_iter)
            metrics = self.video_predictor.train(batch)

            # log metrics
            if self.global_step % 1000 == 0:
                metrics = {k + "_init": v for k, v in metrics.items()}
                self.logger.log_metrics(metrics, self.global_step, ty='train')

            self._global_step += 1

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='wm_ft_config')
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()

    # snapshot src code
    os.makedirs('src', exist_ok=True)
    os.system(f"rsync -rv --exclude-from=../../../.gitignore ../../.. src")

    workspace.train()


if __name__ == '__main__':
    main()
