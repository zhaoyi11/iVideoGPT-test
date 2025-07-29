# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import uuid
import traceback
from collections import defaultdict
import pathlib
import glob
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

np.set_printoptions(precision=3, suppress=True)


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode

def convert(value):
  value = np.array(value)
  if np.issubdtype(value.dtype, np.floating):
    return value.astype(np.float32)
  elif np.issubdtype(value.dtype, np.signedinteger):
    return value.astype(np.int32)
  elif np.issubdtype(value.dtype, np.uint8):
    return value.astype(np.uint8)
  return value


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        # create the base folder for storing the episodes
        self._replay_dir = pathlib.Path(replay_dir).expanduser()
        self._replay_dir.mkdir(parents=True, exist_ok=True)

        self._data_specs = data_specs
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions
    
    def add(self, time_step):
        """ Add one transition to the replay buffer. Save the episode when the episode is done."""
        time_step = time_step._asdict()

        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                assert spec.shape is not None
                value = np.full(spec.shape, value, spec.dtype)
            if isinstance(value, bool):
                value = np.array(value, dtype=np.float32)
            assert spec.dtype == value.dtype, f"Data type mismatch, expected {spec.dtype}, got {value.dtype}."
            if spec.shape is not None: 
                assert spec.shape == value.shape, f"Shape mismatch, expected {spec.shape}, got {value.shape}."

            self._current_episode[spec.name].append(value)

        if time_step['done']:
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)

            self._current_episode = defaultdict(list) # reset the current episode
            self._store_episode(episode)
            self._current_task = None # reset the current task name


    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.rglob('*.npz'):
            _, _, _, eps_len = fn.stem.split('-')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        identifier = str(uuid.uuid4().hex)
        eps_fn = f'{ts}-{identifier}-{eps_idx}-{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)
        return self._replay_dir / eps_fn


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot, demo_path=None):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

        self._num_direct_episodes = 0

        if demo_path is not None:
            files = glob.glob(os.path.join(demo_path, '*.npz'))
            if len(files) == 0:
                assert False
            for display in files:
                display = Path(display)
                if not self._store_episode(display):
                    assert False

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.rglob('*.npz'), reverse=True, key=lambda x: int(x.stem.split('-')[-2])) # sort according to episode index
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('-')[2:]]

            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + int(eps_len) > self._max_size:
                break
            fetched_size += int(eps_len)
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


class ReplaySegmentBuffer(ReplayBuffer):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot, segment_length, demo_path=None):
        super().__init__(replay_dir, max_size, num_workers, nstep, discount,
                         fetch_every, save_snapshot, demo_path)
        self._segment_length = segment_length

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        idx = np.random.randint(1, episode_len(episode) - self._segment_length)
        obs = episode['observation'][idx - 1: idx + self._segment_length - 1, -3:]
        action = episode['action'][idx: idx + self._segment_length]
        reward = episode['reward'][idx: idx + self._segment_length]
        return (obs, action, reward)


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount, demo_path=None):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot,
                            demo_path=demo_path)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader


def make_segment_replay_loader(replay_dir, max_size, batch_size, num_workers,
                               save_snapshot, nstep, discount, segment_length, demo_path=None):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplaySegmentBuffer(replay_dir,
                                   max_size_per_worker,
                                   num_workers,
                                   nstep,
                                   discount,
                                   fetch_every=1000,
                                   save_snapshot=save_snapshot,
                                   segment_length=segment_length,
                                   demo_path=demo_path)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader
