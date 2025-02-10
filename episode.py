from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class EpisodeMetrics:
    episode_length: int
    episode_return: float


@dataclass
class Episode:
    observations: torch.ByteTensor
    actions: torch.LongTensor
    rewards: torch.FloatTensor
    ends: torch.LongTensor
    mask_padding: torch.BoolTensor # True表示对应时刻是轨迹有效的，False表示轨迹已经结束，是padding上去的

    def __post_init__(self):
        assert len(self.observations) == len(self.actions) == len(self.rewards) == len(self.ends) == len(self.mask_padding)
        if self.ends.sum() > 0:
            idx_end = torch.argmax(self.ends) + 1
            self.observations = self.observations[:idx_end]
            self.actions = self.actions[:idx_end]
            self.rewards = self.rewards[:idx_end]
            self.ends = self.ends[:idx_end]
            self.mask_padding = self.mask_padding[:idx_end]

    def __len__(self) -> int:
        return self.observations.size(0)

    def merge(self, other: Episode) -> Episode:
        return Episode(
            torch.cat((self.observations, other.observations), dim=0),
            torch.cat((self.actions, other.actions), dim=0),
            torch.cat((self.rewards, other.rewards), dim=0),
            torch.cat((self.ends, other.ends), dim=0),
            torch.cat((self.mask_padding, other.mask_padding), dim=0),
        )

    # 切割轨迹片段，当轨迹长度不足
    def segment(self, start: int, stop: int, should_pad: bool = False) -> Episode:
        assert start < len(self) and stop > 0 and start < stop
        padding_length_right = max(0, stop - len(self))
        padding_length_left = max(0, -start)
        assert padding_length_right == padding_length_left == 0 or should_pad

        def pad(x):
            pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
            return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

        start = max(0, start)
        stop = min(len(self), stop)
        segment = Episode(
            self.observations[start:stop],
            self.actions[start:stop],
            self.rewards[start:stop],
            self.ends[start:stop],
            self.mask_padding[start:stop],
        )

        segment.observations = pad(segment.observations)
        segment.actions = pad(segment.actions)
        segment.rewards = pad(segment.rewards)
        segment.ends = pad(segment.ends)
        segment.mask_padding = torch.cat((torch.zeros(padding_length_left, dtype=torch.bool), segment.mask_padding, torch.zeros(padding_length_right, dtype=torch.bool)), dim=0)

        return segment

    def compute_metrics(self) -> EpisodeMetrics:
        return EpisodeMetrics(len(self), self.rewards.sum())

    def save(self, path: Path) -> None:
        torch.save(self.__dict__, path)

#### used for ManiSkill2 ####
###### TODO: env state dim varies between different URDF model loading
# @dataclass
# class MS2Episode:
#     obs: torch.FloatTensor
#     actions: torch.FloatTensor
#     states: torch.FloatTensor
#     reward: torch.FloatTensor
#     ends: torch.LongTensor
#     filled: torch.BoolTensor
#     success: torch.BoolTensor

#     def __post_init__(self):
#         assert len(self.obs) == len(self.actions) == len(self.states) == len(self.reward) == len(self.ends) == len(self.filled) == len(self.success)
#         if self.ends.sum() > 0:
#             idx_end = torch.argmax(self.ends) + 1
#             self.obs = self.obs[:idx_end]
#             self.actions = self.actions[:idx_end]
#             self.states = self.states[:idx_end]
#             self.reward = self.reward[:idx_end]
#             self.ends = self.ends[:idx_end]
#             self.filled = self.filled[:idx_end]
#             self.success = self.success[:idx_end]
    
#     @property
#     def state_dim(self):
#         return self.states.shape[-1]
    
#     def segment(self, start: int, stop: int, should_pad: bool = False) -> Episode:
#         assert start < len(self) and stop > 0 and start < stop
#         padding_length_right = max(0, stop - len(self))
#         padding_length_left = max(0, -start)
#         assert padding_length_right == padding_length_left == 0 or should_pad

#         def pad(x):
#             # pad_right 将x的last维度往后pad padding_length_right个长度，value 默认为None
#             pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
#             # return这一行 将x的last维度向前pad padding_length_left个长度，value 默认为None
#             return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

#         start = max(0, start)
#         stop = min(len(self), stop)
#         segment = MS2Episode(
#             self.obs[start:stop],
#             self.actions[start:stop],
#             torch.zeros(size=(stop - start, 1)),  # TODO
#             self.reward[start:stop],
#             self.ends[start:stop],
#             self.filled[start:stop],
#             self.success[start:stop]
#         )

#         segment.obs = pad(segment.obs)
#         segment.actions = pad(segment.actions)
#         segment.states = pad(segment.states)
#         segment.reward = pad(segment.reward)
#         segment.ends = pad(segment.ends)
#         segment.filled = torch.cat((torch.zeros(padding_length_left, dtype=torch.bool), segment.filled, torch.zeros(padding_length_right, dtype=torch.bool)), dim=0)
#         segment.success = torch.cat((torch.zeros(padding_length_left, dtype=torch.bool), segment.success, torch.zeros(padding_length_right, dtype=torch.bool)), dim=0)

#         return segment


### used for SC2 ###
@dataclass
class SC2Episode:
    observation: torch.FloatTensor
    action: torch.FloatTensor
    av_action: torch.FloatTensor
    reward: torch.FloatTensor
    done: torch.FloatTensor
    filled: torch.BoolTensor

    def __post_init__(self):
        assert len(self.observation) == len(self.action) == len(self.av_action) == len(self.reward) == len(self.done) == len(self.filled)
        if self.done.sum() > 0:
            idx_end = self.done.size(0) # torch.argmax(self.done) + 1
            self.observation = self.observation[:idx_end]
            self.action = self.action[:idx_end]
            self.av_action = self.av_action[:idx_end]
            self.reward = self.reward[:idx_end]
            self.done = self.done[:idx_end]
            self.filled = self.filled[:idx_end]
    
    def __len__(self) -> int:
        # accounting for the existence of absorbing state
        num_absorbing_state = int(self.done[:, 0].sum()) #.to(torch.long)
        if num_absorbing_state > 1:
            return self.observation.size(0) - num_absorbing_state + 1
        else:
            return self.observation.size(0)
    
    def segment(self, start: int, stop: int, should_pad: bool = False) -> Episode:
        assert start < len(self) and stop > 0 and start < stop
        padding_length_right = max(0, stop - len(self))
        padding_length_left = max(0, -start)
        assert padding_length_right == padding_length_left == 0 or should_pad

        def pad(x):
            pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
            return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

        start = max(0, start)
        stop = min(len(self), stop)
        segment = SC2Episode(
            self.observation[start:stop],
            self.action[start:stop],
            self.av_action[start:stop],
            self.reward[start:stop],
            self.done[start:stop],
            self.filled[start:stop],
        )

        segment.observation = pad(segment.observation)
        segment.action = pad(segment.action)
        segment.av_action = pad(segment.av_action)
        segment.reward = pad(segment.reward)
        segment.done = pad(segment.done)
        segment.filled = torch.cat((torch.zeros(padding_length_left, dtype=torch.bool), segment.filled, torch.zeros(padding_length_right, dtype=torch.bool)), dim=0)

        return segment
    
@dataclass
class MpeEpisode:
    observation: torch.FloatTensor
    action: torch.FloatTensor
    reward: torch.FloatTensor
    done: torch.FloatTensor
    filled: torch.BoolTensor

    def __post_init__(self):
        assert len(self.observation) == len(self.action) == len(self.reward) == len(self.done) == len(self.filled)
        if self.done.sum() > 0:
            idx_end = self.done.size(0) # torch.argmax(self.done) + 1
            self.observation = self.observation[:idx_end]
            self.action = self.action[:idx_end]
            self.reward = self.reward[:idx_end]
            self.done = self.done[:idx_end]
            self.filled = self.filled[:idx_end]
    
    def __len__(self) -> int:
        # accounting for the existence of absorbing state
        num_absorbing_state = int(self.done[:, 0].sum()) #.to(torch.long)
        if num_absorbing_state > 1:
            return self.observation.size(0) - num_absorbing_state + 1
        else:
            return self.observation.size(0)
    
    def segment(self, start: int, stop: int, should_pad: bool = False) -> Episode:
        assert start < len(self) and stop > 0 and start < stop
        padding_length_right = max(0, stop - len(self))
        padding_length_left = max(0, -start)
        assert padding_length_right == padding_length_left == 0 or should_pad

        def pad(x):
            pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
            return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

        start = max(0, start)
        stop = min(len(self), stop)
        segment = MpeEpisode(
            self.observation[start:stop],
            self.action[start:stop],
            self.reward[start:stop],
            self.done[start:stop],
            self.filled[start:stop],
        )

        segment.observation = pad(segment.observation)
        segment.action = pad(segment.action)
        segment.reward = pad(segment.reward)
        segment.done = pad(segment.done)
        segment.filled = torch.cat((torch.zeros(padding_length_left, dtype=torch.bool), segment.filled, torch.zeros(padding_length_right, dtype=torch.bool)), dim=0)

        return segment

@dataclass
class GRFEpisode:
    observation: torch.FloatTensor
    action: torch.FloatTensor
    reward: torch.FloatTensor
    done: torch.FloatTensor
    filled: torch.BoolTensor

    def __post_init__(self):
        assert len(self.observation) == len(self.action) == len(self.reward) == len(self.done) == len(self.filled)
        if self.done.sum() > 0:
            idx_end = self.done.size(0) # torch.argmax(self.done) + 1
            self.observation = self.observation[:idx_end]
            self.action = self.action[:idx_end]
            self.reward = self.reward[:idx_end]
            self.done = self.done[:idx_end]
            self.filled = self.filled[:idx_end]
    
    def __len__(self) -> int:
        # accounting for the existence of absorbing state
        num_absorbing_state = int(self.done[:, 0].sum()) #.to(torch.long)
        if num_absorbing_state > 1:
            return self.observation.size(0) - num_absorbing_state + 1
        else:
            return self.observation.size(0)
    
    def segment(self, start: int, stop: int, should_pad: bool = False) -> Episode:
        assert start < len(self) and stop > 0 and start < stop
        padding_length_right = max(0, stop - len(self))
        padding_length_left = max(0, -start)
        assert padding_length_right == padding_length_left == 0 or should_pad

        def pad(x):
            pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
            return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

        start = max(0, start)
        stop = min(len(self), stop)
        segment = GRFEpisode(
            self.observation[start:stop],
            self.action[start:stop],
            self.reward[start:stop],
            self.done[start:stop],
            self.filled[start:stop],
        )

        segment.observation = pad(segment.observation)
        segment.action = pad(segment.action)
        segment.reward = pad(segment.reward)
        segment.done = pad(segment.done)
        segment.filled = torch.cat((torch.zeros(padding_length_left, dtype=torch.bool), segment.filled, torch.zeros(padding_length_right, dtype=torch.bool)), dim=0)

        return segment


@dataclass
class MamujocoEpisode:
    observation: torch.FloatTensor
    action: torch.FloatTensor
    reward: torch.FloatTensor
    done: torch.FloatTensor
    filled: torch.BoolTensor

    def __post_init__(self):
        assert len(self.observation) == len(self.action) == len(self.reward) == len(self.done) == len(self.filled)
        if self.done.sum() > 0:
            idx_end = self.done.size(0) # torch.argmax(self.done) + 1
            self.observation = self.observation[:idx_end]
            self.action = self.action[:idx_end]
            self.reward = self.reward[:idx_end]
            self.done = self.done[:idx_end]
            self.filled = self.filled[:idx_end]
    
    def __len__(self) -> int:
        # accounting for the existence of absorbing state
        num_absorbing_state = int(self.done[:, 0].sum()) #.to(torch.long)
        if num_absorbing_state > 1:
            return self.observation.size(0) - num_absorbing_state + 1
        else:
            return self.observation.size(0)
    
    def segment(self, start: int, stop: int, should_pad: bool = False) -> Episode:
        assert start < len(self) and stop > 0 and start < stop
        padding_length_right = max(0, stop - len(self))
        padding_length_left = max(0, -start)
        assert padding_length_right == padding_length_left == 0 or should_pad

        def pad(x):
            pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
            return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

        start = max(0, start)
        stop = min(len(self), stop)
        segment = MpeEpisode(
            self.observation[start:stop],
            self.action[start:stop],
            self.reward[start:stop],
            self.done[start:stop],
            self.filled[start:stop],
        )

        segment.observation = pad(segment.observation)
        segment.action = pad(segment.action)
        segment.reward = pad(segment.reward)
        segment.done = pad(segment.done)
        segment.filled = torch.cat((torch.zeros(padding_length_left, dtype=torch.bool), segment.filled, torch.zeros(padding_length_right, dtype=torch.bool)), dim=0)

        return segment