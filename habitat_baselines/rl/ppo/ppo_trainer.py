#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from gym import spaces
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, VectorEnv, logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils import profiling_wrapper
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.construct_vector_env import construct_envs
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetPolicy,
)
from habitat_baselines.rl.hrl.hierarchical_policy import (  # noqa: F401.
    HierarchicalPolicy
)
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    batch_obs,
    generate_video,
    get_num_actions,
    is_continuous_action_space,
)


@baseline_registry.register_trainer(name="ddppo")
@baseline_registry.register_trainer(name="ppo")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _is_distributed: bool
    _obs_batching_cache: ObservationBatchingCache
    envs: VectorEnv
    agent: PPO
    actor_critic: NetPolicy

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []

        self._static_encoder = False
        self._encoder = None
        self._obs_space = None

        # Distributed if the world size would be
        # greater than 1
        self._is_distributed = get_distrib_size()[2] > 1
        # self._is_distributed = False
        self._obs_batching_cache = ObservationBatchingCache()

    @property
    def obs_space(self):
        if self._obs_space is None and self.envs is not None:
            self._obs_space = self.envs.observation_spaces[0]

        return self._obs_space

    @obs_space.setter
    def obs_space(self, new_obs_space):
        self._obs_space = new_obs_space

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        if not self._is_distributed: 
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)  # PointNavResNetPolicy or HierarchicalPolicy
        observation_space = self.obs_space  # space.Dict ???????????????????????? ??????????????? ????????????GPS ???????????????sensor ??????GPS ??????sensor ?????????????????? D????????? 
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(  # ???obs_space?????????img transform
            observation_space, self.obs_transforms
        )

        self.actor_critic = policy.from_config( 
            self.config,
            observation_space,
            self.policy_action_space,
            orig_action_space=self.orig_policy_action_space,
            device=self.device,
            envs=self.envs,
        )  # ?????????CNN???RNN??????,???super().__init__?????????action???value head
        self.obs_space = observation_space  # space.Dict
        self.actor_critic.to(self.device)

        if (
            self.config.RL.DDPPO.pretrained_encoder
            or self.config.RL.DDPPO.pretrained
        ):  # False & False
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )  # ???????????????????????????, False

        if self.config.RL.DDPPO.pretrained:  # False
            self.actor_critic.load_state_dict(
                {  # type: ignore
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.DDPPO.pretrained_encoder:  # False
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.DDPPO.train_encoder:  # False
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:  # True
            # nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            # nn.init.constant_(self.actor_critic.critic.fc.bias, 0)
            nn.init.orthogonal_(self.actor_critic._high_level_policy.critic.fc.weight)  # * Hierarchical
            nn.init.constant_(self.actor_critic._high_level_policy.critic.fc.bias, 0)

        self.agent = (DDPPO if self._is_distributed else PPO)(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,  # 0.2
            ppo_epoch=ppo_cfg.ppo_epoch,  # 2
            num_mini_batch=ppo_cfg.num_mini_batch,  # 2
            value_loss_coef=ppo_cfg.value_loss_coef,  # 0.5
            entropy_coef=ppo_cfg.entropy_coef,  # 1e-4
            lr=ppo_cfg.lr,   # 2.5e-4
            eps=ppo_cfg.eps,  # 1e-5
            max_grad_norm=ppo_cfg.max_grad_norm,  # 0.2
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,  # False
        )  # ?????????PPO??????,??????agent???

    def _init_envs(self, config=None):
        if config is None:
            config = self.config

        self.envs = construct_envs(
            config,
            workers_ignore_signals=is_slurm_batch_job(),
        )

    def _init_train(self):
        resume_state = load_resume_state(self.config)  # None
        if resume_state is not None:
            self.config: Config = resume_state["config"]

        if self.config.RL.DDPPO.force_distributed:  # False
            self._is_distributed = True

        if is_slurm_batch_job():  # False
            add_signal_handlers()

        if self._is_distributed:  # False
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.DDPPO.distrib_backend
            )  # distrib_backend=NCCL
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:  # True & False
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,  # -1
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,  # -1
        )

        self._init_envs()

        action_space = self.envs.action_spaces[0]  # spaces.Box (11, ) Arm (7???, [-1, 1]) + Grasp(1???) + Base(2???) + Stop(1???) 
        self.policy_action_space = action_space
        self.orig_policy_action_space = self.envs.orig_action_spaces[0]  # Dict??????
        # if is_continuous_action_space(action_space):  # * ??????????????????????????????,?????????????????????high-level controller
        #     # Assume ALL actions are NOT discrete
        #     action_shape = (get_num_actions(action_space),)  # (11, )
        #     discrete_actions = False
        # else:
        #     # For discrete pointnav
        #     action_shape = (1,)
        #     discrete_actions = True
        action_shape = (1,)
        discrete_actions = True

        ppo_cfg = self.config.RL.PPO
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):  # True & True ??????checkpoint??????
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(ppo_cfg)
        if self._is_distributed:  # False
            self.agent.init_distributed(find_unused_params=True)  # type: ignore

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.actor_critic.parameters())  # * actor_critic?????????????????????
            )
        )

        obs_space = self.obs_space
        if self._static_encoder:  # False
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self._nbuffers = 2 if ppo_cfg.use_double_buffered_sampler else 1  # 1

        self.rollouts = RolloutStorage(  
            ppo_cfg.num_steps,  # 128
            self.envs.num_envs,  # 32
            obs_space,  # space.Dict
            self.policy_action_space,  # space.Dict (11, )
            ppo_cfg.hidden_size,  # 512
            num_recurrent_layers=self.actor_critic._high_level_policy.net.num_recurrent_layers,  # 4
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,  # False
            action_shape=action_shape,  # (11, )
            discrete_actions=discrete_actions,  # False
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()  # list ??????32???orderedDict, ??????Dict??????8???obs, 
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )  # ?????????TensorDict, ???????????????obs??????(robot_head_depth)???tensor???????????????
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        if self._static_encoder:  # False
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.buffers["observations"][0] = batch  # type: ignore 0 ??????128+1 steps???????????????,???rnn?????????

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)  # (32, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)  # 50
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    @rank0_only
    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if not isinstance(k, str) or k in cls.METRICS_BLACKLIST:  # False
                continue

            if isinstance(v, dict):  # False
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if isinstance(subk, str)
                        and k + "." + subk not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs  # 32 - self._paused
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        # sample actions
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idxs[buffer_index],  # 0
                env_slice,
            ]

            profiling_wrapper.range_push("compute actions")
            (
                observations,
                low_not_done_masks,
                low_rewards,
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
                low_infos,
            ) = self.actor_critic.act(
                step_batch["observations"],  # TensorDict
                step_batch["recurrent_hidden_states"],  # tensor
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        self.pth_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        # t_step_env = time.time()  # * ???????????????actions???step

        # for index_env, act in zip(
        #     range(env_slice.start, env_slice.stop), actions.unbind(0)  # ?????????0??????action????????????env???action????????????tuple
        # ):
        #     act = act.cpu().item()
        #     self.envs.async_step_at(index_env, act)

        # outputs = [
        #     self.envs.wait_step_at(index_env)
        #     for index_env in range(env_slice.start, env_slice.stop)
        # ]

        # observations, rewards_l, dones, infos = [
        #     list(x) for x in zip(*outputs)
        # ]

        # self.env_time += time.time() - t_step_env

        # rewards = torch.tensor(
        #     rewards_l,
        #     dtype=torch.float,
        #     device=self.current_episode_reward.device,
        # )
        # not_done_masks = torch.tensor(
        #     [[not done] for done in dones],
        #     dtype=torch.bool,
        #     device=self.current_episode_reward.device,
        # )  # (32, 1)
        # done_masks = torch.logical_not(not_done_masks)  # ???????????????, ??????env???????????????
        
        # * high level???reward??????low level???reward??????
        not_done_masks = low_not_done_masks.to(self.current_episode_reward.device)
        done_masks = torch.logical_not(not_done_masks)  # ???????????????, ??????env???????????????
        self.current_episode_reward[env_slice] += low_rewards.to(self.current_episode_reward.device)
        
        # rewards = rewards + low_rewards.to(self.current_episode_reward.device)
        # self.current_episode_reward[env_slice] += rewards
        current_ep_reward = self.current_episode_reward[env_slice]  # (32, 1)
        # ?????????episode????????????current_ep_reward, ?????????0;????????????episodes???count+1
        self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
        self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(low_infos).items():  # ????????????info_type???value?????????
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:  # True
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )
            self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)  # ??????episodes?????????reward

        if self._static_encoder:  # False
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.pth_time += time.time() - self.pth_time

        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        # batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore
        observations = batch
        # observations['is_holding'] = observations['is_holding'].unsqueeze(1)  # * ???????????????shape???????????????????

        self.rollouts.insert(
            next_observations=observations,
            next_recurrent_hidden_states=recurrent_hidden_states.squeeze(1),
            actions=actions.squeeze(1),  # ??????prev_action
            action_log_probs=actions_log_probs.squeeze(1),
            value_preds=values.squeeze(1),
            rewards=low_rewards,
            next_masks=low_not_done_masks,
            buffer_index=buffer_index,
        )
        self.rollouts.advance_rollout(buffer_index)

        return env_slice.stop - env_slice.start

    def _collect_environment_result(self, buffer_index: int = 0, low_rewards=None):
        num_envs = self.envs.num_envs  # 32 - num_paused
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_step_env = time.time()
        outputs = [
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]

        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        self.env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self.current_episode_reward.device,
        )
        rewards = rewards.unsqueeze(1)  # (32, 1)
        if low_rewards is not None:
            rewards = rewards + low_rewards

        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )  # (32, 1)
        done_masks = torch.logical_not(not_done_masks)  # ???????????????, ??????env???????????????

        self.current_episode_reward[env_slice] += rewards
        current_ep_reward = self.current_episode_reward[env_slice]  # (32, 1)
        # ?????????episode????????????current_ep_reward, ?????????0;????????????episodes???count+1
        self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
        self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():  # ????????????info_type???value?????????
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:  # True
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )
            self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)  # ??????episodes?????????reward

        if self._static_encoder:  # False
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self.rollouts.advance_rollout(buffer_index)

        self.pth_time += time.time() - t_update_stats

        return env_slice.stop - env_slice.start

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(self):
        self._compute_actions_and_step_envs()
        return self._collect_environment_result()

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self):
        ppo_cfg = self.config.RL.PPO
        t_update_model = time.time()
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idx
            ]

            next_value = self.actor_critic._high_level_policy.get_value(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        self.rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )  # ????????????step???returns

        self.agent.train()

        value_loss, action_loss, dist_entropy = self.agent.update(
            self.rollouts
        )

        self.rollouts.after_update()  # ??????rollout_step=0,obs?????????rollout?????????
        self.pth_time += time.time() - t_update_model

        return (
            value_loss,
            action_loss,
            dist_entropy,
        )

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())  # 17???key
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )  # (17, 32, 1)

        stats = self._all_reduce(stats)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        if self._is_distributed:  # False
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            stats /= torch.distributed.get_world_size()

            losses = {
                k: stats[i].item() for i, k in enumerate(loss_name_ordering)
            }

        if self._is_distributed and rank0_only():  # False
            self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        return losses

    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"losses/{k}", v, self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)
        writer.add_scalar("metrics/fps", fps, self.num_steps_done)

        # log stats
        if self.num_updates_done % self.config.LOG_INTERVAL == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )

    def should_end_early(self, rollout_step) -> bool:
        if not self._is_distributed:  # True
            return False
        # This is where the preemption of workers happens.  If a
        # worker detects it will be a straggler, it preempts itself!
        return (
            rollout_step
            >= self.config.RL.PPO.num_steps * self.SHORT_ROLLOUT_THRESHOLD
        ) and int(self.num_rollouts_done_store.get("num_done")) >= (
            self.config.RL.DDPPO.sync_frac * torch.distributed.get_world_size()
        )  # ???DDPPO?????????,?????????rollout_step?????????????????? num_steps????????????????????????????????????worker

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        resume_state = load_resume_state(self.config)  # None
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )

        ppo_cfg = self.config.RL.PPO

        with (
            get_writer(self.config, flush_secs=self.flush_secs)
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if ppo_cfg.use_linear_clip_decay:  # False
                    self.agent.clip_param = ppo_cfg.clip_param * (
                        1 - self.percent_done()
                    )

                if rank0_only() and self._should_save_resume_state():  # False
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                    )

                    save_resume_state(
                        dict(
                            state_dict=self.agent.state_dict(),
                            optim_state=self.agent.optimizer.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():  # False
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                self.agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                for buffer_index in range(self._nbuffers):  # 1 
                    count_steps_delta += self._compute_actions_and_step_envs(buffer_index)  # ??????rollout??????action???hidden_state,????????????(value, reward...)????????????
                    

                for step in range(ppo_cfg.num_steps):  # 128 lstm ??????step??????????????????transition
                    is_last_step = (
                        self.should_end_early(step + 1)
                        or (step + 1) == ppo_cfg.num_steps
                    )

                    for buffer_index in range(self._nbuffers):  # 1
                        # count_steps_delta += self._collect_environment_result(
                        #     buffer_index, low_rewards=low_rewards
                        # )  # ???????????????step?????????

                        if (buffer_index + 1) == self._nbuffers:
                            profiling_wrapper.range_pop()  # _collect_rollout_step

                        if not is_last_step:  # True
                            if (buffer_index + 1) == self._nbuffers:  # True
                                profiling_wrapper.range_push(
                                    "_collect_rollout_step"
                                )

                            count_steps_delta += self._compute_actions_and_step_envs(buffer_index)  # ???????????????rollout

                    if is_last_step:
                        break

                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:  # False
                    self.num_rollouts_done_store.add("num_done", 1)

                (
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent()

                if ppo_cfg.use_linear_lr_decay:  # False
                    lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    dict(
                        value_loss=value_loss,
                        action_loss=action_loss,
                        entropy=dist_entropy,
                    ),
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        # # TODO ?????????,????????????subskills
        # self.config.defrost()
        # self.config.EVAL.SHOULD_LOAD_CKPT = True
        # self.config.EVAL.USE_CKPT_CONFIG = False  # ???True?????????train???config
        # self.config.NUM_ENVIRONMENTS = 1
        # self.config.freeze()
        if self.config.EVAL.SHOULD_LOAD_CKPT:  # True
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
        else:
            ckpt_dict = {}

        if self.config.EVAL.USE_CKPT_CONFIG:  # False
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        # # TODO ??????????????????eval subskills
        # self.config.defrost()
        # self.config.EVAL_CKPT_PATH_DIR = './open_cab_checkpoints'
        # self.config.LOG_FILE = 'open_cab_eval.log'
        # self.config.NUM_ENVIRONMENTS = 1
        # self.config.VIDEO_OPTION = []
        # self.config.TEST_EPISODE_COUNT = 100
        # self.config.freeze()
        
        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if (
            len(self.config.VIDEO_OPTION) > 0
            and self.config.VIDEO_RENDER_TOP_DOWN
        ):  # ['disk'] & False -> False
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if (
            len(config.VIDEO_RENDER_VIEWS) > 0
            and len(self.config.VIDEO_OPTION) > 0
        ):  # [THIRD_RGB_SENSOR] & ["disk"] -> True
            config.defrost()
            for render_view in config.VIDEO_RENDER_VIEWS:
                uuid = config.TASK_CONFIG.SIMULATOR[render_view].UUID
                config.TASK_CONFIG.GYM.OBS_KEYS.append(uuid)
                config.SENSORS.append(render_view)
            config.freeze()

        if config.VERBOSE:  # False
            logger.info(f"env config: {config}")

        self._init_envs(config)

        action_space = self.envs.action_spaces[0]
        self.policy_action_space = action_space
        self.orig_policy_action_space = self.envs.orig_action_spaces[0]
        if is_continuous_action_space(action_space):  # True
            # Assume NONE of the actions are discrete
            action_shape = (get_num_actions(action_space),)
            discrete_actions = False
        else:
            # For discrete pointnav
            action_shape = (1,)
            discrete_actions = True

        self._setup_actor_critic_agent(ppo_cfg)
        # logger.info(f"=======current_ckpt: {checkpoint_path}=======")

        if self.agent.actor_critic.should_load_agent_state:  # TODO Hierarchical???False PointNavPolicy???True ???????????????subskill???????????????
            self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        
        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.actor_critic.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:  # ["disk"]
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)  # "video_dir"

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT  # ?????????-1
        if number_of_eval_episodes == -1:  # True
            number_of_eval_episodes = sum(self.envs.number_of_episodes)  # mini_val: 20
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps  # ?????????dataset??????episodes
        assert (
            number_of_eval_episodes > 1
        ), "The number TEST_EPISODE_COUNT needs to be strictly positive."

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(self.policy_action_space):  # True
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.detach().cpu().numpy(),
                        self.policy_action_space.low,
                        self.policy_action_space.high,
                    )
                    for a in actions
                ]
            else:
                step_data = [a.item() for a in actions.cpu()]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                if len(self.config.VIDEO_OPTION) > 0:  # ['disk']
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    if self.config.VIDEO_RENDER_ALL_INFO:  # True
                        frame = overlay_frame(frame, infos[i])
                    rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:  # ["disk"]
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,  # ['disk']
                            video_dir=self.config.VIDEO_DIR,  # 'vedio_dir'
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,  
                            checkpoint_idx=checkpoint_index,  # 0
                            metrics=self._extract_scalars_from_info(infos[i]),
                            fps=self.config.VIDEO_FPS,  # 30
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.EVAL_KEYS_TO_INCLUDE_IN_NAME,  # ['reward', 'force', 'composite_success']
                        )

                        rgb_frames[i] = []

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":  # False
                        write_gfx_replay(
                            gfx_str,
                            self.config.TASK_CONFIG.TASK,
                            current_episodes[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()
