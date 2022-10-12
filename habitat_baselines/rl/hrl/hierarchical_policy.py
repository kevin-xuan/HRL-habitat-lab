import os.path as osp
from typing import Dict

import gym.spaces as spaces
import torch
import numpy as np
from habitat.core.spaces import ActionSpace
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.high_level_policy import (  # noqa: F401.
    GtHighLevelPolicy,
    HighLevelPolicy,
    PPOHighLevelPolicy,
)
from habitat_baselines.rl.hrl.skills import (  # noqa: F401.
    ArtObjSkillPolicy,
    NavSkillPolicy,
    OracleNavPolicy,
    PickSkillPolicy,
    PlaceSkillPolicy,
    ResetArmSkill,
    SkillPolicy,
    WaitSkillPolicy,
)
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import get_num_actions, batch_obs
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    get_active_obs_transforms,
)
@baseline_registry.register_policy
class HierarchicalPolicy(Policy):
    def __init__(
        self,
        config,  # 里面包含离散的action类别
        full_config,
        observation_space: spaces.Space,
        action_space: ActionSpace,
        org_action_space,
        num_envs: int,
        device=None,
        envs=None,
    ):
        super().__init__()  # pass

        self._action_space = action_space
        self._org_action_space = org_action_space
        self._num_envs: int = num_envs
        self.config = config
        self.full_config = full_config
        self.device = device
        self.envs = envs
        self.obs_transforms = get_active_obs_transforms(self.full_config)

        # Maps (skill idx -> skill)
        self._skills: Dict[int, SkillPolicy] = {}
        self._name_to_idx: Dict[str, int] = {}
        self._idx_to_name: Dict[int, str] = {}
        # 定义每个skill所使用的config、observation、action
        for i, (skill_id, use_skill_name) in enumerate(
            config.USE_SKILLS.items()  
        ):  # dict_items([('pick', 'NN_PICK'), ('place', 'NN_PLACE'), ('nav', 'NN_NAV'), ('nav_to_receptacle', 'NN_NAV'), ('wait', 'WAIT_SKILL'), ('reset_arm', 'RESET_ARM_SKILL')])
            if use_skill_name == "":
                # Skip loading this skill if no name is provided
                continue
            skill_config = config.DEFINED_SKILLS[use_skill_name]  # 包括使用了哪些sensor

            cls = eval(skill_config.skill_name)  # <class 'habitat_baselines.rl.hrl.skills.pick.PickSkillPolicy'>
            skill_policy = cls.from_config(  # 加载pre-train skill
                skill_config,
                observation_space,
                org_action_space,
                self._num_envs,
                full_config,
            )
            self._skills[i + 1] = skill_policy
            self._name_to_idx[skill_id] = i + 1
            self._idx_to_name[i + 1] = skill_id

        self._call_high_level: torch.Tensor = torch.ones(
            self._num_envs, dtype=torch.bool
        )  # 默认为True,为True时调用high-level policy选择next skill
        self._cur_skills: torch.Tensor = torch.zeros(self._num_envs)

        high_level_cls = eval(config.high_level_policy.name)  # 'GtHighLevelPolicy'
        if config.high_level_policy.name == 'GtHighLevelPolicy':
            self._high_level_policy: HighLevelPolicy = high_level_cls(  # 定义high-level的transitions,要改这里
                config.high_level_policy,
                osp.join(
                    full_config.TASK_CONFIG.TASK.TASK_SPEC_BASE_PATH,  # configs/pddl/
                    full_config.TASK_CONFIG.TASK.TASK_SPEC + ".yaml",  # rearrange_easy
                ),
                num_envs,
                self._name_to_idx,
            )
        else:
            self._high_level_policy: HighLevelPolicy = high_level_cls(  # 定义high-level的transitions,要改这里
                config.high_level_policy,
                full_config,
                observation_space,
                org_action_space,
                osp.join(
                    full_config.TASK_CONFIG.TASK.TASK_SPEC_BASE_PATH,  # configs/pddl/
                    full_config.TASK_CONFIG.TASK.TASK_SPEC + ".yaml",  # rearrange_easy
                ),
                num_envs,
                self._name_to_idx,
                self._idx_to_name,
            )
        self._stop_action_idx, _ = find_action_range(
            org_action_space, "REARRANGE_STOP"
        )
        # * 定义low-level policy
        self.reset()

    def reset(self) -> None:  # 重置low-level policy
        self.test_recurrent_hidden_states = torch.zeros(
            self._num_envs,
            self._high_level_policy.num_recurrent_layers,  # 4
            self.full_config.RL.PPO.hidden_size,  # 512
            device=self.device,
        )
        self.not_done_masks = torch.zeros(self._num_envs, 1, device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros(
            self._num_envs,
            get_num_actions(self._action_space),  # 11
            dtype=torch.float32,
            device=self.device,
        )
        self.rewards = torch.zeros(self._num_envs, 1, device=self.device)

    def reset_at(self, batch_idx=0):

        self.test_recurrent_hidden_states[batch_idx].fill_(0)
        self.not_done_masks[batch_idx].fill_(False)
        self.prev_actions[batch_idx].fill_(0)
        self.rewards[batch_idx].fill_(0)
    
    def collect_low_policy_results(self, actions, batch_idx, buffer_index=0):
        # num_envs = self._num_envs  # 32 - self._paused
        actions = actions.unsqueeze(0)
        num_envs = actions.shape[0]
        env_slice = slice(
            int(buffer_index * num_envs),
            int((buffer_index + 1) * num_envs),
        )
        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions.unbind(0)  # 沿着第0维将action分成每个env的action所组成的tuple
        ):
            # Clipping actions to the specified limits [-1, 1]
            act = np.clip(
                act.detach().cpu().numpy(),
                self._action_space.low,
                self._action_space.high,
            )
            self.envs.async_step_at(index_env, act)
        
        outputs = [
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]

        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        batch = batch_obs(
            observations, device=self.device
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self.device,
        )

        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.device,
        )  # (32, 1)

        self.rewards[batch_idx] += rewards
        self.not_done_masks[batch_idx].copy_(not_done_masks.squeeze(1))

        return batch, infos

    def eval(self):
        pass

    @property
    def num_recurrent_layers(self):
        return self._skills[0].num_recurrent_layers

    @property
    def should_load_agent_state(self):
        return False

    def parameters(self):
        return self._high_level_policy.parameters()  # self._skills[0].parameters()

    def to(self, device):
        for skill in self._skills.values():
            skill.to(device)
        self._high_level_policy = self._high_level_policy.to(device)
        self._call_high_level = self._call_high_level.to(device)
        self._cur_skills = self._cur_skills.to(device)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):

        # self._high_level_policy.apply_mask(masks)  # 这里的mask是指not done mask, 如果next_sol为False时意味着需要切换next skill
        use_device = prev_actions.device

        batched_observations = [
            {k: v[batch_idx].unsqueeze(0) for k, v in observations.items()}
            for batch_idx in range(self._num_envs)
        ]
        # * high-level input
        batched_rnn_hidden_states = rnn_hidden_states.unsqueeze(1)  # (32, 1, 4, 512)
        batched_prev_actions = prev_actions.unsqueeze(1)  # (32, 1, 1)
        batched_masks = masks.unsqueeze(1)  # 表示episode是否结束 这里的mask是not done,意味着1代表未完成episode而0则完成 (32, 1, 1)
        
        # * low-level input
        low_batched_rnn_hidden_states = self.test_recurrent_hidden_states.unsqueeze(1)  # (32, 1, 4, 512)
        low_batched_prev_actions = self.prev_actions.unsqueeze(1) # (32, 1, 11)
        low_batched_masks = self.not_done_masks.unsqueeze(1) # (32, 1, 1)
        low_batched_rewards = self.rewards  # (32, 1)

        batched_bad_should_terminate = torch.zeros(
            self._num_envs, device=use_device, dtype=torch.bool
        )  # 表示是否因为超时而结束skill
        
        # * Check if skills should terminate. 应该要用low-level action来判断
        for batch_idx, skill_idx in enumerate(self._cur_skills):
            if low_batched_masks[batch_idx].item() == 0.0:  # episode未完成=0意味着已经完成一个skill,那么就不需要判断是否应该结束该skill而是直接使用next skill
                # Don't check if the skill is done if the episode ended.
                continue
            should_terminate, bad_should_terminate = self._skills[
                skill_idx.item()
            ].should_terminate(  # * 使用low-level hidden_state & actions来判断当前skill是否应该结束
                batched_observations[batch_idx],
                low_batched_rnn_hidden_states[batch_idx],
                low_batched_prev_actions[batch_idx],
                low_batched_masks[batch_idx],
                batch_idx,
            )  # 超时则should_terminate, bad_should_terminate都为1,需要结束当前skill
            batched_bad_should_terminate[batch_idx] = bad_should_terminate
            self._call_high_level[batch_idx] = should_terminate

        # Always call high-level if the episode is over.
        self._call_high_level = self._call_high_level | (~low_batched_masks).view(-1)

        # If any skills want to terminate invoke the high-level policy to get
        # the next skill.
        hl_terminate = torch.zeros(
            self._num_envs, device=use_device, dtype=torch.bool
        )
        if self._call_high_level.sum() > 0:  # True则调用next skill
            (
                new_skills,
                new_skill_args,
                hl_terminate,  # 表明是否强制agent使用wait skill
                high_values, 
                high_actions,
                high_action_log_probs,
                high_hidden_states,
            ) = self._high_level_policy.get_next_skill(  # * 使用high-level hidden_state & actions来判断是否切换next skill
                batched_observations,
                batched_rnn_hidden_states,
                batched_prev_actions,
                batched_masks,
                self._call_high_level,
            )

            for new_skill_batch_idx in torch.nonzero(self._call_high_level):
                skill_idx = new_skills[new_skill_batch_idx.item()]

                skill = self._skills[skill_idx.item()]
                # * 对于新的skill重置low-level的hidden_state和prev_action, 确定skil_args,即参数
                (
                    low_batched_rnn_hidden_states[new_skill_batch_idx],
                    low_batched_prev_actions[new_skill_batch_idx],
                ) = skill.on_enter(
                    new_skill_args[new_skill_batch_idx],
                    new_skill_batch_idx.item(),
                    batched_observations[new_skill_batch_idx],
                    low_batched_rnn_hidden_states,
                    low_batched_prev_actions,
                )
            self._cur_skills = (
                (~self._call_high_level) * self._cur_skills
            ) + (self._call_high_level * new_skills)  # 确定当前skill

        # Compute the actions from the current skills  # * 前面的是为了确定是否更换skill
        actions = torch.zeros_like(self.prev_actions, device=self.prev_actions.device)  # (32, 11)
        # actions.unsqueeze(1) # (32, 1, 11)
        should_terminate = torch.zeros(
            self._num_envs, device=use_device, dtype=torch.bool
        )  # (32, )
        infos_list = [[] for _ in range(self._num_envs)]
        for batch_idx, skill_idx in enumerate(self._cur_skills): # * 这里是low-level的actions
            while (
                should_terminate[batch_idx].item() == False and self._skills[
                    skill_idx.item()]._cur_skill_step[batch_idx] <= 30
            ):
                action_low, rnn_hidden_states_low = self._skills[
                    skill_idx.item()
                ].act(
                    batched_observations[batch_idx],
                    low_batched_rnn_hidden_states[batch_idx],
                    low_batched_prev_actions[batch_idx],
                    low_batched_masks[batch_idx],
                    batch_idx,
                )
                self.prev_actions[batch_idx].copy_(action_low.squeeze(0))
                self.test_recurrent_hidden_states[batch_idx].copy_(rnn_hidden_states_low.squeeze(0))
                actions[batch_idx].copy_(action_low.squeeze(0))

                next_observations, infos = self.collect_low_policy_results(actions[batch_idx], batch_idx)
                batched_observations[batch_idx] = next_observations

                # * low-level input
                low_batched_rnn_hidden_states = self.test_recurrent_hidden_states.unsqueeze(1)  # (1, 1, 4, 512)
                low_batched_prev_actions = self.prev_actions.unsqueeze(1) # (32, 1, 11)
                low_batched_masks = self.not_done_masks.unsqueeze(1) # (32, 1, 1)

                should_terminate, bad_should_terminate = self._skills[
                skill_idx.item()
            ].should_terminate(  # * 使用low-level hidden_state & actions来判断当前skill是否应该结束
                batched_observations[batch_idx],
                low_batched_rnn_hidden_states[batch_idx],
                low_batched_prev_actions[batch_idx],
                low_batched_masks[batch_idx],
                batch_idx,
            ) 
                should_terminate = should_terminate | bad_should_terminate | hl_terminate
            
            low_batched_rewards[batch_idx] = self.rewards[batch_idx]
            batched_masks[batch_idx].fill_(True)
            infos_list[batch_idx] = infos[0]
            self.reset_at(batch_idx)

        should_terminate = batched_bad_should_terminate | hl_terminate
        for batch_idx, terminate_signal in enumerate(should_terminate): 
            if terminate_signal.sum() > 0:
                # self.reset()  # * skill结束时应重置low-level policy hidden_state和action等
                # End the episode where requested.
                baselines_logger.info(
                    f"Calling stop action for batch {batch_idx}, {batched_bad_should_terminate[batch_idx]}, {hl_terminate[batch_idx]}"
                )
                self.prev_actions[batch_idx, self._stop_action_idx] = 1.0

        # batched_observations['is_holding'] = batched_observations['is_holding'].unsqueeze(1)
        batch = batch_obs(
            batched_observations, device=self.device
        )
        observations = [
            {k: v[batch_idx].squeeze(0) for k, v in batch.items()}
            for batch_idx in range(self._num_envs)
        ]

        return (
            observations,
            batched_masks.squeeze(1),
            low_batched_rewards,
            high_values, 
            high_actions,
            high_action_log_probs,
            high_hidden_states,
            infos_list,
        )

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        orig_action_space,
        device,
        envs,
        **kwargs,
    ):
        return cls(
            config.RL.POLICY,  # HierarchicalPolicy
            config,
            observation_space,
            action_space,
            orig_action_space,
            config.NUM_ENVIRONMENTS,  # 1
            device,
            envs,
        )
