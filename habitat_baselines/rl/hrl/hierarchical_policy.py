import os.path as osp
from typing import Dict

import gym.spaces as spaces
import torch

from habitat.core.spaces import ActionSpace
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.high_level_policy import (  # noqa: F401.
    GtHighLevelPolicy,
    HighLevelPolicy,
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
from habitat_baselines.utils.common import get_num_actions


@baseline_registry.register_policy
class HierarchicalPolicy(Policy):
    def __init__(
        self,
        config,
        full_config,
        observation_space: spaces.Space,
        action_space: ActionSpace,
        num_envs: int,
    ):
        super().__init__()  # pass

        self._action_space = action_space
        self._num_envs: int = num_envs

        # Maps (skill idx -> skill)
        self._skills: Dict[int, SkillPolicy] = {}
        self._name_to_idx: Dict[str, int] = {}
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
                action_space,
                self._num_envs,
                full_config,
            )
            self._skills[i] = skill_policy
            self._name_to_idx[skill_id] = i

        self._call_high_level: torch.Tensor = torch.ones(
            self._num_envs, dtype=torch.bool
        )  # 默认为True,为True时调用high-level policy选择next skill
        self._cur_skills: torch.Tensor = torch.zeros(self._num_envs)

        high_level_cls = eval(config.high_level_policy.name)  # 'GtHighLevelPolicy'
        self._high_level_policy: HighLevelPolicy = high_level_cls(
            config.high_level_policy,
            osp.join(
                full_config.TASK_CONFIG.TASK.TASK_SPEC_BASE_PATH,  # configs/pddl/
                full_config.TASK_CONFIG.TASK.TASK_SPEC + ".yaml",  # rearrange_easy
            ),
            num_envs,
            self._name_to_idx,
        )
        self._stop_action_idx, _ = find_action_range(
            action_space, "REARRANGE_STOP"
        )

    def eval(self):
        pass

    @property
    def num_recurrent_layers(self):
        return self._skills[0].num_recurrent_layers

    @property
    def should_load_agent_state(self):
        return False

    def parameters(self):
        return self._skills[0].parameters()

    def to(self, device):
        for skill in self._skills.values():
            skill.to(device)
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

        self._high_level_policy.apply_mask(masks)  # 这里的mask是指not done mask, 如果next_sol为False时意味着需要切换next skill
        use_device = prev_actions.device

        batched_observations = [
            {k: v[batch_idx].unsqueeze(0) for k, v in observations.items()}
            for batch_idx in range(self._num_envs)
        ]
        batched_rnn_hidden_states = rnn_hidden_states.unsqueeze(1)  # (1, 1, 4, 512)
        batched_prev_actions = prev_actions.unsqueeze(1)  # (1, 1, 11)
        batched_masks = masks.unsqueeze(1)  # 表示episode是否结束 这里的mask是not done,意味着1代表未完成episode而0则完成 (1, 1, 1)

        batched_bad_should_terminate = torch.zeros(
            self._num_envs, device=use_device, dtype=torch.bool
        )  # 表示是否因为超时而结束skill

        # Check if skills should terminate.
        for batch_idx, skill_idx in enumerate(self._cur_skills):
            if masks[batch_idx] == 0.0:  # episode未完成=0意味着已经完成一个skill,那么就不需要判断是否应该结束该skill而是直接使用next skill
                # Don't check if the skill is done if the episode ended.
                continue
            should_terminate, bad_should_terminate = self._skills[
                skill_idx.item()
            ].should_terminate(
                batched_observations[batch_idx],
                batched_rnn_hidden_states[batch_idx],
                batched_prev_actions[batch_idx],
                batched_masks[batch_idx],
            )  # 超时则should_terminate, bad_should_terminate都为1,需要结束当前skill
            batched_bad_should_terminate[batch_idx] = bad_should_terminate
            self._call_high_level[batch_idx] = should_terminate

        # Always call high-level if the episode is over.
        self._call_high_level = self._call_high_level | (~masks).view(-1)

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
            ) = self._high_level_policy.get_next_skill(
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                self._call_high_level,
            )

            for new_skill_batch_idx in torch.nonzero(self._call_high_level):
                skill_idx = new_skills[new_skill_batch_idx.item()]

                skill = self._skills[skill_idx.item()]
                # 对于新的skill重置hidden_state和prev_action, 确定skil_args,即参数
                (
                    batched_rnn_hidden_states[new_skill_batch_idx],
                    batched_prev_actions[new_skill_batch_idx],
                ) = skill.on_enter(
                    new_skill_args[new_skill_batch_idx],
                    new_skill_batch_idx.item(),
                    batched_observations[new_skill_batch_idx],
                    batched_rnn_hidden_states[new_skill_batch_idx],
                    batched_prev_actions[new_skill_batch_idx],
                )
            self._cur_skills = (
                (~self._call_high_level) * self._cur_skills
            ) + (self._call_high_level * new_skills)  # 确定当前skill

        # Compute the actions from the current skills 前面的是为了确定是否更换skill
        actions = torch.zeros(
            self._num_envs, get_num_actions(self._action_space)
        )
        for batch_idx, skill_idx in enumerate(self._cur_skills):
            action, batched_rnn_hidden_states[batch_idx] = self._skills[
                skill_idx.item()
            ].act(
                batched_observations[batch_idx],
                batched_rnn_hidden_states[batch_idx],
                batched_prev_actions[batch_idx],
                batched_masks[batch_idx],
                batch_idx,
            )
            actions[batch_idx] = action

        should_terminate = batched_bad_should_terminate | hl_terminate
        if should_terminate.sum() > 0:
            # End the episode where requested.
            for batch_idx in torch.nonzero(should_terminate):
                baselines_logger.info(
                    f"Calling stop action for batch {batch_idx}, {bad_should_terminate}, {hl_terminate}"
                )
                actions[batch_idx, self._stop_action_idx] = 1.0

        return (
            None,
            actions,
            None,
            batched_rnn_hidden_states.view(rnn_hidden_states.shape),
        )

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        orig_action_space,
        **kwargs,
    ):
        return cls(
            config.RL.POLICY,  # HierarchicalPolicy
            config,
            observation_space,
            orig_action_space,
            config.NUM_ENVIRONMENTS,  # 1
        )
