from typing import Any, List, Tuple

import gym.spaces as spaces
import torch

from habitat.tasks.rearrange.rearrange_sensors import IsHoldingSensor
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import get_num_actions


class SkillPolicy(Policy):
    def __init__(
        self,
        config,
        action_space: spaces.Space,
        batch_size,
        should_keep_hold_state: bool = False,
    ):
        """
        :param action_space: The overall action space of the entire task, not task specific.
        """
        self._config = config
        self._batch_size = batch_size  # 1

        self._cur_skill_step = torch.zeros(self._batch_size)
        self._should_keep_hold_state = should_keep_hold_state  # False arm在navskill, waitskill和resetArm下为True

        self._cur_skill_args: List[Any] = [
            None for _ in range(self._batch_size)
        ]

        self._grip_ac_idx = 0
        found_grip = False
        for k, space in action_space.items():
            if k != "ARM_ACTION":
                self._grip_ac_idx += get_num_actions(space)
            else:
                # The last actioin in the arm action is the grip action.
                self._grip_ac_idx += get_num_actions(space) - 1
                found_grip = True
                break
        if not found_grip:  # False
            raise ValueError(f"Could not find grip action in {action_space}")

    def _internal_log(self, s, observations=None):
        baselines_logger.debug(
            f"Skill {self._config.skill_name} @ step {self._cur_skill_step}: {s}"
        )

    def _get_multi_sensor_index(self, batch_idx: int, sensor_name: str) -> int:
        """
        Gets the index to select the observation object index in `_select_obs`.
        Used when there are multiple possible goals in the scene, such as
        multiple objects to possibly rearrange.
        """
        if self._cur_skill_args[batch_idx] is not None:
            return self._cur_skill_args[batch_idx]
        return 0

    def _keep_holding_state(
        self, full_action: torch.Tensor, observations
    ) -> torch.Tensor:
        """
        Makes the action so it does not result in dropping or picking up an
        object. Used in navigation and other skills which are not supposed to
        interact through the gripper.
        """
        # Keep the same grip state as the previous action.
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        # If it is not holding (0) want to keep releasing -> output -1.
        # If it is holding (1) want to keep grasping -> output +1.
        full_action[:, self._grip_ac_idx] = is_holding + (is_holding - 1.0)
        return full_action

    def should_terminate(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx=None,
    ) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
        """
        :returns: A (batch_size,) size tensor where 1 indicates the skill wants to end and 0 if not.
        """
        is_skill_done = self._is_skill_done(
            observations, rnn_hidden_states, prev_actions, masks, batch_idx,
        )  # 1表示skill完成,0则未完成
        if is_skill_done.sum() > 0:  # False
            self._internal_log(
                f"Requested skill termination {is_skill_done}",
                observations,
            )
        if batch_idx is None:
            bad_terminate = torch.zeros(
                self._cur_skill_step.shape,
                device=self._cur_skill_step.device,
                dtype=torch.bool,
            )  # 1代表因此超时而导致skill结束
        else:
            bad_terminate = torch.zeros(
                self._cur_skill_step[[batch_idx]].shape,
                device=self._cur_skill_step.device,
                dtype=torch.bool,
            ) 
        if self._config.MAX_SKILL_STEPS > 0:  # True
            over_max_len = self._cur_skill_step > self._config.MAX_SKILL_STEPS
            if self._config.FORCE_END_ON_TIMEOUT:  # nav、wait、reserArm为False,其余为True
                bad_terminate = over_max_len
            else:
                is_skill_done = is_skill_done | over_max_len  # 超时则为1

        if bad_terminate.sum() > 0:
            self._internal_log(
                f"Bad terminating due to timeout {self._cur_skill_step}, {bad_terminate}",
                observations,
            )

        return is_skill_done, bad_terminate

    def on_enter(
        self,
        skill_arg: List[str],  # None
        batch_idx: int,
        observations,
        rnn_hidden_states,
        prev_actions,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes in the data at the current `batch_idx`
        :returns: The new hidden state and prev_actions ONLY at the batch_idx.
        """
        self._cur_skill_step[batch_idx] = 0
        self._cur_skill_args[batch_idx] = self._parse_skill_arg(skill_arg)

        self._internal_log(
            f"Entering skill with arguments {skill_arg} parsed to {self._cur_skill_args[batch_idx]}",
            observations,
        )
        
        # prev_actions[batch_idx] = prev_actions[batch_idx] * 0.0 
        # tmp_actions = prev_actions[batch_idx].clone().long()
        return (
            rnn_hidden_states[batch_idx] * 0.0,
            prev_actions[batch_idx] * 0.0, 
        )

    @classmethod
    def from_config(
        cls, config, observation_space, action_space, batch_size, full_config
    ):
        return cls(config, action_space, batch_size)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        """
        :returns: Predicted action and next rnn hidden state.
        """
        self._cur_skill_step[cur_batch_idx] += 1
        action, hxs = self._internal_act(  # 返回action和hidden_state
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,  # False
        )

        if self._should_keep_hold_state:  # 如果True,那么gripper的状态要保持不变,即-1代表not holding,而1代表holding
            action = self._keep_holding_state(action, observations)
        return action, hxs

    def to(self, device):
        self._cur_skill_step = self._cur_skill_step.to(device)

    def _select_obs(self, obs, cur_batch_idx):
        """
        Selects out the part of the observation that corresponds to the current goal of the skill.
        """
        for k in self._config.OBS_SKILL_INPUTS:
            cur_multi_sensor_index = self._get_multi_sensor_index(
                cur_batch_idx, k
            )  # 0
            if cur_multi_sensor_index is None:
                cur_multi_sensor_index = 0
            if k not in obs:
                raise ValueError(
                    f"Skill {self._config.skill_name}: Could not find {k} out of {obs.keys()}"
                )
            entity_positions = obs[k].view(
                1, -1, self._config.get("OBS_SKILL_INPUT_DIM", 3)
            )  # (1, 2) -> (1, 1, 2)
            obs[k] = entity_positions[:, cur_multi_sensor_index]  # (1, 2) 感觉这两步没啥意义
        return obs

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx=None,
    ) -> torch.BoolTensor:
        """
        :returns: A (batch_size,) size tensor where 1 indicates the skill wants to end and 0 if not.
        """
        return torch.zeros(observations.shape[0], dtype=torch.bool).to(
            masks.device
        )

    def _parse_skill_arg(self, skill_arg: str) -> Any:
        """
        Parses the skill argument string identifier and returns parsed skill argument information.
        """
        return skill_arg

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()
