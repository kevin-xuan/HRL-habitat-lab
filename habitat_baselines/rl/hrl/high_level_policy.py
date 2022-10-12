from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from gym import spaces
from habitat.config import Config
from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.ppo.policy import NetPolicy

class HighLevelPolicy:
    def get_next_skill(
        self, observations, rnn_hidden_states, prev_actions, masks, plan_masks
    ) -> Tuple[torch.Tensor, List[Any], torch.BoolTensor]:
        """
        :returns: A tuple containing the next skill index, a list of arguments
            for the skill, and if the high-level policy requests immediate
            termination.
        """


class GtHighLevelPolicy:
    """
    :property _solution_actions: List of tuples were first tuple element is the
        action name and the second is the action arguments.
    """

    _solution_actions: List[Tuple[str, List[str]]]

    def __init__(self, config, task_spec_file, num_envs, skill_name_to_idx):
        with open(task_spec_file, "r") as f:
            task_spec = yaml.safe_load(f)

        self._solution_actions = []
        if "solution" not in task_spec:  # 只有rearrange_easy才给定了解决task的solution
            raise ValueError(
                f"The ground truth task planner only works when the task solution is hard-coded in the PDDL problem file at {task_spec_file}"
            )
        for i, sol_step in enumerate(task_spec["solution"]):
            sol_action = parse_func(sol_step)  # 二元组,第一个为skill fcuntion,比如nav,第二个为skill的参数, 比如机器人位置和物体位置
            self._solution_actions.append(sol_action)
            if i < (len(task_spec["solution"]) - 1):  # 3
                self._solution_actions.append(parse_func("reset_arm(0)"))  # 除最后一个skill之外,每个skill后添加重置robot arm skill
        # Add a wait action at the end.
        self._solution_actions.append(parse_func("wait(30)"))  # 在最后添加wait skill

        self._next_sol_idxs = torch.zeros(num_envs, dtype=torch.int32)  # 表明接下来使用的skill
        self._num_envs = num_envs
        self._skill_name_to_idx = skill_name_to_idx

    def apply_mask(self, mask):
        self._next_sol_idxs *= mask.cpu().view(-1)

    def get_next_skill(
        self, observations, rnn_hidden_states, prev_actions, masks, plan_masks
    ):  # plan_masks是指self._call_high_level,即是否使用high-level policy
        next_skill = torch.zeros(self._num_envs, device=prev_actions.device)
        skill_args_data = [None for _ in range(self._num_envs)]
        immediate_end = torch.zeros(
            self._num_envs, device=prev_actions.device, dtype=torch.bool
        )
        for batch_idx, should_plan in enumerate(plan_masks):  # plan_masks是指self._call_high_level, 即是否使用high-level policy
            if should_plan == 1.0:
                if self._next_sol_idxs[batch_idx] >= len(
                    self._solution_actions
                ):  # 如果next skill的索引超出阈值,则应该快速结束,重置next skill为最后一个skill:wait
                    baselines_logger.info(
                        f"Calling for immediate end with {self._next_sol_idxs[batch_idx]}"
                    )
                    immediate_end[batch_idx] = True
                    use_idx = len(self._solution_actions) - 1
                else:
                    use_idx = self._next_sol_idxs[batch_idx].item()

                skill_name, skill_args = self._solution_actions[use_idx]
                baselines_logger.info(
                    f"Got next element of the plan with {skill_name}, {skill_args}"
                )
                if skill_name not in self._skill_name_to_idx:
                    raise ValueError(
                        f"Could not find skill named {skill_name} in {self._skill_name_to_idx}"
                    )
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]

                skill_args_data[batch_idx] = skill_args

                self._next_sol_idxs[batch_idx] += 1  # 使用一个skill后,下一个正确的skil,即准确知道solution

        return next_skill, skill_args_data, immediate_end

class PPOHighLevelPolicy(NetPolicy):
    def __init__(
        self, 
        config: Config = None, 
        full_config: Config = None, 
        observation_space: spaces.Dict = None,
        action_space = None,
        task_spec_file: str = "", 
        num_envs: int = 1, 
        skill_name_to_idx: Dict = None,
        skill_idx_to_name: Dict = None,
        hidden_size: int = 512,  # 512
        rnn_type: str = "lstm",  # LSTM
        num_recurrent_layers: int = 2,  # 2
        backbone: str = "resnet18",  # resnet18
        normalize_visual_inputs: bool = False,  # False
        force_blind_policy: bool = False,  # False
        policy_config: Config = None,
        fuse_keys: Optional[List[str]] = None,
        rgb_encoder: str = "",
        ):
        with open(task_spec_file, "r") as f:
            task_spec = yaml.safe_load(f)

        self._num_envs = num_envs
        self._skill_name_to_idx = skill_name_to_idx
        self._skill_idx_to_name = skill_idx_to_name
        self.dim_actions=len(self._skill_name_to_idx)

        super().__init__(
            PointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,  # lstm
                backbone=backbone,
                resnet_baseplanes=32,  # 32
                normalize_visual_inputs=normalize_visual_inputs,  # False
                fuse_keys=fuse_keys,  # None
                force_blind_policy=force_blind_policy,  # False
                discrete_actions=True,  # False
                rgb_encoder=rgb_encoder,
                dim_actions = self.dim_actions,
            ),
            dim_actions=self.dim_actions,
            policy_config=policy_config, 
        )


    # def apply_mask(self, mask):
    #     self._next_sol_idxs *= mask.cpu().view(-1)

    def get_next_skill(
        self, observations, rnn_hidden_states, prev_actions, masks, plan_masks, deterministic=False
    ):  # plan_masks是指self._call_high_level,即是否使用high-level policy
        next_skill = torch.zeros(self._num_envs, device=prev_actions.device)
        skill_args_data = [None for _ in range(self._num_envs)]
        immediate_end = torch.zeros(
            self._num_envs, device=prev_actions.device, dtype=torch.bool
        )
        # * 存储high-level policy results
        actions = torch.zeros_like(prev_actions, device=prev_actions.device)
        values = torch.zeros(masks.shape, device=masks.device)
        action_log_probs = torch.zeros(masks.shape, device=masks.device)
        hidden_states = torch.zeros_like(rnn_hidden_states, device=rnn_hidden_states.device)

        for batch_idx, should_plan in enumerate(plan_masks):  # plan_masks是指self._call_high_level, 即是否使用high-level policy
            if should_plan == 1.0:
                features, net_hidden_states = self.net(  # action net
                    observations[batch_idx], rnn_hidden_states[batch_idx], prev_actions[batch_idx], masks[batch_idx]
                )  # (1, 512)  (1, 4, 512)
                distribution = self.action_distribution(features)  
                value = self.critic(features)  # (1, 1)

                if deterministic:  # False
                    if self.action_distribution_type == "categorical":
                        action = distribution.mode()
                    elif self.action_distribution_type == "gaussian":
                        action = distribution.mean
                else:
                    action = distribution.sample()  # (1, 1)

                action_log_prob = distribution.log_probs(action)  # (1, 1)
                
                actions[batch_idx] = action.item()
                values[batch_idx] = value
                action_log_probs[batch_idx] = action_log_prob.item()
                hidden_states[batch_idx] = net_hidden_states

                if action.item() <= 0:  # 如果next skill的索引为0, 代表重置action, 是个dummy action
                    baselines_logger.info(
                        f"Calling for immediate end with {action.item()}"
                    )
                    immediate_end[batch_idx] = True
                    use_idx = self.dim_actions - 1
                else:
                    use_idx = action.item()

                skill_name = self._skill_idx_to_name[use_idx]
                baselines_logger.info(
                    f"Got next element of the plan with {skill_name}"
                )
                if skill_name not in self._skill_name_to_idx:
                    raise ValueError(
                        f"Could not find skill named {skill_name} in {self._skill_name_to_idx}"
                    )
                next_skill[batch_idx] = use_idx

        return next_skill, skill_args_data, immediate_end, values, actions, action_log_probs, hidden_states
    
    @classmethod
    def from_config(
        cls,
        config: Config,
        full_config: Config,
        observation_space: spaces.Dict,
        action_space,
        task_spec_file, 
        num_envs, 
        skill_name_to_idx,
        **kwargs,
    ):
        return cls(
            config=config, 
            full_config=full_config, 
            observation_space=observation_space,
            action_space=action_space,
            task_spec_file=task_spec_file, 
            num_envs=num_envs, 
            skill_name_to_idx=skill_name_to_idx,
            hidden_size=full_config.RL.PPO.hidden_size,  # 512
            rnn_type=full_config.RL.DDPPO.rnn_type,  # LSTM
            num_recurrent_layers=full_config.RL.DDPPO.num_recurrent_layers,  # 2
            backbone=full_config.RL.DDPPO.backbone,  # resnet18
            normalize_visual_inputs="rgb" in observation_space.spaces,  # False
            force_blind_policy=full_config.FORCE_BLIND_POLICY,  # False
            policy_config=full_config.RL.POLICY,
            fuse_keys=None,
            rgb_encoder=full_config.RGB_ENCODER,
        )