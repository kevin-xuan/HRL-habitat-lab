#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F

from habitat.config import Config
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions
# from scripts.test_r3m  import load_r3m
# from r3m import load_r3m
# from vip import load_vip
# import clip
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
# BICUBIC = InterpolationMode.BICUBIC

@baseline_registry.register_policy
class PointNavResNetPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU", 
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: Config = None,
        fuse_keys: Optional[List[str]] = None,
        rgb_encoder: str = "",
        **kwargs,
    ):
        if policy_config is not None:  # True
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"  # False
            )
            self.action_distribution_type = (
                policy_config.action_distribution_type  # gaussian
            )
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"
        # print("PointNavResNetPolicy: {}".format(rgb_encoder))
        super().__init__(
            PointNavResNetNet(  # 初始化CNN和RNN主体,而super().__init__初始化action和value head
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,  # lstm
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,  # 32
                normalize_visual_inputs=normalize_visual_inputs,  # False
                fuse_keys=fuse_keys,  # None
                force_blind_policy=force_blind_policy,  # False
                discrete_actions=discrete_actions,  # False
                rgb_encoder=rgb_encoder,
            ),
            dim_actions=get_num_actions(action_space),  # 11
            policy_config=policy_config,
        )

    @classmethod
    def from_config(
        cls,
        config: Config,
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        if not config.RL.POLICY.get("order_keys", False):
            fuse_keys = config.TASK_CONFIG.GYM.OBS_KEYS
        else:
            fuse_keys = None
        # print("from config: {}".format(config.RGB_ENCODER))
        rgb_encoder = config.get('RGB_ENCODER', "")
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,  # 512
            rnn_type=config.RL.DDPPO.rnn_type,  # LSTM
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,  # 2
            backbone=config.RL.DDPPO.backbone,  # resnet18
            normalize_visual_inputs="rgb" in observation_space.spaces,  # False
            force_blind_policy=config.FORCE_BLIND_POLICY,  # False
            policy_config=config.RL.POLICY,
            fuse_keys=fuse_keys,
            rgb_encoder=rgb_encoder,
        )


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
    ):
        super().__init__()

        # Determine which visual observations are present
        # self.rgb_keys = [k for k in observation_space.spaces if "rgb" in k]  # * [] 虽然指定了rgb,但是不用它的ResNet,而是R3M
        self.rgb_keys = []
        self.depth_keys = [k for k in observation_space.spaces if "depth" in k]  # ['robot-head-depth']

        # Count total # of channels for rgb and for depth
        self._n_input_rgb, self._n_input_depth = [
            # sum() returns 0 for an empty list
            sum(observation_space.spaces[k].shape[2] for k in keys)
            for keys in [self.rgb_keys, self.depth_keys]
        ]  # 0  1

        if normalize_visual_inputs:  # False
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:  # True
            all_keys = self.rgb_keys + self.depth_keys  # ['robot-head-depth']
            spatial_size_h = (
                observation_space.spaces[all_keys[0]].shape[0] // 2
            )  # 256 // 2
            spatial_size_w = (
                observation_space.spaces[all_keys[0]].shape[1] // 2
            )  # 256 // 2
            input_channels = self._n_input_depth + self._n_input_rgb  # 1
            self.backbone = make_backbone(input_channels, baseplanes, ngroups)  # make_backbone=resnet18

            final_spatial_h = int(
                np.ceil(spatial_size_h * self.backbone.final_spatial_compress)
            )  # 4
            final_spatial_w = int(
                np.ceil(spatial_size_w * self.backbone.final_spatial_compress)
            )  # 4
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(
                    after_compression_flat_size
                    / (final_spatial_h * final_spatial_w)
                )
            )  # 128
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial_h,
                final_spatial_w,
            )  # (128, 4, 4)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:  # False
            return None

        cnn_input = []
        for k in self.rgb_keys:  # []
            rgb_observations = observations[k]
            # NHWC -> NCHW permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH] 
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = (
                rgb_observations.float() / 255.0
            )  # normalize RGB
            cnn_input.append(rgb_observations)

        for k in self.depth_keys:
            depth_observations = observations[k]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)  # depth已经归一化[0, 1]之间了
            cnn_input.append(depth_observations)

        x = torch.cat(cnn_input, dim=1)  # 沿通道数拼接
        x = F.avg_pool2d(x, 2)  # [BATCH x CHANNEL x HEIGHT//2 X WIDTH//2]

        x = self.running_mean_and_var(x)
        x = self.backbone(x)  # (32, 1, 128, 128) -> (32, 256, 4, 4)
        x = self.compression(x)  # (32, 128, 4, 4)
        return x


class PointNavResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    prev_action_embedding: nn.Module

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        fuse_keys: Optional[List[str]],
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        rgb_encoder: str = "",
    ):
        super().__init__()  # pass
        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions  # False
        self._n_prev_action = 32  # embedding 维度
        if discrete_actions:  # False
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            num_actions = get_num_actions(action_space)  # 11
            self.prev_action_embedding = nn.Linear(
                num_actions, self._n_prev_action
            )  # 把离散action通过MLP变成continuous
        rnn_input_size = self._n_prev_action  # test

        # Only fuse the 1D state inputs. Other inputs are processed by the
        # visual encoder
        if fuse_keys is None:
            # fuse_keys = observation_space.spaces.keys()  # odict_keys(['is_holding', 'joint', 'obj_goal_gps_compass', 'obj_goal_sensor', 'obj_start_gps_compass', 'obj_start_sensor', 'relative_resting_position', 'robot_head_depth'])
            fuse_keys = sorted(observation_space.spaces.keys())
            # removing keys that correspond to goal sensors
            goal_sensor_keys = {
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid,
                ObjectGoalSensor.cls_uuid,
                EpisodicGPSSensor.cls_uuid,
                PointGoalSensor.cls_uuid,
                HeadingSensor.cls_uuid,
                ProximitySensor.cls_uuid,
                EpisodicCompassSensor.cls_uuid,
                ImageGoalSensor.cls_uuid,
            }  # {'gps', 'objectgoal', 'compass', 'pointgoal', 'proximity', 'imagegoal', 'pointgoal_with_gps_compass', 'heading'}
            fuse_keys = [k for k in fuse_keys if k not in goal_sensor_keys]  # 都不在里面,保持不变 ['is_holding', 'joint', 'obj_goal_gps_compass', 'obj_goal_sensor', 'obj_start_gps_compass', 'obj_start_sensor', 'relative_resting_position', 'robot_head_depth']
        self._fuse_keys_1d: List[str] = [
            k for k in fuse_keys if len(observation_space.spaces[k].shape) == 1
        ]  # 这是过滤掉了'robot_head_depth' obs, 因为其shape是(256, 256, 1)
        if len(self._fuse_keys_1d) != 0:  # 把shape=(x, )的输入与rnn初始输出融合, rnn_input_size = 32+1+7+2+3+2+3+3
            rnn_input_size += sum(
                observation_space.spaces[k].shape[0]
                for k in self._fuse_keys_1d
            )
        # 下面的if都是False
        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):  # False
            n_input_goal = (
                observation_space.spaces[
                    IntegratedPointGoalGPSAndCompassSensor.cls_uuid
                ].shape[0]
                + 1
            )
            self.tgt_embeding = nn.Linear(n_input_goal, 32)
            rnn_input_size += 32

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:  # False
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32

        if PointGoalSensor.cls_uuid in observation_space.spaces:
            input_pointgoal_dim = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
            self.pointgoal_embedding = nn.Linear(input_pointgoal_dim, 32)
            rnn_input_size += 32

        if HeadingSensor.cls_uuid in observation_space.spaces:
            input_heading_dim = (
                observation_space.spaces[HeadingSensor.cls_uuid].shape[0] + 1
            )
            assert input_heading_dim == 2, "Expected heading with 2D rotation."
            self.heading_embedding = nn.Linear(input_heading_dim, 32)
            rnn_input_size += 32

        if ProximitySensor.cls_uuid in observation_space.spaces:
            input_proximity_dim = observation_space.spaces[
                ProximitySensor.cls_uuid
            ].shape[0]
            self.proximity_embedding = nn.Linear(input_proximity_dim, 32)
            rnn_input_size += 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32

        if ImageGoalSensor.cls_uuid in observation_space.spaces:
            goal_observation_space = spaces.Dict(
                {"rgb": observation_space.spaces[ImageGoalSensor.cls_uuid]}
            )
            self.goal_visual_encoder = ResNetEncoder(
                goal_observation_space,
                baseplanes=resnet_baseplanes,  # 32
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
            )

            self.goal_visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.goal_visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

            rnn_input_size += hidden_size

        self._hidden_size = hidden_size  # 512

        if force_blind_policy:  # False
            use_obs_space = spaces.Dict({})
        else:
            use_obs_space = spaces.Dict(
                {
                    k: observation_space.spaces[k]
                    for k in fuse_keys
                    if len(observation_space.spaces[k].shape) == 3
                }
            )  # 只包含robot-head_depth (256, 256, 1)

        self.visual_encoder = ResNetEncoder(
            use_obs_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,  # False
        )
        

        if not self.visual_encoder.is_blind:  # True
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )
        self.test_rgb = rgb_encoder
        # print("PointNavResNetNet: {}".format(rgb_encoder))
        if rgb_encoder == "r3m":
            self.rgb_encoder = load_r3m("resnet34")
            self.image_transform = Compose([
                Resize([256, 256]),
                CenterCrop(224),
                # ToTensor()
            ]) # ToTensor() divides by 255
            self.extra_input = self._hidden_size if rgb_encoder != "resnet50" else self._hidden_size * 2 # resnet18/34输出是512,而resnet50是1024
        elif rgb_encoder == "clip":
            self.rgb_encoder, _ = clip.load("ViT-B/32")
            input_size = self.rgb_encoder.visual.input_resolution  # 224
            self.image_transform = Compose([
                Resize([input_size, input_size], interpolation=BICUBIC),
                CenterCrop(input_size),
                # ToTensor(),
                # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])  # 归一化RGB N x C x H x W
            self.extra_input = self._hidden_size 
        elif rgb_encoder == "vip":
            self.rgb_encoder = load_vip()
            self.image_transform = Compose([
                Resize([256, 256]),
                CenterCrop(224),
                # ToTensor()
            ]) # ToTensor() divides by 255
            self.extra_input = self._hidden_size * 2
        else:
            self.rgb_encoder = None
            self.image_transform = None
            self.extra_input = 0
            
        if self.extra_input != 0:
            self.map_back = nn.Linear(self._hidden_size + self.extra_input, self._hidden_size)
        else:
            self.map_back = nn.Sequential()
        
        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,  # * 注意这里的 * 2,是指用r3m处理的RGB的输出
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()  # self.training = True

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,  # not_done_masks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = []
        if not self.is_blind: # True
            visual_feats = observations.get(
                "visual_features", self.visual_encoder(observations)  # Dict中没有"visual_features",直接调用self.visual_encoder
            )
            visual_feats = self.visual_fc(visual_feats)  # Flatten, Linear, ReLu (32, 512)
            if self.rgb_encoder is not None:
                self.rgb_encoder.eval()
                with torch.no_grad():
                    rgb_input = observations['robot_head_rgb'].permute(0, 3, 1, 2)  # (N, 3, H, W)
                    if self.image_transform is not None:
                        rgb_input = self.image_transform(rgb_input) / 255.0  # 变相于ToTensor() [0, 1]
                        if self.test_rgb != "clip":  # r3m和vip expects image input to be [0-255]
                            rgb_input = rgb_input * 255.0
                            rgb_visual_feats = self.rgb_encoder(rgb_input) # (N, 512)
                        else:
                            clip_norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                            rgb_input = clip_norm(rgb_input)
                            rgb_visual_feats = self.rgb_encoder.encode_image(rgb_input) # (N, 512)
                    # rgb_visual_feats = self.rgb_encoder(rgb_input) # (N, 512)
                visual_feats = torch.cat([visual_feats, rgb_visual_feats], dim=-1)
                visual_feats = self.map_back(visual_feats)
            x.append(visual_feats)

        if len(self._fuse_keys_1d) != 0:  # 将CNN输出与robot state拼接
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys_1d], dim=-1
            )  # (32, 14)
            x.append(fuse_states)

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if goal_observations.shape[1] == 2:
                # Polar Dimensionality 2
                # 2D polar transform
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            else:
                assert (
                    goal_observations.shape[1] == 3
                ), "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                # Polar Dimensionality 3
                # 3D Polar transformation
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.cos(goal_observations[:, 2]),
                    ],
                    -1,
                )

            x.append(self.tgt_embeding(goal_observations))

        if PointGoalSensor.cls_uuid in observations:
            goal_observations = observations[PointGoalSensor.cls_uuid]
            x.append(self.pointgoal_embedding(goal_observations))

        if ProximitySensor.cls_uuid in observations:
            sensor_observations = observations[ProximitySensor.cls_uuid]
            x.append(self.proximity_embedding(sensor_observations))

        if HeadingSensor.cls_uuid in observations:
            sensor_observations = observations[HeadingSensor.cls_uuid]
            sensor_observations = torch.stack(
                [
                    torch.cos(sensor_observations[0]),
                    torch.sin(sensor_observations[0]),
                ],
                -1,
            )
            x.append(self.heading_embedding(sensor_observations))

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )

        if ImageGoalSensor.cls_uuid in observations:
            goal_image = observations[ImageGoalSensor.cls_uuid]
            goal_output = self.goal_visual_encoder({"rgb": goal_image})
            x.append(self.goal_visual_fc(goal_output))

        if self.discrete_actions:  # False
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            # The mask means the previous action will be zero, an extra dummy action
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(
                masks * prev_actions.float()
            )  # (32, 11) -> (32, 32)

        x.append(prev_actions)

        out = torch.cat(x, dim=1)  # (32, 565) obs_ouput + state_ouput + prev_actions
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks
        )

        return out, rnn_hidden_states
