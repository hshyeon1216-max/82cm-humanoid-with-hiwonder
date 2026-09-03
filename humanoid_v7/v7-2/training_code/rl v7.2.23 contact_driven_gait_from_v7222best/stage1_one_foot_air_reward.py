from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils

from .one_foot_terms import _body_id, _foot_contact_forces


MIN_REWARD_SCORE = 0.001
MAX_REWARD_SCORE = 1.0
ROBOT_HEIGHT_M = 0.824
DEFAULT_PELVIS_HEIGHT_M = 0.719
MIN_PELVIS_HEIGHT_M = DEFAULT_PELVIS_HEIGHT_M - 0.10
PRESSURE_SENSOR_CAPACITY_N = 490.3325
ROBOT_MASS_KG = 6.993
ROBOT_WEIGHT_N = ROBOT_MASS_KG * 9.80665
# This robot's CAD/URDF frame uses X as left-right, Y as front-back, Z as up.
# The foot mesh protrudes farther toward negative Y, so robot-local forward is -Y.
# Walking rewards use the initial BNO085/root heading as the forward reference.
# This prevents the policy from turning first and then treating the new heading as forward.
FORWARD_AXIS = 1
FORWARD_SIGN = -1.0
LATERAL_AXIS = 0
FOOT_GAP_AXIS = 1
FOOT_CORNER_XY = (
    (-0.03, -0.055),  # front_left
    (0.03, -0.055),  # front_right
    (-0.03, 0.045),  # rear_left
    (0.03, 0.045),  # rear_right
)
# Foot gap rewards use center-to-center distance, not edge-to-edge distance.
# The center is defined as the mean of the four pressure sensor corner positions.
FOOT_CENTER_OFFSET_XY = (0.0, -0.005)


def _step_dt(env: ManagerBasedEnv) -> float:
    return float(getattr(env, "step_dt", 1.0 / 60.0))


def _episode_time(env: ManagerBasedEnv) -> torch.Tensor:
    return env.episode_length_buf.float() * _step_dt(env)


def _phase(env: ManagerBasedEnv, gait_freq: float) -> torch.Tensor:
    return 2.0 * math.pi * gait_freq * _episode_time(env)


def gait_phase_obs(env: ManagerBasedEnv, gait_freq: float = 1.2) -> torch.Tensor:
    phase = _phase(env, gait_freq)
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


def fixed_base_velocity_command(
    env: ManagerBasedEnv,
    lin_vel_x: float = 0.25,
    lin_vel_y: float = 0.0,
    ang_vel_z: float = 0.0,
) -> torch.Tensor:
    command = torch.empty(env.num_envs, 1, device=env.device)
    command[:, 0] = lin_vel_x
    return command


def _clamp_score(score: torch.Tensor) -> torch.Tensor:
    return torch.clamp(score, min=MIN_REWARD_SCORE, max=MAX_REWARD_SCORE)


def _exp_score(error: torch.Tensor, sigma: float) -> torch.Tensor:
    return _clamp_score(torch.exp(-torch.square(error) / max(sigma * sigma, 1.0e-8)))


def _exp_score_l2(error_l2: torch.Tensor, sigma: float) -> torch.Tensor:
    return _clamp_score(torch.exp(-error_l2 / max(sigma * sigma, 1.0e-8)))


def _root_frame_foot_positions(env: ManagerBasedRLEnv, asset: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
    left_pos_w = asset.data.body_pos_w[:, _body_id(asset, "left_foot_1"), :3]
    right_pos_w = asset.data.body_pos_w[:, _body_id(asset, "right_foot_1"), :3]
    root_pos_w = asset.data.root_pos_w[:, :3]
    left_rel_w = left_pos_w - root_pos_w
    right_rel_w = right_pos_w - root_pos_w
    left_rel_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, left_rel_w)
    right_rel_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, right_rel_w)
    return left_rel_b, right_rel_b


def _foot_center_pos_w(asset: Articulation, body_id: int) -> torch.Tensor:
    center_offset = torch.zeros_like(asset.data.body_pos_w[:, body_id, :3])
    center_offset[:, 0] = FOOT_CENTER_OFFSET_XY[0]
    center_offset[:, 1] = FOOT_CENTER_OFFSET_XY[1]
    return asset.data.body_pos_w[:, body_id, :3] + math_utils.quat_apply(
        asset.data.body_quat_w[:, body_id],
        center_offset,
    )


def _root_frame_foot_center_positions(env: ManagerBasedRLEnv, asset: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
    left_center_w = _foot_center_pos_w(asset, _body_id(asset, "left_foot_1"))
    right_center_w = _foot_center_pos_w(asset, _body_id(asset, "right_foot_1"))
    root_pos_w = asset.data.root_pos_w[:, :3]
    left_rel_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, left_center_w - root_pos_w)
    right_rel_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, right_center_w - root_pos_w)
    return left_rel_b, right_rel_b


def _root_forward_velocity(env: ManagerBasedRLEnv, asset: Articulation, body_id: int) -> torch.Tensor:
    local_forward = torch.zeros(env.num_envs, 3, device=env.device)
    local_forward[:, FORWARD_AXIS] = FORWARD_SIGN
    forward_w = math_utils.quat_apply(asset.data.root_quat_w, local_forward)
    body_vel_w = asset.data.body_lin_vel_w[:, body_id, :3]
    return torch.sum(body_vel_w * forward_w, dim=1)


def _current_forward_w(env: ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    local_forward = torch.zeros(env.num_envs, 3, device=env.device)
    local_forward[:, FORWARD_AXIS] = FORWARD_SIGN
    forward_w = math_utils.quat_apply(asset.data.root_quat_w, local_forward)
    forward_w[:, 2] = 0.0
    return torch.nn.functional.normalize(forward_w, dim=-1)


def _initial_forward_w(env: ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    current_forward = _current_forward_w(env, asset)
    stored = getattr(env, "_pleas_initial_forward_w", None)
    if stored is None or stored.shape != current_forward.shape or stored.device != current_forward.device:
        stored = current_forward.detach().clone()
    reset_mask = env.episode_length_buf <= 1
    if torch.any(reset_mask):
        stored[reset_mask] = current_forward[reset_mask].detach()
    env._pleas_initial_forward_w = stored
    return stored


def _heading_locked_forward_speed(asset: Articulation, desired_forward_w: torch.Tensor) -> torch.Tensor:
    return torch.sum(asset.data.root_lin_vel_w[:, :3] * desired_forward_w, dim=1)


def _heading_locked_body_forward_velocity(
    asset: Articulation,
    body_id: int,
    desired_forward_w: torch.Tensor,
) -> torch.Tensor:
    return torch.sum(asset.data.body_lin_vel_w[:, body_id, :3] * desired_forward_w, dim=1)


def _heading_locked_forward_position(
    asset: Articulation,
    body_id: int,
    desired_forward_w: torch.Tensor,
) -> torch.Tensor:
    relative_pos_w = asset.data.body_pos_w[:, body_id, :3] - asset.data.root_pos_w[:, :3]
    return torch.sum(relative_pos_w * desired_forward_w, dim=1)


def _heading_locked_foot_center_forward_position(
    asset: Articulation,
    body_id: int,
    desired_forward_w: torch.Tensor,
) -> torch.Tensor:
    relative_pos_w = _foot_center_pos_w(asset, body_id) - asset.data.root_pos_w[:, :3]
    return torch.sum(relative_pos_w * desired_forward_w, dim=1)


def _heading_locked_lateral_w(desired_forward_w: torch.Tensor) -> torch.Tensor:
    lateral_w = torch.zeros_like(desired_forward_w)
    lateral_w[:, 0] = -desired_forward_w[:, 1]
    lateral_w[:, 1] = desired_forward_w[:, 0]
    return torch.nn.functional.normalize(lateral_w, dim=-1)


def _heading_locked_lateral_speed(asset: Articulation, desired_forward_w: torch.Tensor) -> torch.Tensor:
    lateral_w = _heading_locked_lateral_w(desired_forward_w)
    return torch.sum(asset.data.root_lin_vel_w[:, :3] * lateral_w, dim=1)


def _heading_locked_lateral_position(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    desired_forward_w: torch.Tensor,
) -> torch.Tensor:
    lateral_w = _heading_locked_lateral_w(desired_forward_w)
    relative_pos_w = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    return torch.sum(relative_pos_w * lateral_w, dim=1)


def _heading_lock_score(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    desired_forward_w: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    current_forward = _current_forward_w(env, asset)
    cross_z = desired_forward_w[:, 0] * current_forward[:, 1] - desired_forward_w[:, 1] * current_forward[:, 0]
    dot_xy = desired_forward_w[:, 0] * current_forward[:, 0] + desired_forward_w[:, 1] * current_forward[:, 1]
    heading_error = torch.atan2(cross_z, dot_xy)
    return _exp_score(heading_error, sigma)


def _world_forward_velocity(asset: Articulation, body_id: int) -> torch.Tensor:
    return FORWARD_SIGN * asset.data.body_lin_vel_w[:, body_id, FORWARD_AXIS]


def _world_forward_position(asset: Articulation, body_id: int) -> torch.Tensor:
    return FORWARD_SIGN * asset.data.body_pos_w[:, body_id, FORWARD_AXIS]


def _root_frame_forward_position(rel_pos_b: torch.Tensor) -> torch.Tensor:
    return FORWARD_SIGN * rel_pos_b[:, FORWARD_AXIS]


def _heading_alignment_score(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    sigma: float,
) -> torch.Tensor:
    local_forward = torch.zeros(env.num_envs, 3, device=env.device)
    local_forward[:, FORWARD_AXIS] = FORWARD_SIGN
    forward_w = math_utils.quat_apply(asset.data.root_quat_w, local_forward)
    desired_forward_w = torch.zeros_like(forward_w)
    desired_forward_w[:, FORWARD_AXIS] = FORWARD_SIGN
    cross_z = desired_forward_w[:, 0] * forward_w[:, 1] - desired_forward_w[:, 1] * forward_w[:, 0]
    dot_xy = desired_forward_w[:, 0] * forward_w[:, 0] + desired_forward_w[:, 1] * forward_w[:, 1]
    heading_error = torch.atan2(cross_z, dot_xy)
    return _exp_score(heading_error, sigma)


def _pelvis_reference_height(env: ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    body_names = asset.data.body_names
    pelvis_ids = [idx for idx, name in enumerate(body_names) if name in ("left_pelvis_pitch_1", "right_pelvis_pitch_1")]
    if not pelvis_ids:
        pelvis_ids = [idx for idx, name in enumerate(body_names) if name == "pelvis_1"]
    if not pelvis_ids:
        return asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    pelvis_z = asset.data.body_pos_w[:, pelvis_ids, 2].mean(dim=1)
    return pelvis_z - env.scene.env_origins[:, 2]


def _foot_corner_ratios(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    foot_id: int,
    tilt_gain: float = 18.0,
) -> torch.Tensor:
    corners = torch.tensor(FOOT_CORNER_XY, device=env.device, dtype=asset.data.root_pos_w.dtype)
    corners = corners.unsqueeze(0).expand(env.num_envs, -1, -1)
    gravity_w = torch.zeros(env.num_envs, 3, device=env.device, dtype=asset.data.root_pos_w.dtype)
    gravity_w[:, 2] = -1.0
    gravity_foot = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, foot_id], gravity_w)
    # A tilted foot loads the downhill corners. Flat foot -> uniform 0.25 each.
    logits = tilt_gain * torch.sum(corners * gravity_foot[:, None, :2], dim=-1)
    return torch.softmax(logits, dim=-1)


def _corner_pressure_from_force(force: torch.Tensor, ratios: torch.Tensor) -> torch.Tensor:
    force = torch.clamp(force, min=0.0, max=4.0 * PRESSURE_SENSOR_CAPACITY_N)
    corner_force = force.unsqueeze(-1) * ratios
    return torch.clamp(corner_force, min=0.0, max=PRESSURE_SENSOR_CAPACITY_N)


def _left_right_corner_pressures(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    asset: Articulation = env.scene["robot"]
    left_force, right_force = _foot_contact_forces(env, sensor_cfg)
    left_id = _body_id(asset, "left_foot_1")
    right_id = _body_id(asset, "right_foot_1")
    left_pressure = _corner_pressure_from_force(left_force, _foot_corner_ratios(env, asset, left_id))
    right_pressure = _corner_pressure_from_force(right_force, _foot_corner_ratios(env, asset, right_id))
    return left_pressure, right_pressure


def foot_corner_pressure_obs(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces",
        body_names=["left_foot_1", "right_foot_1"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    left_pressure, right_pressure = _left_right_corner_pressures(env, sensor_cfg)
    # Scale like the existing foot force observation: Newtons * 0.01.
    return torch.cat((left_pressure, right_pressure), dim=-1) * 0.01


def _pressure_tilt_from_corners(corner_pressure: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.clamp(torch.sum(corner_pressure, dim=1), min=1.0)
    front = corner_pressure[:, 0] + corner_pressure[:, 1]
    rear = corner_pressure[:, 2] + corner_pressure[:, 3]
    left = corner_pressure[:, 0] + corner_pressure[:, 2]
    right = corner_pressure[:, 1] + corner_pressure[:, 3]
    pressure_pitch = torch.clamp((front - rear) / total, min=-1.0, max=1.0)
    pressure_roll = torch.clamp((right - left) / total, min=-1.0, max=1.0)
    return pressure_roll, pressure_pitch


def foot_pressure_tilt_estimate_obs(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces",
        body_names=["left_foot_1", "right_foot_1"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    left_pressure, right_pressure = _left_right_corner_pressures(env, sensor_cfg)
    left_roll, left_pitch = _pressure_tilt_from_corners(left_pressure)
    right_roll, right_pitch = _pressure_tilt_from_corners(right_pressure)
    return torch.stack((left_roll, left_pitch, right_roll, right_pitch), dim=-1)


def pressure_cop_xy_obs(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces",
        body_names=["left_foot_1", "right_foot_1"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    left_pressure, right_pressure = _left_right_corner_pressures(env, sensor_cfg)
    corners = torch.tensor(FOOT_CORNER_XY, device=env.device, dtype=left_pressure.dtype)
    scale = torch.tensor((0.03, 0.055), device=env.device, dtype=left_pressure.dtype)

    def _cop(pressure: torch.Tensor) -> torch.Tensor:
        total = torch.sum(pressure, dim=1, keepdim=True)
        cop_xy = torch.sum(pressure.unsqueeze(-1) * corners.unsqueeze(0), dim=1) / torch.clamp(total, min=1.0)
        cop_xy = torch.where(total > 1.0, cop_xy, torch.zeros_like(cop_xy))
        return torch.clamp(cop_xy / scale, min=-1.0, max=1.0)

    return torch.cat((_cop(left_pressure), _cop(right_pressure)), dim=-1)


def pressure_front_rear_balance_obs(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces",
        body_names=["left_foot_1", "right_foot_1"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    left_pressure, right_pressure = _left_right_corner_pressures(env, sensor_cfg)
    _, left_pitch = _pressure_tilt_from_corners(left_pressure)
    _, right_pitch = _pressure_tilt_from_corners(right_pressure)
    return torch.stack((left_pitch, right_pitch), dim=-1)


def pressure_left_right_balance_obs(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces",
        body_names=["left_foot_1", "right_foot_1"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    left_pressure, right_pressure = _left_right_corner_pressures(env, sensor_cfg)
    left_roll, _ = _pressure_tilt_from_corners(left_pressure)
    right_roll, _ = _pressure_tilt_from_corners(right_pressure)
    return torch.stack((left_roll, right_roll), dim=-1)


def _action_term_joint_ids(env: ManagerBasedRLEnv) -> list[int] | slice:
    action_term = env.action_manager.get_term("joint_pos")
    joint_ids = getattr(action_term, "_joint_ids", slice(None))
    if isinstance(joint_ids, slice):
        return joint_ids
    if isinstance(joint_ids, torch.Tensor):
        return [int(item) for item in joint_ids.detach().cpu().tolist()]
    return [int(item) for item in joint_ids]


def joint_position_error_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    action_term = env.action_manager.get_term("joint_pos")
    joint_ids = _action_term_joint_ids(env)
    target_pos = action_term.processed_actions
    current_pos = asset.data.joint_pos[:, joint_ids]
    return torch.clamp(target_pos - current_pos, min=-1.5, max=1.5)


def joint_limit_margin_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _action_term_joint_ids(env)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    limits = asset.data.soft_joint_pos_limits[:, joint_ids, :]
    lower = limits[:, :, 0]
    upper = limits[:, :, 1]
    half_range = torch.clamp(0.5 * (upper - lower), min=1.0e-5)
    margin = torch.minimum(joint_pos - lower, upper - joint_pos) / half_range
    return torch.clamp(margin, min=0.0, max=1.0)


def _pressure_flat_contact_score(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    left_force: torch.Tensor,
    right_force: torch.Tensor,
    contact_threshold: float,
    sigma: float = 0.35,
) -> torch.Tensor:
    left_id = _body_id(asset, "left_foot_1")
    right_id = _body_id(asset, "right_foot_1")
    left_ratios = _foot_corner_ratios(env, asset, left_id)
    right_ratios = _foot_corner_ratios(env, asset, right_id)
    ideal = torch.full_like(left_ratios, 0.25)
    left_imbalance = torch.mean(torch.abs(left_ratios - ideal), dim=1) / 0.375
    right_imbalance = torch.mean(torch.abs(right_ratios - ideal), dim=1) / 0.375
    left_score = _exp_score(left_imbalance, sigma)
    right_score = _exp_score(right_imbalance, sigma)
    left_contact = (left_force > contact_threshold).float()
    right_contact = (right_force > contact_threshold).float()
    contact_count = torch.clamp(left_contact + right_contact, min=1.0)
    contact_weighted_score = (left_score * left_contact + right_score * right_contact) / contact_count
    no_contact_score = torch.ones_like(contact_weighted_score)
    return torch.where((left_contact + right_contact) > 0.0, contact_weighted_score, no_contact_score)


def _joint_group_ids(asset: Articulation, tokens: tuple[str, ...]) -> list[int]:
    return [index for index, name in enumerate(asset.data.joint_names) if any(token in name for token in tokens)]


def _joint_safe_score(asset: Articulation, tokens: tuple[str, ...], safe_range: float) -> torch.Tensor:
    ids = _joint_group_ids(asset, tokens)
    if not ids:
        return torch.ones(asset.data.joint_pos.shape[0], device=asset.data.joint_pos.device)
    deviation = torch.abs(asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids])
    excess = torch.clamp((deviation - safe_range) / max(safe_range, 1.0e-6), min=0.0)
    return _exp_score_l2(torch.mean(torch.square(excess), dim=1), sigma=1.0)


def _motor_safe_joint_usage_score(
    asset: Articulation,
    hip_roll_safe_range: float = 0.12,
    hip_pitch_safe_range: float = 0.35,
    hip_yaw_safe_range: float = 0.18,
    knee_safe_range: float = 0.45,
    ankle_pitch_safe_range: float = 0.16,
    ankle_roll_safe_range: float = 0.12,
) -> torch.Tensor:
    group_scores = (
        _joint_safe_score(asset, ("185", "163"), hip_roll_safe_range),
        _joint_safe_score(asset, ("188", "166"), hip_pitch_safe_range),
        _joint_safe_score(asset, ("192", "172"), hip_yaw_safe_range),
        _joint_safe_score(asset, ("196", "176"), knee_safe_range),
        _joint_safe_score(asset, ("199", "179"), ankle_pitch_safe_range),
        _joint_safe_score(asset, ("202", "182"), ankle_roll_safe_range),
    )
    return _clamp_score(torch.stack(group_scores, dim=1).mean(dim=1))


def _roll_suppression_score(
    asset: Articulation,
    hip_roll_safe_range: float = 0.035,
    ankle_roll_safe_range: float = 0.035,
) -> torch.Tensor:
    hip_roll_score = _joint_safe_score(asset, ("185", "163"), hip_roll_safe_range)
    ankle_roll_score = _joint_safe_score(asset, ("202", "182"), ankle_roll_safe_range)
    return _clamp_score(0.5 * (hip_roll_score + ankle_roll_score))


def _hip_yaw_suppression_score(asset: Articulation, hip_yaw_safe_range: float = 0.055) -> torch.Tensor:
    return _joint_safe_score(asset, ("192", "172"), hip_yaw_safe_range)


def _joint_abs_deviation(asset: Articulation, tokens: tuple[str, ...]) -> torch.Tensor:
    ids = _joint_group_ids(asset, tokens)
    if not ids:
        return torch.zeros(asset.data.joint_pos.shape[0], device=asset.data.joint_pos.device)
    deviation = torch.abs(asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids])
    return torch.mean(deviation, dim=1)


def _joint_velocity_regularization_score(asset: Articulation, velocity_scale: float) -> torch.Tensor:
    scaled_vel = asset.data.joint_vel / max(velocity_scale, 1.0e-6)
    return _exp_score_l2(torch.mean(torch.square(scaled_vel), dim=1), sigma=1.0)


def _joint_acceleration_regularization_score(asset: Articulation, acceleration_scale: float) -> torch.Tensor:
    scaled_acc = asset.data.joint_acc / max(acceleration_scale, 1.0e-6)
    return _exp_score_l2(torch.mean(torch.square(scaled_acc), dim=1), sigma=1.0)


def _torque_effort_regularization_score(asset: Articulation, torque_scale: float) -> torch.Tensor:
    scaled_torque = asset.data.applied_torque / max(torque_scale, 1.0e-6)
    return _exp_score_l2(torch.mean(torch.square(scaled_torque), dim=1), sigma=1.0)


def _safe_foot_contact_force_score(
    left_force: torch.Tensor,
    right_force: torch.Tensor,
    max_safe_foot_contact_force: float,
) -> torch.Tensor:
    left_excess = torch.clamp(left_force - max_safe_foot_contact_force, min=0.0)
    right_excess = torch.clamp(right_force - max_safe_foot_contact_force, min=0.0)
    force_excess_l2 = torch.square(left_excess / max(max_safe_foot_contact_force, 1.0e-6))
    force_excess_l2 = force_excess_l2 + torch.square(right_excess / max(max_safe_foot_contact_force, 1.0e-6))
    return _exp_score_l2(force_excess_l2, sigma=1.0)


def _phase_symmetric_knee_usage_score(
    asset: Articulation,
    want_left_swing: torch.Tensor,
    swing_target: float = 0.22,
    stance_target: float = 0.07,
    swing_sigma: float = 0.12,
    stance_sigma: float = 0.05,
) -> torch.Tensor:
    right_knee = _joint_abs_deviation(asset, ("176",))
    left_knee = _joint_abs_deviation(asset, ("196",))

    swing_knee = torch.where(want_left_swing, left_knee, right_knee)
    stance_knee = torch.where(want_left_swing, right_knee, left_knee)

    swing_target = max(swing_target, 1.0e-6)
    stance_target = max(stance_target, 1.0e-6)
    swing_gate = torch.clamp(swing_knee / (0.65 * swing_target), min=0.0, max=1.0)
    stance_gate = torch.clamp(stance_knee / (0.65 * stance_target), min=0.0, max=1.0)

    swing_score = _exp_score(swing_knee - swing_target, swing_sigma) * swing_gate
    stance_score = _exp_score(stance_knee - stance_target, stance_sigma) * stance_gate
    both_knee_activity = torch.clamp((left_knee + right_knee) / (swing_target + stance_target), min=0.0, max=1.0)
    return _clamp_score(0.60 * swing_score + 0.30 * stance_score + 0.10 * both_knee_activity)


def _contact_symmetric_knee_usage_score(
    asset: Articulation,
    actual_left_swing: torch.Tensor,
    actual_right_swing: torch.Tensor,
    swing_target: float = 0.22,
    stance_target: float = 0.07,
    swing_sigma: float = 0.12,
    stance_sigma: float = 0.05,
) -> torch.Tensor:
    right_knee = _joint_abs_deviation(asset, ("176",))
    left_knee = _joint_abs_deviation(asset, ("196",))
    swing_active = actual_left_swing | actual_right_swing

    swing_knee = torch.where(actual_left_swing, left_knee, right_knee)
    stance_knee = torch.where(actual_left_swing, right_knee, left_knee)

    swing_target = max(swing_target, 1.0e-6)
    stance_target = max(stance_target, 1.0e-6)
    swing_gate = torch.clamp(swing_knee / (0.65 * swing_target), min=0.0, max=1.0)
    stance_gate = torch.clamp(stance_knee / (0.65 * stance_target), min=0.0, max=1.0)

    swing_score = _exp_score(swing_knee - swing_target, swing_sigma) * swing_gate
    stance_score = _exp_score(stance_knee - stance_target, stance_sigma) * stance_gate
    both_knee_activity = torch.clamp((left_knee + right_knee) / (swing_target + stance_target), min=0.0, max=1.0)
    contact_score = _clamp_score(0.60 * swing_score + 0.30 * stance_score + 0.10 * both_knee_activity)
    return torch.where(swing_active, contact_score, torch.ones_like(contact_score))


def _startup_pose_guard_score(
    asset: Articulation,
    episode_time: torch.Tensor,
    contact_count: torch.Tensor,
    startup_guard_time: float = 0.75,
    joint_guard_range: float = 0.025,
    hip_yaw_guard_range: float = 0.035,
) -> torch.Tensor:
    joint_error = torch.abs(asset.data.joint_pos - asset.data.default_joint_pos)
    joint_excess = joint_error / max(joint_guard_range, 1.0e-6)
    joint_score = _exp_score_l2(torch.mean(torch.square(joint_excess), dim=1), sigma=1.0)
    hip_yaw_score = _hip_yaw_suppression_score(asset, hip_yaw_guard_range)
    both_feet_score = _clamp_score((contact_count >= 2.0).float())
    startup_score = _clamp_score(0.50 * joint_score + 0.30 * hip_yaw_score + 0.20 * both_feet_score)
    startup_mask = episode_time < startup_guard_time
    return torch.where(startup_mask, startup_score, torch.ones_like(startup_score))


def _named_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    name_tokens: tuple[str, ...],
    threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_names = list(getattr(contact_sensor, "body_names", []))
    body_ids = getattr(sensor_cfg, "body_ids", None)
    if body_ids is None:
        body_ids = list(range(len(body_names)))
    elif isinstance(body_ids, slice):
        start, stop, step = body_ids.indices(len(body_names))
        body_ids = list(range(start, stop, step))
    elif isinstance(body_ids, torch.Tensor):
        body_ids = [int(item) for item in body_ids.detach().cpu().tolist()]
    else:
        body_ids = [int(item) for item in body_ids]

    selected = [local for local, body_id in enumerate(body_ids) if any(token in body_names[body_id] for token in name_tokens)]
    if not selected:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    forces = contact_sensor.data.net_forces_w_history[:, :, body_ids]
    force_norm = torch.max(torch.linalg.norm(forces[:, :, selected], dim=-1), dim=1)[0]
    return torch.any(force_norm > threshold, dim=1)


def reward_humanoid_v7(
    env: ManagerBasedRLEnv,
    target_lin_vel_x: float = 0.25,
    target_lin_vel_y: float = 0.0,
    target_ang_vel_z: float = 0.0,
    forward_velocity_sigma: float = 0.35,
    yaw_velocity_sigma: float = 0.35,
    heading_alignment_sigma: float = 0.35,
    lateral_velocity_sigma: float = 0.08,
    lateral_position_sigma: float = 0.12,
    gait_freq: float = 1.2,
    swing_foot_forward_vel_target: float = 0.25,
    target_forward_step_length: float = 0.12,
    forward_step_sigma: float = 0.04,
    gait_balance_ema_alpha: float = 0.025,
    gait_balance_warmup_time: float = 1.2,
    gait_step_balance_sigma: float = 0.35,
    gait_activity_balance_sigma: float = 0.30,
    target_min_foot_lateral_gap: float = 0.08,
    target_max_foot_lateral_gap: float = 0.20,
    target_min_double_support_side_gap: float = 0.14,
    target_max_double_support_side_gap: float = 0.26,
    target_stance_lateral_gap: float = 0.20,
    foot_lateral_gap_sigma: float = 0.04,
    stance_width_sigma: float = 0.08,
    double_support_forward_gap_sigma: float = 0.045,
    target_double_support_forward_gap: float = 0.14,
    min_double_support_forward_gap: float = 0.08,
    max_double_support_forward_gap: float = 0.22,
    double_support_stride_gap_sigma: float = 0.035,
    lead_foot_gap_target: float = 0.10,
    lead_foot_gap_sigma: float = 0.04,
    max_foot_forward_from_center: float = 0.25,
    foot_forward_overreach_sigma: float = 0.05,
    double_support_grace_time: float = 0.20,
    double_support_penalty_ramp_time: float = 0.30,
    min_continuous_forward_speed: float = 0.05,
    stall_penalty_start_time: float = 1.0,
    feet_clearance_target: float = 0.04,
    feet_clearance_sigma: float = 0.03,
    pelvis_height_sigma: float = 0.06,
    upright_orientation_sigma: float = 0.35,
    feet_slide_sigma: float = 0.15,
    pressure_balance_sigma: float = 0.35,
    action_rate_sigma: float = 1.2,
    joint_velocity_scale: float = 4.0,
    joint_acceleration_scale: float = 60.0,
    torque_effort_scale: float = 12.0,
    max_safe_foot_contact_force: float = 4.0 * ROBOT_WEIGHT_N,
    hip_roll_suppression_range: float = 0.035,
    ankle_roll_suppression_range: float = 0.035,
    hip_yaw_suppression_range: float = 0.055,
    knee_usage_swing_target: float = 0.22,
    knee_usage_stance_target: float = 0.07,
    knee_usage_swing_sigma: float = 0.12,
    knee_usage_stance_sigma: float = 0.05,
    startup_guard_time: float = 0.75,
    startup_joint_guard_range: float = 0.025,
    startup_hip_yaw_guard_range: float = 0.035,
    contact_threshold: float = 8.0,
    max_tilt_xy: float = math.sin(math.radians(40.0)) ** 2,
    grace_time: float = 0.25,
    survival_reward_ramp_time: float = 2.0,
    feet_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces",
        body_names=["left_foot_1", "right_foot_1"],
        preserve_order=True,
    ),
    all_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Humanoid v7-2 single-stage lower-body walking reward.

    Every term is normalized to 0.001..1.0, and the final per-step reward is
    clamped to the same range. Bad pose or fall-like contact returns 0.001.
    """

    asset: Articulation = env.scene["robot"]
    episode_time = _episode_time(env)
    left_force, right_force = _foot_contact_forces(env, feet_sensor_cfg)
    left_contact = left_force > contact_threshold
    right_contact = right_force > contact_threshold
    feet_contact = torch.stack((left_contact.float(), right_contact.float()), dim=1)

    desired_forward_w = _initial_forward_w(env, asset)
    heading_lock_score = _heading_lock_score(env, asset, desired_forward_w, heading_alignment_sigma)
    root_forward_vel = _heading_locked_forward_speed(asset, desired_forward_w)
    forward_motion_gate = _clamp_score(
        torch.clamp((root_forward_vel + 0.02) / max(target_lin_vel_x, 1.0e-6), min=0.0, max=1.0)
    )
    forward_velocity_score = _exp_score(root_forward_vel - target_lin_vel_x, forward_velocity_sigma)
    forward_velocity_score = _clamp_score(forward_velocity_score * forward_motion_gate * heading_lock_score)
    yaw_stability_score = _exp_score(asset.data.root_ang_vel_b[:, 2] - target_ang_vel_z, yaw_velocity_sigma)
    lateral_speed = _heading_locked_lateral_speed(asset, desired_forward_w)
    lateral_position = _heading_locked_lateral_position(env, asset, desired_forward_w)
    lateral_velocity_score = _exp_score(lateral_speed, lateral_velocity_sigma)
    lateral_position_score = _exp_score(lateral_position, lateral_position_sigma)
    no_lateral_drift_score = _clamp_score(0.60 * lateral_velocity_score + 0.40 * lateral_position_score)

    left_swing_pattern = right_contact.float() * (1.0 - left_contact.float())
    right_swing_pattern = left_contact.float() * (1.0 - right_contact.float())
    actual_left_swing = left_swing_pattern > right_swing_pattern
    actual_right_swing = right_swing_pattern > left_swing_pattern
    actual_single_swing = actual_left_swing | actual_right_swing
    actual_swing_side = torch.where(
        actual_left_swing,
        torch.ones_like(root_forward_vel),
        torch.where(actual_right_swing, -torch.ones_like(root_forward_vel), torch.zeros_like(root_forward_vel)),
    )
    reset_mask = env.episode_length_buf <= 1
    previous_single_swing = getattr(env, "_pleas_previous_actual_single_swing", None)
    if (
        previous_single_swing is None
        or previous_single_swing.shape != actual_single_swing.shape
        or previous_single_swing.device != actual_single_swing.device
    ):
        previous_single_swing = torch.zeros_like(actual_single_swing)
    previous_swing_side = getattr(env, "_pleas_previous_swing_side", None)
    if (
        previous_swing_side is None
        or previous_swing_side.shape != actual_swing_side.shape
        or previous_swing_side.device != actual_swing_side.device
    ):
        previous_swing_side = torch.zeros_like(actual_swing_side)
    previous_swing_side_valid = getattr(env, "_pleas_previous_swing_side_valid", None)
    if (
        previous_swing_side_valid is None
        or previous_swing_side_valid.shape != actual_single_swing.shape
        or previous_swing_side_valid.device != actual_single_swing.device
    ):
        previous_swing_side_valid = torch.zeros_like(actual_single_swing)

    swing_event = actual_single_swing & (~previous_single_swing)
    alternated_swing = (previous_swing_side * actual_swing_side) < 0.0
    raw_swing_alternation_score = torch.where(
        swing_event & previous_swing_side_valid,
        alternated_swing.float(),
        torch.where(actual_single_swing, torch.ones_like(root_forward_vel), torch.zeros_like(root_forward_vel)),
    )
    same_foot_repeat_penalty = torch.where(
        swing_event & previous_swing_side_valid & (~alternated_swing),
        torch.ones_like(root_forward_vel),
        torch.zeros_like(root_forward_vel),
    )
    swing_alternation_warmup = torch.clamp(
        episode_time / max(gait_balance_warmup_time, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    swing_alternation_score = _clamp_score(
        (1.0 - swing_alternation_warmup) + swing_alternation_warmup * raw_swing_alternation_score
    )
    same_foot_repeat_penalty = swing_alternation_warmup * same_foot_repeat_penalty

    next_previous_swing_side = previous_swing_side.detach().clone()
    next_previous_swing_side_valid = previous_swing_side_valid.detach().clone()
    if torch.any(swing_event):
        next_previous_swing_side[swing_event] = actual_swing_side[swing_event].detach()
        next_previous_swing_side_valid[swing_event] = True
    if torch.any(reset_mask):
        next_previous_swing_side[reset_mask] = 0.0
        next_previous_swing_side_valid[reset_mask] = False
    env._pleas_previous_swing_side = next_previous_swing_side
    env._pleas_previous_swing_side_valid = next_previous_swing_side_valid
    env._pleas_previous_actual_single_swing = actual_single_swing.detach().clone()

    actual_single_support_score = _clamp_score(left_swing_pattern + right_swing_pattern)
    alternating_gait_score = _clamp_score(0.70 * actual_single_support_score + 0.30 * swing_alternation_score)

    left_foot_id = _body_id(asset, "left_foot_1")
    right_foot_id = _body_id(asset, "right_foot_1")
    left_forward_vel = _heading_locked_body_forward_velocity(asset, left_foot_id, desired_forward_w)
    right_forward_vel = _heading_locked_body_forward_velocity(asset, right_foot_id, desired_forward_w)
    swing_forward_vel = torch.where(
        actual_left_swing,
        left_forward_vel,
        torch.where(actual_right_swing, right_forward_vel, torch.zeros_like(left_forward_vel)),
    )
    swing_foot_forward_score = _clamp_score(
        torch.clamp((swing_forward_vel + 0.02) / max(swing_foot_forward_vel_target, 1.0e-6), min=0.0, max=1.0)
    )

    left_rel_b, right_rel_b = _root_frame_foot_center_positions(env, asset)
    foot_y_gap = torch.abs(left_rel_b[:, FOOT_GAP_AXIS] - right_rel_b[:, FOOT_GAP_AXIS])
    foot_y_gap_deficit = torch.clamp(target_min_foot_lateral_gap - foot_y_gap, min=0.0)
    foot_y_gap_excess = torch.clamp(foot_y_gap - target_max_foot_lateral_gap, min=0.0)
    foot_y_gap_violation = foot_y_gap_deficit + foot_y_gap_excess
    # The Y-gap target should shape only the double-support landing stance.
    # During swing it stays neutral so the swing foot can move forward freely.
    double_support = left_contact & right_contact
    no_scissor_score = torch.ones_like(foot_y_gap_violation)
    stance_width_raw_score = _exp_score(foot_y_gap_violation, double_support_forward_gap_sigma)
    stance_width_score = torch.where(
        double_support,
        stance_width_raw_score,
        torch.full_like(stance_width_raw_score, MIN_REWARD_SCORE),
    )
    foot_side_gap = torch.abs(left_rel_b[:, LATERAL_AXIS] - right_rel_b[:, LATERAL_AXIS])
    foot_side_gap_deficit = torch.clamp(target_min_double_support_side_gap - foot_side_gap, min=0.0)
    foot_side_gap_excess = torch.clamp(foot_side_gap - target_max_double_support_side_gap, min=0.0)
    foot_side_gap_violation = foot_side_gap_deficit + foot_side_gap_excess
    double_support_side_gap_score = torch.where(
        double_support,
        _exp_score(foot_side_gap_violation, stance_width_sigma),
        torch.full_like(foot_side_gap_violation, MIN_REWARD_SCORE),
    )
    # FK-equivalent lead-foot order term:
    # in real ROS2 this is reproduced from servo angles + link lengths.
    left_forward_fk = _root_frame_forward_position(left_rel_b)
    right_forward_fk = _root_frame_forward_position(right_rel_b)
    double_support_forward_gap = torch.abs(left_forward_fk - right_forward_fk)
    double_support_forward_gap_score = torch.where(
        double_support,
        _exp_score(double_support_forward_gap - target_double_support_forward_gap, double_support_stride_gap_sigma),
        torch.full_like(double_support_forward_gap, MIN_REWARD_SCORE),
    )
    same_forward_line_penalty = torch.where(
        double_support,
        torch.clamp(
            (min_double_support_forward_gap - double_support_forward_gap)
            / max(min_double_support_forward_gap, 1.0e-6),
            min=0.0,
            max=1.0,
        ),
        torch.zeros_like(double_support_forward_gap),
    )
    excessive_double_support_gap_penalty = torch.where(
        double_support,
        torch.clamp(
            (double_support_forward_gap - max_double_support_forward_gap)
            / max(max_double_support_forward_gap - target_double_support_forward_gap, 1.0e-6),
            min=0.0,
            max=1.0,
        ),
        torch.zeros_like(double_support_forward_gap),
    )
    lead_foot_order_score = torch.where(
        double_support,
        _exp_score(double_support_forward_gap - lead_foot_gap_target, lead_foot_gap_sigma),
        torch.full_like(double_support_forward_gap, MIN_REWARD_SCORE),
    )
    same_line_penalty = torch.clamp(
        ((0.65 * lead_foot_gap_target) - torch.abs(left_forward_fk - right_forward_fk))
        / max(0.65 * lead_foot_gap_target, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    lead_foot_order_penalty = torch.where(
        double_support,
        same_line_penalty,
        torch.zeros_like(double_support_forward_gap),
    )
    max_foot_forward = torch.maximum(torch.abs(left_forward_fk), torch.abs(right_forward_fk))
    foot_forward_overreach = torch.clamp(max_foot_forward - max_foot_forward_from_center, min=0.0)
    foot_forward_overreach_penalty = 1.0 - _exp_score(foot_forward_overreach, foot_forward_overreach_sigma)
    cross_forward_step_score = _clamp_score(
        alternating_gait_score * swing_foot_forward_score * no_scissor_score * heading_lock_score * forward_motion_gate
    )
    left_forward_pos = _heading_locked_foot_center_forward_position(asset, left_foot_id, desired_forward_w)
    right_forward_pos = _heading_locked_foot_center_forward_position(asset, right_foot_id, desired_forward_w)
    actual_swing_ahead = torch.where(
        actual_left_swing,
        left_forward_pos - right_forward_pos,
        torch.where(actual_right_swing, right_forward_pos - left_forward_pos, torch.zeros_like(left_forward_pos)),
    )
    swing_ahead = actual_swing_ahead
    swing_ahead_gate = _clamp_score(
        torch.clamp(swing_ahead / max(target_forward_step_length, 1.0e-6), min=0.0, max=1.0)
    )
    alternating_forward_step_score = _exp_score(
        torch.clamp(swing_ahead, min=0.0) - target_forward_step_length,
        forward_step_sigma,
    )
    alternating_forward_step_score = _clamp_score(
        actual_single_swing.float()
        * alternating_forward_step_score
        * swing_ahead_gate
        * no_scissor_score
        * heading_lock_score
        * forward_motion_gate
    )
    actual_swing_ahead_gate = _clamp_score(
        torch.clamp(actual_swing_ahead / max(target_forward_step_length, 1.0e-6), min=0.0, max=1.0)
    )
    symmetric_swing_forward_score = _exp_score(
        torch.clamp(actual_swing_ahead, min=0.0) - target_forward_step_length,
        forward_step_sigma,
    )
    symmetric_swing_forward_score = _clamp_score(
        actual_single_swing.float()
        * symmetric_swing_forward_score
        * actual_swing_ahead_gate
        * no_scissor_score
        * heading_lock_score
        * forward_motion_gate
    )
    swing_backward_penalty = torch.where(
        actual_single_swing,
        torch.clamp(-actual_swing_ahead / max(target_forward_step_length, 1.0e-6), min=0.0, max=1.0),
        torch.zeros_like(actual_swing_ahead),
    )
    left_step_signal = left_swing_pattern * torch.clamp(
        (left_forward_pos - right_forward_pos) / max(target_forward_step_length, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    right_step_signal = right_swing_pattern * torch.clamp(
        (right_forward_pos - left_forward_pos) / max(target_forward_step_length, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    reset_mask = env.episode_length_buf <= 1
    alpha = max(min(gait_balance_ema_alpha, 1.0), 0.0)

    def _update_gait_ema(name: str, value: torch.Tensor) -> torch.Tensor:
        stored = getattr(env, name, None)
        if stored is None or stored.shape != value.shape or stored.device != value.device:
            stored = value.detach().clone()
        stored = (1.0 - alpha) * stored + alpha * value.detach()
        if torch.any(reset_mask):
            stored[reset_mask] = value[reset_mask].detach()
        setattr(env, name, stored)
        return stored

    left_step_ema = _update_gait_ema("_pleas_left_step_signal_ema", left_step_signal)
    right_step_ema = _update_gait_ema("_pleas_right_step_signal_ema", right_step_signal)
    left_swing_ema = _update_gait_ema("_pleas_left_swing_activity_ema", left_swing_pattern)
    right_swing_ema = _update_gait_ema("_pleas_right_swing_activity_ema", right_swing_pattern)
    step_balance_error = torch.abs(left_step_ema - right_step_ema) / torch.clamp(
        left_step_ema + right_step_ema,
        min=1.0e-4,
    )
    activity_balance_error = torch.abs(left_swing_ema - right_swing_ema) / torch.clamp(
        left_swing_ema + right_swing_ema,
        min=1.0e-4,
    )
    step_balance_score = _exp_score(step_balance_error, gait_step_balance_sigma)
    activity_balance_score = _exp_score(activity_balance_error, gait_activity_balance_sigma)
    balance_warmup = torch.clamp(episode_time / max(gait_balance_warmup_time, 1.0e-6), min=0.0, max=1.0)
    raw_balanced_limp_score = _clamp_score(0.70 * step_balance_score + 0.30 * activity_balance_score)
    balanced_limp_score = _clamp_score((1.0 - balance_warmup) + balance_warmup * raw_balanced_limp_score)
    limp_imbalance_penalty = balance_warmup * torch.clamp(
        0.70 * step_balance_error + 0.30 * activity_balance_error,
        min=0.0,
        max=1.0,
    )

    contact_count = left_contact.float() + right_contact.float()
    single_support_score = _clamp_score((contact_count == 1.0).float())
    swing_air_score = actual_single_swing.float()
    step_continuation_score = _clamp_score(
        swing_air_score * swing_foot_forward_score * forward_motion_gate * heading_lock_score
    )
    double_support_hold_penalty = torch.where(
        double_support,
        torch.clamp(
            (episode_time - double_support_grace_time) / max(double_support_penalty_ramp_time, 1.0e-6),
            min=0.0,
            max=1.0,
        ),
        torch.zeros_like(episode_time),
    )
    stall_raw = torch.clamp(
        (min_continuous_forward_speed - root_forward_vel) / max(min_continuous_forward_speed, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    stall_penalty = torch.where(
        episode_time > stall_penalty_start_time,
        stall_raw,
        torch.zeros_like(stall_raw),
    )

    left_z = asset.data.body_pos_w[:, left_foot_id, 2] - env.scene.env_origins[:, 2]
    right_z = asset.data.body_pos_w[:, right_foot_id, 2] - env.scene.env_origins[:, 2]
    support_z = torch.where(actual_left_swing, right_z, torch.where(actual_right_swing, left_z, torch.minimum(left_z, right_z)))
    swing_z = torch.where(actual_left_swing, left_z, torch.where(actual_right_swing, right_z, support_z))
    swing_clearance = torch.clamp(swing_z - support_z, min=0.0)
    feet_clearance_score = _exp_score(swing_clearance - feet_clearance_target, feet_clearance_sigma)
    feet_clearance_score = _clamp_score(feet_clearance_score * actual_single_swing.float())

    tilt_l2 = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    upright_orientation_score = _exp_score_l2(tilt_l2, upright_orientation_sigma)
    pelvis_height = _pelvis_reference_height(env, asset)
    pelvis_height_score = _exp_score(pelvis_height - DEFAULT_PELVIS_HEIGHT_M, pelvis_height_sigma)
    pelvis_min_height_score = _clamp_score(
        torch.clamp((pelvis_height - MIN_PELVIS_HEIGHT_M) / 0.08, min=0.0, max=1.0)
    )
    pelvis_posture_score = _clamp_score(0.65 * pelvis_height_score + 0.35 * pelvis_min_height_score)

    left_vel_xy = asset.data.body_lin_vel_w[:, left_foot_id, :2]
    right_vel_xy = asset.data.body_lin_vel_w[:, right_foot_id, :2]
    feet_slide = torch.sum(torch.square(left_vel_xy), dim=1) * left_contact.float()
    feet_slide = feet_slide + torch.sum(torch.square(right_vel_xy), dim=1) * right_contact.float()
    no_feet_slide_score = _exp_score_l2(feet_slide, feet_slide_sigma)
    pressure_flat_contact_score = _pressure_flat_contact_score(
        env,
        asset,
        left_force,
        right_force,
        contact_threshold=contact_threshold,
        sigma=pressure_balance_sigma,
    )

    action_rate_l2 = torch.mean(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    smooth_action_score = _exp_score_l2(action_rate_l2, action_rate_sigma)
    joint_velocity_score = _joint_velocity_regularization_score(asset, joint_velocity_scale)
    joint_acceleration_score = _joint_acceleration_regularization_score(asset, joint_acceleration_scale)
    torque_effort_score = _torque_effort_regularization_score(asset, torque_effort_scale)
    contact_force_safe_score = _safe_foot_contact_force_score(
        left_force,
        right_force,
        max_safe_foot_contact_force,
    )
    motor_safe_joint_usage_score = _motor_safe_joint_usage_score(asset)
    knee_usage_score = _contact_symmetric_knee_usage_score(
        asset,
        actual_left_swing,
        actual_right_swing,
        swing_target=knee_usage_swing_target,
        stance_target=knee_usage_stance_target,
        swing_sigma=knee_usage_swing_sigma,
        stance_sigma=knee_usage_stance_sigma,
    )
    roll_suppression_score = _roll_suppression_score(
        asset,
        hip_roll_safe_range=hip_roll_suppression_range,
        ankle_roll_safe_range=ankle_roll_suppression_range,
    )
    hip_yaw_suppression_score = _hip_yaw_suppression_score(asset, hip_yaw_suppression_range)
    startup_pose_guard_score = _startup_pose_guard_score(
        asset,
        episode_time,
        contact_count,
        startup_guard_time=startup_guard_time,
        joint_guard_range=startup_joint_guard_range,
        hip_yaw_guard_range=startup_hip_yaw_guard_range,
    )

    total_reward = (
        0.09 * forward_velocity_score
        + 0.075 * heading_lock_score
        + 0.07 * no_lateral_drift_score
        + 0.04 * yaw_stability_score
        + 0.085 * alternating_gait_score
        + 0.075 * cross_forward_step_score
        + 0.085 * alternating_forward_step_score
        + 0.060 * symmetric_swing_forward_score
        + 0.080 * balanced_limp_score
        + 0.055 * swing_alternation_score
        + 0.070 * single_support_score
        + 0.060 * step_continuation_score
        + 0.04 * feet_clearance_score
        + 0.065 * upright_orientation_score
        + 0.025 * no_feet_slide_score
        + 0.030 * pressure_flat_contact_score
        + 0.020 * smooth_action_score
        + 0.012 * joint_velocity_score
        + 0.012 * joint_acceleration_score
        + 0.016 * torque_effort_score
        + 0.020 * contact_force_safe_score
        + 0.010 * motor_safe_joint_usage_score
        + 0.045 * knee_usage_score
        + 0.055 * roll_suppression_score
        + 0.040 * hip_yaw_suppression_score
        + 0.065 * startup_pose_guard_score
        + 0.030 * pelvis_posture_score
        + 0.035 * stance_width_score
        + 0.030 * double_support_side_gap_score
        + 0.070 * double_support_forward_gap_score
        + 0.040 * lead_foot_order_score
        - 0.035 * lead_foot_order_penalty
        - 0.055 * swing_backward_penalty
        - 0.080 * limp_imbalance_penalty
        - 0.075 * same_foot_repeat_penalty
        - 0.095 * same_forward_line_penalty
        - 0.040 * excessive_double_support_gap_penalty
        - 0.040 * foot_forward_overreach_penalty
        - 0.070 * double_support_hold_penalty
        - 0.050 * stall_penalty
    )

    # Prevent the policy from collecting high early rewards and dying around the grace boundary.
    survival_gate = torch.clamp(episode_time / max(survival_reward_ramp_time, 1.0e-6), min=0.12, max=1.0)
    total_reward = total_reward * survival_gate
    total_reward = _clamp_score(total_reward)

    too_tilted = tilt_l2 > max_tilt_xy
    fall_or_bad_pose = too_tilted
    return torch.where(fall_or_bad_pose, torch.full_like(total_reward, MIN_REWARD_SCORE), total_reward)


def fall_or_bad_pose_humanoid_v7_2(
    env: ManagerBasedRLEnv,
    max_tilt_xy: float = math.sin(math.radians(40.0)) ** 2,
    all_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    tilt_l2 = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    too_tilted = tilt_l2 > max_tilt_xy
    return too_tilted


reward_stage1_one_foot_air = reward_humanoid_v7
