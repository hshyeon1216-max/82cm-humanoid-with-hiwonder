import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.pleas_one_foot_balance.mdp as mdp


ROBOT_USD = r"C:\Users\hsh\OneDrive\바탕 화면\humanoid_git\humanoid_v7\v7-2\robot_asset\mass_6993g_robot\usd_with_sensors\humanoid_v7_mass6993_sensors.usd"
KGF_CM_TO_NM = 0.0980665
HTD_85H_TORQUE_NM = 85.0 * KGF_CM_TO_NM
HTD_45H_TORQUE_NM = 45.0 * KGF_CM_TO_NM
HTD_85H_DUAL_TORQUE_NM = 2.0 * HTD_85H_TORQUE_NM
SERVO_MIN_DELAY_STEPS = 1
SERVO_MAX_DELAY_STEPS = 3


@configclass
class PleasOneFootSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        actuators={
            "a185_htd45h_single": DelayedPDActuatorCfg(
                joint_names_expr=[".*185"],
                effort_limit=HTD_45H_TORQUE_NM,
                effort_limit_sim=HTD_45H_TORQUE_NM,
                velocity_limit=2.5,
                velocity_limit_sim=2.5,
                stiffness=9000.0,
                damping=900.0,
                armature=0.18,
                friction=4.0,
                dynamic_friction=1.8,
                viscous_friction=1.2,
                min_delay=SERVO_MIN_DELAY_STEPS,
                max_delay=SERVO_MAX_DELAY_STEPS,
            ),
            "a163_htd45h_single": DelayedPDActuatorCfg(
                joint_names_expr=[".*163"],
                effort_limit=HTD_45H_TORQUE_NM,
                effort_limit_sim=HTD_45H_TORQUE_NM,
                velocity_limit=2.5,
                velocity_limit_sim=2.5,
                stiffness=9000.0,
                damping=900.0,
                armature=0.18,
                friction=4.0,
                dynamic_friction=1.8,
                viscous_friction=1.2,
                min_delay=SERVO_MIN_DELAY_STEPS,
                max_delay=SERVO_MAX_DELAY_STEPS,
            ),
            "a192_htd45h_single": DelayedPDActuatorCfg(
                joint_names_expr=[".*192"],
                effort_limit=HTD_45H_TORQUE_NM,
                effort_limit_sim=HTD_45H_TORQUE_NM,
                velocity_limit=2.5,
                velocity_limit_sim=2.5,
                stiffness=9000.0,
                damping=900.0,
                armature=0.18,
                friction=4.0,
                dynamic_friction=1.8,
                viscous_friction=1.2,
                min_delay=SERVO_MIN_DELAY_STEPS,
                max_delay=SERVO_MAX_DELAY_STEPS,
            ),
            "a172_htd45h_single": DelayedPDActuatorCfg(
                joint_names_expr=[".*172"],
                effort_limit=HTD_45H_TORQUE_NM,
                effort_limit_sim=HTD_45H_TORQUE_NM,
                velocity_limit=2.5,
                velocity_limit_sim=2.5,
                stiffness=9000.0,
                damping=900.0,
                armature=0.18,
                friction=4.0,
                dynamic_friction=1.8,
                viscous_friction=1.2,
                min_delay=SERVO_MIN_DELAY_STEPS,
                max_delay=SERVO_MAX_DELAY_STEPS,
            ),
            "a202_htd45h_single": DelayedPDActuatorCfg(
                joint_names_expr=[".*202"],
                effort_limit=HTD_45H_TORQUE_NM,
                effort_limit_sim=HTD_45H_TORQUE_NM,
                velocity_limit=2.5,
                velocity_limit_sim=2.5,
                stiffness=9000.0,
                damping=900.0,
                armature=0.18,
                friction=4.0,
                dynamic_friction=1.8,
                viscous_friction=1.2,
                min_delay=SERVO_MIN_DELAY_STEPS,
                max_delay=SERVO_MAX_DELAY_STEPS,
            ),
            "a182_htd45h_single": DelayedPDActuatorCfg(
                joint_names_expr=[".*182"],
                effort_limit=HTD_45H_TORQUE_NM,
                effort_limit_sim=HTD_45H_TORQUE_NM,
                velocity_limit=2.5,
                velocity_limit_sim=2.5,
                stiffness=9000.0,
                damping=900.0,
                armature=0.18,
                friction=4.0,
                dynamic_friction=1.8,
                viscous_friction=1.2,
                min_delay=SERVO_MIN_DELAY_STEPS,
                max_delay=SERVO_MAX_DELAY_STEPS,
            ),
            "a188_htd85h_dual": DelayedPDActuatorCfg(
                joint_names_expr=[".*188"],
                effort_limit=HTD_85H_DUAL_TORQUE_NM,
                effort_limit_sim=HTD_85H_DUAL_TORQUE_NM,
                velocity_limit=2.5,
                velocity_limit_sim=2.5,
                stiffness=22000.0,
                damping=2200.0,
                armature=0.3,
                friction=6.0,
                dynamic_friction=2.6,
                viscous_friction=1.8,
                min_delay=SERVO_MIN_DELAY_STEPS,
                max_delay=SERVO_MAX_DELAY_STEPS,
            ),
            "a166_htd85h_dual": DelayedPDActuatorCfg(
                joint_names_expr=[".*166"],
                effort_limit=HTD_85H_DUAL_TORQUE_NM,
                effort_limit_sim=HTD_85H_DUAL_TORQUE_NM,
                velocity_limit=2.5,
                velocity_limit_sim=2.5,
                stiffness=22000.0,
                damping=2200.0,
                armature=0.3,
                friction=6.0,
                dynamic_friction=2.6,
                viscous_friction=1.8,
                min_delay=SERVO_MIN_DELAY_STEPS,
                max_delay=SERVO_MAX_DELAY_STEPS,
            ),
            "a196_htd85h_dual": DelayedPDActuatorCfg(
                joint_names_expr=[".*196"],
                effort_limit=HTD_85H_DUAL_TORQUE_NM,
                effort_limit_sim=HTD_85H_DUAL_TORQUE_NM,
                velocity_limit=2.5,
                velocity_limit_sim=2.5,
                stiffness=22000.0,
                damping=2200.0,
                armature=0.3,
                friction=6.0,
                dynamic_friction=2.6,
                viscous_friction=1.8,
                min_delay=SERVO_MIN_DELAY_STEPS,
                max_delay=SERVO_MAX_DELAY_STEPS,
            ),
            "a176_htd85h_dual": DelayedPDActuatorCfg(
                joint_names_expr=[".*176"],
                effort_limit=HTD_85H_DUAL_TORQUE_NM,
                effort_limit_sim=HTD_85H_DUAL_TORQUE_NM,
                velocity_limit=2.5,
                velocity_limit_sim=2.5,
                stiffness=22000.0,
                damping=2200.0,
                armature=0.3,
                friction=6.0,
                dynamic_friction=2.6,
                viscous_friction=1.8,
                min_delay=SERVO_MIN_DELAY_STEPS,
                max_delay=SERVO_MAX_DELAY_STEPS,
            ),
            "a199_htd85h_single": DelayedPDActuatorCfg(
                joint_names_expr=[".*199"],
                effort_limit=HTD_85H_TORQUE_NM,
                effort_limit_sim=HTD_85H_TORQUE_NM,
                velocity_limit=2.5,
                velocity_limit_sim=2.5,
                stiffness=14000.0,
                damping=1400.0,
                armature=0.22,
                friction=5.0,
                dynamic_friction=2.2,
                viscous_friction=1.5,
                min_delay=SERVO_MIN_DELAY_STEPS,
                max_delay=SERVO_MAX_DELAY_STEPS,
            ),
            "a179_htd85h_single": DelayedPDActuatorCfg(
                joint_names_expr=[".*179"],
                effort_limit=HTD_85H_TORQUE_NM,
                effort_limit_sim=HTD_85H_TORQUE_NM,
                velocity_limit=2.5,
                velocity_limit_sim=2.5,
                stiffness=14000.0,
                damping=1400.0,
                armature=0.22,
                friction=5.0,
                dynamic_friction=2.2,
                viscous_friction=1.5,
                min_delay=SERVO_MIN_DELAY_STEPS,
                max_delay=SERVO_MAX_DELAY_STEPS,
            ),
        },
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.DefaultCenteredJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=[
            ".*163",
            ".*166",
            ".*172",
            ".*176",
            ".*179",
            ".*182",
            ".*185",
            ".*188",
            ".*192",
            ".*196",
            ".*199",
            ".*202",
        ],
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.noisy_base_ang_vel, scale=0.25, params={"std": 0.010, "bias_std": 0.002})
        projected_gravity = ObsTerm(func=mdp.noisy_projected_gravity, params={"std": 0.010})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel_backlash_obs, params={"backlash_std": 0.006, "backlash_bias_std": 0.003})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        foot_heights = ObsTerm(
            func=mdp.foot_heights,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_1", "right_foot_1"], preserve_order=True)},
        )
        foot_positions_rel_root = ObsTerm(
            func=mdp.foot_positions_rel_root,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_1", "right_foot_1"], preserve_order=True)},
        )
        foot_contact_forces = ObsTerm(
            func=mdp.foot_contact_forces_obs,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_1", "right_foot_1"], preserve_order=True)},
        )
        foot_corner_pressures = ObsTerm(
            func=mdp.foot_corner_pressure_obs,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_1", "right_foot_1"], preserve_order=True)},
        )
        foot_pressure_tilt_estimate = ObsTerm(
            func=mdp.foot_pressure_tilt_estimate_obs,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_1", "right_foot_1"], preserve_order=True)},
        )
        joint_position_error = ObsTerm(func=mdp.joint_position_error_obs)
        joint_limit_margin = ObsTerm(func=mdp.joint_limit_margin_obs)
        pressure_cop_xy = ObsTerm(
            func=mdp.pressure_cop_xy_obs,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_1", "right_foot_1"], preserve_order=True)},
        )
        pressure_front_rear_balance = ObsTerm(
            func=mdp.pressure_front_rear_balance_obs,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_1", "right_foot_1"], preserve_order=True)},
        )
        pressure_left_right_balance = ObsTerm(
            func=mdp.pressure_left_right_balance_obs,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_1", "right_foot_1"], preserve_order=True)},
        )
        support_swing_contact = ObsTerm(
            func=mdp.support_swing_contact_obs,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_1", "right_foot_1"], preserve_order=True)},
        )
        foot_tilt = ObsTerm(func=mdp.foot_tilt_obs)
        base_velocity_command = ObsTerm(func=mdp.fixed_base_velocity_command)
        gait_phase = ObsTerm(func=mdp.gait_phase_obs, params={"gait_freq": 1.2})
        actions = ObsTerm(func=mdp.last_action)
        action_rate = ObsTerm(func=mdp.action_rate_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    robot_material_variation = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (1.02, 1.20),
            "dynamic_friction_range": (0.82, 1.05),
            "restitution_range": (0.0, 0.02),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    actuator_gain_variation = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.90, 1.10),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    joint_friction_variation = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_distribution_params": (0.85, 1.15),
            "armature_distribution_params": (0.90, 1.10),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "yaw": (-0.02, 0.02)},
            "velocity_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005), "roll": (-0.01, 0.01), "pitch": (-0.01, 0.01), "yaw": (-0.01, 0.01)},
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.01, 0.01), "velocity_range": (-0.01, 0.01)},
    )


@configclass
class RewardsCfg:
    reward_humanoid_v7 = RewTerm(
        func=mdp.reward_humanoid_v7,
        weight=1.0,
        params={
            "target_lin_vel_x": 0.25,
            "target_lin_vel_y": 0.0,
            "target_ang_vel_z": 0.0,
            "forward_velocity_sigma": 0.35,
            "yaw_velocity_sigma": 0.35,
            "heading_alignment_sigma": 0.35,
            "lateral_velocity_sigma": 0.08,
            "lateral_position_sigma": 0.12,
            "gait_freq": 1.2,
            "swing_foot_forward_vel_target": 0.25,
            "target_forward_step_length": 0.12,
            "gait_balance_ema_alpha": 0.025,
            "gait_balance_warmup_time": 1.2,
            "gait_step_balance_sigma": 0.35,
            "gait_activity_balance_sigma": 0.30,
            "target_min_foot_lateral_gap": 0.10,
            "target_max_foot_lateral_gap": 0.20,
            "target_min_double_support_side_gap": 0.14,
            "target_max_double_support_side_gap": 0.26,
            "target_stance_lateral_gap": 0.15,
            "foot_lateral_gap_sigma": 0.04,
            "stance_width_sigma": 0.08,
            "double_support_forward_gap_sigma": 0.045,
            "target_double_support_forward_gap": 0.14,
            "min_double_support_forward_gap": 0.08,
            "max_double_support_forward_gap": 0.22,
            "double_support_stride_gap_sigma": 0.035,
            "lead_foot_gap_target": 0.10,
            "lead_foot_gap_sigma": 0.04,
            "max_foot_forward_from_center": 0.25,
            "foot_forward_overreach_sigma": 0.05,
            "double_support_grace_time": 0.20,
            "double_support_penalty_ramp_time": 0.30,
            "min_continuous_forward_speed": 0.05,
            "stall_penalty_start_time": 1.0,
            "feet_clearance_target": 0.04,
            "feet_clearance_sigma": 0.03,
            "pelvis_height_sigma": 0.06,
            "upright_orientation_sigma": 0.35,
            "feet_slide_sigma": 0.15,
            "pressure_balance_sigma": 0.35,
            "action_rate_sigma": 1.2,
            "joint_velocity_scale": 4.0,
            "joint_acceleration_scale": 60.0,
            "torque_effort_scale": 12.0,
            "max_safe_foot_contact_force": 274.3,
            "hip_roll_suppression_range": 0.035,
            "ankle_roll_suppression_range": 0.035,
            "hip_yaw_suppression_range": 0.055,
            "startup_guard_time": 0.75,
            "startup_joint_guard_range": 0.025,
            "startup_hip_yaw_guard_range": 0.035,
            "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_1", "right_foot_1"], preserve_order=True),
            "all_contact_sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall_or_bad_pose = DoneTerm(
        func=mdp.fall_or_bad_pose_humanoid_v7_2,
        params={
            "max_tilt_xy": 0.41317591116653485,
            "all_contact_sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )


@configclass
class PleasOneFootBalanceEnvCfg(ManagerBasedRLEnvCfg):
    scene: PleasOneFootSceneCfg = PleasOneFootSceneCfg(num_envs=8192, env_spacing=3.0, clone_in_fabric=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = 1.2
        self.sim.physics_material.dynamic_friction = 0.9
        self.sim.physics_material.restitution = 0.0










