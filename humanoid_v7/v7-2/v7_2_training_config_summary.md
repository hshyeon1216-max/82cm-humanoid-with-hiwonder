# Humanoid v7-2 Training Config

## Run

```python
num_envs = 8192
num_steps_per_env = 32
max_iterations = 100000
save_interval = 100

num_learning_epochs = 5
num_mini_batches = 8

batch_size = num_envs * num_steps_per_env  # 262144
mini_batch_size = batch_size // num_mini_batches  # 32768
updates_per_iteration = num_learning_epochs * num_mini_batches  # 40
```

This keeps VRAM headroom for a separate 1-env GUI playback window.

## Policy Network

```text
observations -> Linear(obs_dim -> 512) -> ELU
             -> Linear(512 -> 256) -> ELU
             -> Linear(256 -> 128) -> ELU
             -> Linear(128 -> 64) -> ELU
             -> Linear(64 -> 12)
```

Current observation dimension is `85`.

Additional observations:
- `gait_phase`: 2 values, `sin(phase), cos(phase)`
- `foot_corner_pressures`: 8 values, left/right foot 4-corner pressure readings
- `foot_pressure_tilt_estimate`: 4 values, pressure-estimated left/right sole roll/pitch

## Reward

All reward terms are normalized to `0.001..1.0`.
The final per-step reward is clamped to `0.001..1.0`.
If `fall_or_bad_pose` is true, final reward is `0.001`.

| Term | Weight | Purpose |
| --- | ---: | --- |
| forward_velocity | 0.14 | Track forward velocity near 0.25 m/s |
| yaw_velocity | 0.04 | Keep yaw rate near command |
| alternating_gait | 0.15 | Encourage left/right alternating support |
| cross_forward_step | 0.13 | Swing foot moves forward without scissor/cross gait |
| single_support | 0.07 | Prefer one-foot support during swing |
| feet_clearance | 0.055 | Swing foot clearance near 4 cm |
| pelvis_height | 0.065 | Keep pelvis/root near 0.824 m |
| pelvis_min_height | 0.095 | Keep pelvis/root above 0.724 m |
| upright_orientation | 0.075 | Keep IMU/projected gravity upright |
| no_feet_slide | 0.04 | Prevent stance-foot sliding |
| pressure_flat_contact | 0.08 | Use 4-corner pressure distribution to keep the stance sole flat |
| smooth_action | 0.035 | Smooth servo commands |
| motor_safe_joint_usage | 0.025 | Keep weak roll/ankle joints inside safe range |

## Sim To Real

- USD: `C:\Users\hsh\OneDrive\바탕 화면\humanoid_v7\v7-2\robot_asset\mass_6993g_robot\usd_with_sensors\humanoid_v7_mass6993_sensors.usd`
- Robot mass target: `6.993 kg`
- Robot measured height: `0.824 m`
- Height reward target: `0.824 m`
- Bad-pose minimum height: `0.724 m`
- Bad-pose grace time: `1.0 s`
- Friction randomization: static `1.02..1.20`, dynamic `0.82..1.05`
- Actuator gain randomization: stiffness `0.90..1.10`, damping `0.85..1.15`
- Joint friction/armature randomization: friction `0.85..1.15`, armature `0.90..1.10`
- Servo delay: `1..3` control steps
- IMU noise: base angular velocity `std=0.010`, bias `0.002`
- Projected gravity noise: `std=0.010`
- Joint backlash/noise: position `std=0.006`, bias `0.003`
- Pressure/contact force noise: force noise is applied in foot-contact observation/reward helper
- Pressure sensor placement: 4 sensors per sole, total 8 sensors
- Pressure sensor capacity: `50 kg` / `490.3325 N` each
- Lower sole plate thickness: `0.010 m`
- Pressure sensor mounting height: `z=-0.002 m`, upper-plate underside above the lower plate
- Pressure sensor corners per foot:
  - front_left: `(-0.030, 0.055, -0.002)`
  - front_right: `(0.030, 0.055, -0.002)`
  - rear_left: `(-0.030, -0.045, -0.002)`
  - rear_right: `(0.030, -0.045, -0.002)`

## Motor Mapping

| Joint group | Joint tokens | Motor in the active IsaacLab actuator cfg |
| --- | --- | --- |
| hip_roll | right 163, left 185 | HTD45H x1 |
| hip_pitch | right 166, left 188 | HTD85H x2 |
| hip_yaw | right 172, left 192 | HTD45H x1 |
| knee_pitch | right 176, left 196 | HTD85H x2 |
| ankle_pitch | right 179, left 199 | HTD85H x1 |
| ankle_roll | right 182, left 202 | HTD45H x1 |
