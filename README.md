# Humanoid v7.2.23 Success Package

This repository contains the curated files for the successful humanoid v7.2.23 training setup.

## Contents

- `humanoid_v7/v7-2/training_code/rl v7.2.23 contact_driven_gait_from_v7222best/`
  - PPO configuration and training code for the successful v7.2.23 contact-driven gait run.
- `humanoid_v7/v7-2/RUN_v7_2_23_current_best_gui_1env_play.cmd`
  - GUI playback command for the current best v7.2.23 result.
- `humanoid_v7/v7-2/RUN_v7_2_23_contact_driven_gait_from_v7222best_headless_8192.cmd`
  - Headless training command for the successful contact-driven gait run.
- `humanoid_v7/v7-2/robot_asset/mass_6993g_robot/`
  - URDF, Xacro, mesh, and USD robot model files required by the setup.
- `humanoid_v7/v7-2/v7_2_training_config_summary.md`
  - Training configuration summary.
- `humanoid_v7/v7-2/reports/humanoid_reward_change_report.md`
  - Reward-change report for the humanoid project.

## Notes

Large experiment outputs, TensorBoard logs, checkpoints, and cache files are intentionally excluded. This keeps the repository focused on the source files needed to understand and reproduce the successful v7.2.23 setup.
