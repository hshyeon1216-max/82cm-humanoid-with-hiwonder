@echo off
title Humanoid v7.2.23 CURRENT BEST GUI play
cd /d C:\tmp\v7_2
set "PYTHONIOENCODING=utf-8"
set "HEADLESS=0"
set "LIVESTREAM=0"
set "ENABLE_CAMERAS=0"
set "PTH=C:\tmp\v7_2\logs\rl_games\humanoid_v7_2_23_contact_driven_gait_from_v7222best_sim10to20_obs117_pelvisBody719_net512_8192env_mb32768\2026-07-26_16-26-37\nn\humanoid_v7_2_23_contact_driven_gait_from_v7222best_sim10to20_obs117_pelvisBody719_net512_8192env_mb32768.pth"
echo [INFO] Humanoid v7.2.23 CURRENT BEST GUI play
echo [INFO] Checkpoint: %PTH%
echo [INFO] Headless training is not touched.
call "C:\IsaacLab\isaaclab.bat" -p "C:\IsaacLab\scripts\reinforcement_learning\rl_games\play.py" --task Isaac-Pleas-OneFootBalance-v0 --num_envs 1 --device cuda:0 --checkpoint "%PTH%" --real-time --experience "C:\IsaacLab\apps\isaaclab.python.rendering.kit" --rendering_mode performance --kit_args "--reset-user --/renderer/activeGpu=0 --/renderer/multiGpu/autoEnable=0 --/renderer/multiGpu/enabled=0 --/renderer/multiGpu/maxGpuCount=1 --/rtx-transient/dlssg/enabled=false --/rtx/verifyDriverVersion/enabled=false"
pause
