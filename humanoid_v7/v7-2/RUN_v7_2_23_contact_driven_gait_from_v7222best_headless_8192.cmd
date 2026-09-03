@echo off
title Humanoid v7.2.23 contact-driven gait from v7.2.22 best
cd /d "C:\tmp\v7_2"
set "PYTHONIOENCODING=utf-8"
set "HEADLESS=1"
set "LIVESTREAM=0"
set "ENABLE_CAMERAS=0"
set "ISAACLAB=C:\IsaacLab"
echo [INFO] Humanoid v7.2.23 headless training
echo [INFO] Base checkpoint: v7.2.22 best
echo [INFO] Replaced phase-forced left/right swing reward with contact-driven gait reward
echo [INFO] Added actual swing-side alternation memory and same-foot repeat penalty
echo [INFO] Knee usage reward now follows the actual airborne foot
echo [INFO] Existing sim-to-real, actuator, friction, noise, network and PPO settings are unchanged
echo [INFO] Sim-to-real: 10-20 percent
echo [INFO] observations=117 network=117 - 512 - 256 - 128 - 64 - 12
echo [INFO] num_envs=8192 horizon_length=32 minibatch_size=32768 max_epochs=100000
echo [INFO] logs/checkpoints: C:\tmp\v7_2\logs\rl_games\humanoid_v7_2_23_contact_driven_gait_from_v7222best_sim10to20_obs117_pelvisBody719_net512_8192env_mb32768
echo.
call "%ISAACLAB%\isaaclab.bat" -p "%ISAACLAB%\scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Pleas-OneFootBalance-v0 --num_envs 8192 --device cuda:0 --headless --max_iterations 100000
echo.
echo [INFO] Training process finished.
pause
