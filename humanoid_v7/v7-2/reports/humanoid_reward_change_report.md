# Humanoid V7 전용 강화학습 보고서

작성일: 2026-07-02  
대상 프로젝트: `humanoid_v7 / v7-2`  
보고서 범위: `humanoid_v7`에서 사용한 USD, 센서 입력, 보상체계, sim-to-real 설정, pth 실행 파일, ONNX export, 학습 run만 기록한다.  
정리 원칙: 이전 버전의 상세 내용은 이 보고서에서 제외하고, V7 관련 내용만 기록한다.

현재 기준 경로:

- 프로젝트 폴더: `C:\Users\hsh\OneDrive\바탕 화면\humanoid_v7\v7-2`
- 보고서 폴더: `C:\Users\hsh\OneDrive\바탕 화면\보고서용 정리\humanoid_v7\보고서`
- pth 실행 파일 폴더: `C:\Users\hsh\OneDrive\바탕 화면\보고서용 정리\humanoid_v7\pth_실행파일`
- 강화학습 코드 기록 폴더: `C:\Users\hsh\OneDrive\바탕 화면\보고서용 정리\humanoid_v7\rl_강화학습코드`
- IsaacLab task 코드: `C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\pleas_one_foot_balance`
- v7-2 작업 폴더: `C:\tmp\v7_2`

## 1. V7 보고서 목적

이 문서는 humanoid_v7에서 실제 하드웨어 적용을 목표로 진행한 보행 강화학습의 변경 이력을 기록한다. 핵심 목적은 ROS2에서 버튼을 눌렀을 때, 그 순간 로봇이 바라보는 방향을 기준으로 앞으로 걷는 정책을 얻는 것이다.

V7에서 중요한 점은 세 가지다.

- 로봇 설정은 사용자가 만든 URDF/USD 질량과 서보모터 구성을 최대한 유지한다.
- 자이로/IMU, 스마트서보 상태, 발바닥 압력센서 8개에서 실제로 얻을 수 있는 값 위주로 observation을 구성한다.
- 보상체계는 staged curriculum 대신 단일 보행 reward를 사용하되, 문제 행동이 발견될 때마다 lateral drift, startup twist, foot landing, backward step 같은 항목을 추가로 보정한다.

## 2. V7 고정 기준 세팅

### 2.1 로봇/자산 기준

| 항목 | 값 |
|---|---|
| 로봇 높이 | `0.824 m` |
| 골반 기준 높이 | `0.719 m` |
| 목표 총 질량 | 약 `6.993 kg` |
| USD | `C:\Users\hsh\OneDrive\바탕 화면\humanoid_v7\v7-2\robot_asset\mass_6993g_robot\usd_with_sensors\humanoid_v7_mass6993_sensors.usd` |
| 좌우 발 중심 간격 | 약 `0.20 m` |
| 로봇 local 좌우축 | `X` |
| 로봇 local 전후축 | `Y` |
| 발 앞쪽 정의 | `-Y` |

### 2.2 모터/액추에이터 기준

| 관절군 | 모터 설정 |
|---|---|
| hip roll | `HTD45H` 단일 |
| hip pitch | `HTD85H x2` |
| hip yaw | `HTD85H` 단일 |
| knee pitch | `HTD85H x2` |
| ankle pitch | `HTD45H` 단일 |
| ankle roll | `HTD45H` 단일 |

모터 설정은 `DelayedPDActuatorCfg`를 사용한다. V7 기준에서는 실제 서보와의 차이를 줄이기 위해 effort limit, stiffness, damping, friction, armature, servo delay를 함께 설정한다.

### 2.3 신경망 기준

| 항목 | 값 |
|---|---|
| observation dim | `117` |
| action dim | `12` |
| network | `117 -> 512 -> 256 -> 128 -> 64 -> 12` |
| activation | `ELU` |
| normalize_input | `True` |
| normalize_value | `True` |
| PPO backend | `rl_games` |

### 2.4 기본 학습 기준

| 항목 | 값 |
|---|---:|
| num_envs | `8192` |
| horizon_length | `32` |
| minibatch_size | `32768` |
| max_iterations | `100000` |
| device | `cuda:0` |
| GUI 확인 | 학습과 별도 1-env play 실행 |

### 2.5 GUI 실행 고정 규칙

GUI는 학습 headless 프로세스를 끄지 않고 별도 1마리 play로 실행한다. Isaac Sim GUI가 회색 화면 또는 GPU 충돌을 보이는 문제를 줄이기 위해 다음 우회 옵션을 고정한다.

```text
--experience C:\IsaacLab\apps\isaaclab.python.rendering.kit
--rendering_mode performance
--kit_args "--reset-user --/renderer/activeGpu 0 --/renderer/multiGpu/autoEnable 0 --/renderer/multiGpu/enabled 0 --/renderer/multiGpu/maxGpuCount 1 --/rtx-transient/dlssg/enabled false --/rtx/verifyDriverVersion/enabled false"
```

### 2.6 V7에서 따로 보존해야 하는 파일

| 구분 | 파일 |
|---|---|
| pth v7.2.5 GUI 실행 | `C:\Users\hsh\OneDrive\바탕 화면\보고서용 정리\humanoid_v7\pth_실행파일\pth v7.2.5 right_foot_not_forward_best_GUI.cmd` |
| pth v7.2.5 ONNX | `C:\Users\hsh\OneDrive\바탕 화면\humanoid_v7\v7-2\onnx\humanoid_v7_2_5_right_foot_not_forward_best_obs117_act12.onnx` |
| sim30to50 재학습 CMD | `C:\Users\hsh\OneDrive\바탕 화면\humanoid_v7\v7-2\RUN_v7_2_lateral_startup_guard_sim30to50_from_v725_headless_8192.cmd` |
| 현재 보고서 | `C:\Users\hsh\OneDrive\바탕 화면\보고서용 정리\humanoid_v7\보고서\humanoid_reward_change_report.md` |

## 3. humanoid_v7 보상체계 변경

### 3.1 전환 배경

사용자가 ROBOTIS K1 Rev.1 계열 오픈소스 보행 reward를 찾아왔고, 이를 기반으로 staged curriculum이 아니라 단일 locomotion reward를 사용하기로 했다.

초기 참고 reward의 핵심 항목:
- alive
- termination penalty
- linear velocity tracking
- yaw velocity tracking
- base height
- vertical velocity penalty
- angular velocity penalty
- flat orientation penalty
- feet air time
- feet single contact
- feet clearance
- feet slide
- feet orientation
- touchdown velocity/acceleration
- action rate
- joint acceleration
- torque penalty
- undesired contact
- joint limit penalty
- joint deviation penalty

### 3.2 humanoid_v7에 맞게 변형한 방향

ROBOTIS K1 reward를 그대로 복사하지 않고, 사용자 로봇의 하드웨어 특성에 맞게 수정했다.

반영한 하드웨어 조건:
- 로봇 높이 약 82.4cm
- 질량 6.993kg
- 하체 12DOF
- 자이로센서
- 양발 네 모서리 압력센서 총 8개
- HTD85H / HTD45H 서보모터 토크 매핑
- 라즈베리파이 5에서 추론 가능한 신경망 크기

### 3.3 V7 초기 단일 보행 reward

초기 설계안:

| 항목 | weight | 목적 |
|---|---:|---|
| forward_velocity | 0.16 | 목표 전진 속도 추종 |
| yaw_velocity | 0.04 | 목표 yaw 속도 추종 |
| alternating_gait | 0.16 | 좌우 교대 보행 |
| cross_forward_step | 0.14 | 스윙발이 앞으로 이동 |
| single_support | 0.07 | 한발 지지 패턴 |
| feet_clearance | 0.06 | 스윙발 들기 |
| pelvis_height | 0.07 | 골반 기준 높이 유지 |
| pelvis_min_height | 0.10 | 골반이 너무 낮아지지 않게 함 |
| upright_orientation | 0.08 | 몸통 기울기 억제 |
| no_feet_slide | 0.05 | 접촉발 미끄러짐 억제 |
| smooth_action | 0.04 | 부드러운 action |
| motor_safe_joint_usage | 0.03 | 모터별 안전 범위 사용 |

총합은 1.0이 되도록 구성했다. 각 term은 0.001~1.0으로 normalize하고, 최종 per-step reward도 0.001~1.0으로 clamp하도록 설계했다.

### 3.4 V7에서 추가된 압력센서 기반 항목

V7-2에서 압력센서를 더 적극적으로 사용하기 위해 다음 관측값과 보상 개념이 추가되었다.

관측값:
- `foot_corner_pressures`: 양발 네 모서리 압력, 총 8개
- `foot_pressure_tilt_estimate`: 압력 차이로 추정한 발바닥 roll/pitch
- `pressure_cop_xy`: 각 발의 CoP x/y
- `pressure_front_rear_balance`: 앞뒤 압력 균형
- `pressure_left_right_balance`: 좌우 압력 균형

보상:
- `pressure_flat_contact_score`: 접촉 중인 발의 네 모서리 압력이 균형 있게 분포하면 보상

목적:
- 발바닥 한쪽 날로 서는 자세 방지
- 실제 압력센서 기반으로 발바닥 각도를 어림잡는 구조 반영
- 지지발이 바닥에 더 평평하게 닿도록 유도

### 3.5 V7에서 추가된 117개 입력 구조

초기 71입력에서 하드웨어가 계산 가능한 범위 안에서 입력을 확장했다.

추가/유지된 주요 입력:
- base angular velocity
- projected gravity
- joint position
- joint velocity
- foot heights
- foot positions relative root
- foot contact forces
- foot corner pressures
- pressure tilt estimate
- joint position error
- joint limit margin
- pressure CoP x/y
- pressure front/rear balance
- pressure left/right balance
- support/swing contact state
- foot tilt
- base velocity command
- gait phase
- previous action
- action rate

배터리 전압은 실제 측정 불가라 입력에서 제외했다.

### 3.6 V7에서 추가된 sim-to-real 요소

- IMU angular velocity noise: std 0.010
- IMU bias noise: std 0.002
- projected gravity noise: std 0.010
- joint backlash std: 0.006
- joint backlash bias std: 0.003
- servo delay: 1~3 step
- ground static friction randomization: 1.02~1.20
- ground dynamic friction randomization: 0.82~1.05
- joint stiffness variation: 0.90~1.10
- joint damping variation: 0.85~1.15
- joint friction variation: 0.85~1.15
- armature variation: 0.90~1.10

### 3.7 V7에서 추가된 roll 억제

로봇이 약한 관절인 발목 roll과 골반 roll로 버티는 문제가 반복되어 다음 보상이 추가되었다.

- `roll_suppression_score`
- hip roll safe range: 0.035 rad
- ankle roll safe range: 0.035 rad

목적:
- 발목 roll로 몸을 받치는 행동 억제
- 골반 roll로 과하게 넘어가는 행동 억제
- hip pitch/knee pitch 중심의 보행 유도

### 3.8 V7에서 추가된 motor safe joint usage

모터별 힘 차이를 반영하기 위해 관절 그룹별 safe range를 두었다.

초기 개념:
- hip_roll: 작게 사용
- hip_pitch: 크게 사용 가능
- hip_yaw: 중간
- knee: 크게 사용 가능
- ankle_pitch: 제한적 사용
- ankle_roll: 작게 사용

목적:
- HTD85H 듀얼을 쓰는 hip pitch/knee pitch 쪽 사용 유도
- HTD45H를 쓰는 roll/ankle 계열 과사용 억제

## 4. V7-2에서 삭제된 항목

### 4.1 삭제: pelvis_height_score

기존 목적:
- 골반 높이를 기준 높이 근처에 유지

삭제 이유:
- 실제 로봇의 몸통/배터리 배치 때문에 골반 높이 기준이 애매했다.
- 골반 파츠 중앙은 약 0.719m이고, 배터리 맨 위까지는 0.824m라 높이 기준이 혼동되었다.
- reward가 이상하게 나오는 원인 후보였다.

삭제 후:
- 현재 reward에는 pelvis height 보상 없음

### 4.2 삭제: pelvis_min_height_score

기존 목적:
- 골반이 기준보다 10cm 이상 내려가면 강하게 막기

삭제 이유:
- 높이 기준이 실제 구조와 완전히 맞지 않아 학습을 왜곡할 수 있었다.
- 사용자가 골반 높이 패널티 삭제를 요청했다.

삭제 후:
- 현재 reward에는 최소 골반 높이 보상 없음

### 4.3 삭제: too_low_pelvis termination

기존 목적:
- episode_time > 0.3 이후 골반 높이가 `default_pelvis_height - 0.10`보다 낮으면 실패 처리

삭제 이유:
- 골반 높이 기준이 불확실했고, 정상 동작도 실패로 처리할 수 있었다.

삭제 후:
- 현재 fall_or_bad_pose는 주로 몸통/베이스 기울기 기준만 사용

### 4.4 삭제: self_proximity_score

기존 목적:
- 다리 부품끼리 가까워지거나 서로 뚫는 자세를 막기

사용했던 파라미터 예:
- self_proximity_min_lateral_gap: 0.055
- 이후 0.1로 강화
- self_proximity_sigma: 0.45
- 이후 0.25로 강화

삭제 이유:
- 부품끼리 닿는 penalty가 GPU/PhysX 접촉 계산과 reward 계산을 복잡하게 만들었다.
- 골반 roll joint limit을 수정해서 물리적으로 위험 각도를 줄이는 방향으로 전환했다.

삭제 후:
- 현재 reward에는 self proximity penalty 없음

### 4.5 삭제: non_foot_contact_score

기존 목적:
- 발이 아닌 부위가 지면 또는 다른 물체와 접촉하면 감점

사용했던 파라미터:
- non_foot_contact_threshold: 8.0
- non_foot_contact_force_scale: 30.0

삭제 이유:
- 내부 접촉/자기충돌과 결합되면서 PhysX patch buffer overflow 문제가 발생했다.
- 사용자가 부품끼리 닿는 패널티 삭제를 요청했다.

삭제 후:
- 현재 reward에는 non-foot contact penalty 없음

### 4.6 비활성화: self collision

기존:
- 로봇 내부 self collision을 켜서 부품끼리 겹침을 막으려 했다.

문제:
- PhysX patch buffer overflow 발생
- 많은 병렬 환경에서 내부 contact pair가 과도하게 증가

현재:
- `enabled_self_collisions = False`

## 5. 현재 V7-2 최종 보상체계

현재 `reward_humanoid_v7`의 최종 per-step reward는 다음과 같다.

```python
total_reward =
    0.16  * forward_velocity_score
  + 0.025 * yaw_velocity_score
  + 0.16  * alternating_gait_score
  + 0.14  * cross_forward_step_score
  + 0.08  * single_support_score
  + 0.075 * feet_clearance_score
  + 0.105 * upright_orientation_score
  + 0.06  * no_feet_slide_score
  + 0.075 * pressure_flat_contact_score
  + 0.04  * smooth_action_score
  + 0.025 * motor_safe_joint_usage_score
  + 0.055 * roll_suppression_score
```

### 5.1 현재 남아있는 보상 항목

| 항목 | weight | 목적 |
|---|---:|---|
| forward_velocity_score | 0.16 | 목표 전진 속도 추종 |
| yaw_velocity_score | 0.025 | 불필요한 yaw 회전 억제 및 명령 yaw 추종 |
| alternating_gait_score | 0.16 | 좌우 발 교대 패턴 유도 |
| cross_forward_step_score | 0.14 | 스윙발이 앞으로 이동하도록 유도 |
| single_support_score | 0.08 | 한발 지지 구간 형성 |
| feet_clearance_score | 0.075 | 스윙발이 약 4cm 들리도록 유도 |
| upright_orientation_score | 0.105 | 몸통이 기울어지지 않게 유지 |
| no_feet_slide_score | 0.06 | 접촉발 미끄러짐 억제 |
| pressure_flat_contact_score | 0.075 | 압력센서 기반 발바닥 평평한 접촉 유도 |
| smooth_action_score | 0.04 | 관절 명령 변화 부드럽게 유지 |
| motor_safe_joint_usage_score | 0.025 | 모터별 안전 관절 범위 사용 |
| roll_suppression_score | 0.055 | 발목 roll/골반 roll 과사용 억제 |

### 5.2 현재 보상 범위

- 각 score는 기본적으로 0.001~1.0 사이로 clamp
- 최종 reward도 0.001~1.0 사이로 clamp
- fall_or_bad_pose 발생 시 최종 reward = 0.001
- `rewards/iter`는 RL Games 집계값이라 1을 넘을 수 있음

## 6. 현재 Done / Termination 구조

현재 남아있는 주요 termination:

- roll/pitch 방향 기울기가 `max_tilt_xy`보다 크면 실패
- 현재 `max_tilt_xy = 0.4131759 rad`
- 약 23.67도 수준

삭제된 termination:
- pelvis too low
- non-foot contact 기반 실패
- self-proximity 기반 실패

## 7. 현재 보상체계의 의도

현재 reward는 다음 행동을 유도한다.

1. 앞으로 천천히 이동한다.
2. 좌우 발을 교대로 사용한다.
3. 스윙발은 앞으로 나간다.
4. 실제 다리가 안쪽으로 교차되는 scissor gait는 피한다.
5. 한 발은 지지하고 다른 발은 스윙하는 구간을 만든다.
6. 발은 너무 높게 들지 않고 약 4cm clearance를 목표로 한다.
7. 몸통은 자이로 기준으로 세워진 상태를 유지한다.
8. 지지발은 압력센서 기준으로 발바닥이 평평하게 닿도록 한다.
9. 발목 roll과 골반 roll을 과하게 쓰지 않는다.
10. 관절 명령은 급격히 변하지 않는다.

## 8. 현재 남은 리스크

1. self collision을 껐기 때문에 USD/관절 limit이 잘못되어 있으면 부품 겹침을 reward가 직접 막지 않는다.
2. pelvis height penalty를 삭제했기 때문에, 로봇이 낮은 자세를 학습할 가능성은 upright/velocity/clearance 보상으로 간접 제어해야 한다.
3. foot clearance target이 4cm라 실제 보행에서 충분한지 계속 확인해야 한다.
4. pressure_flat_contact가 너무 강하면 발을 끌거나 과도하게 flat contact만 추구할 수 있다.
5. roll_suppression이 너무 강하면 필요한 균형 보정까지 막을 수 있다.

## 9. 결론

V7 이전의 스테이지식 접근에서 반복되던 문제는 V7에서도 검증 대상으로 남았다. 특히 발목 roll 사용, 골반 roll 사용, 한쪽 발 편향, 발 톡톡 치기, 공중발 후방 이동, 오른발 전방 스윙 부족, 좌측 drift가 핵심 관찰 항목이었다.

V7에서는 보행을 처음부터 단일 reward로 학습시키는 방식으로 전환했다. 현재 V7-2 reward는 forward velocity, alternating gait, swing forward, single support, foot clearance, upright orientation, pressure flat contact, smooth action, motor-safe joint usage, roll suppression으로 구성되어 있다.

가장 최근 변경은 골반 높이 보상/패널티와 부품 접촉/근접 패널티를 삭제한 것이다. 그 결과 reward 구조가 단순해지고, PhysX self-collision/contact 병목이 줄어들었으며, 현재 학습에서는 GPU 사용률과 FPS가 더 안정적으로 나오고 있다.

## 10. V7-2 Forward Heading / Alternating Step 수정

### 10.1 수정 배경

GUI로 best checkpoint를 확인했을 때, 로봇이 시작 후 뒤로 돈 다음 뒤쪽 방향으로 걷는 현상이 확인되었다.

원인은 기존 `forward_velocity_score`가 world 기준 전진 방향이 아니라 로봇 몸체 기준 x축 속도를 사용했기 때문이다.

기존 구조:

```python
forward_velocity_score = exp(root_lin_vel_b[:, 0] - target_lin_vel_x)
```

이 방식은 로봇이 180도 회전해서 뒤쪽을 바라본 뒤, 자기 몸체 기준 앞으로 걸어도 전진 보상을 받을 수 있다. 즉, 실제 목표인 world +X 방향 보행이 아니라 body-frame forward walking을 보상하고 있었다.

또한 기존 `yaw_velocity_score`는 yaw 속도만 줄였기 때문에, 이미 뒤돌아선 자세 자체를 막지 못했다. 회전 후 yaw 속도가 0에 가까워지면 다시 보상을 받을 수 있었다.

### 10.2 새로 추가한 기준

이번 수정에서는 다음 기준을 추가했다.

| 항목 | 목적 |
|---|---|
| world_forward_velocity | 로봇 root가 world +X 방향으로 실제 이동할 때만 전진 보상 |
| heading_alignment_score | 로봇의 몸체 x축이 world +X 방향을 향하도록 유도 |
| world_forward_gate | world +X 방향 속도가 양수일 때만 전진/스텝 보상이 살아나도록 gate |
| alternating_forward_step_score | 왼발/오른발이 번갈아 swing foot이 되고, 해당 swing foot이 지지발보다 앞쪽으로 나갈 때 보상 |

### 10.3 변경된 보상 항목

기존 `forward_velocity_score`는 body-frame 속도에서 world-frame 속도로 변경했다.

변경 후 개념:

```python
world_forward_vel = root_lin_vel_w[:, 0]
heading_alignment_score = exp(heading_error)
world_forward_gate = clamp((world_forward_vel + 0.02) / target_lin_vel_x)

forward_velocity_score =
    exp(world_forward_vel - target_lin_vel_x)
  * heading_alignment_score
  * world_forward_gate
```

스윙발 전진 보상도 body-frame foot velocity가 아니라 world +X 방향 foot velocity를 사용하도록 변경했다.

```python
left_forward_vel = left_foot_lin_vel_w_x
right_forward_vel = right_foot_lin_vel_w_x
```

그리고 좌우 발을 번갈아 앞으로 내딛는 보상을 추가했다.

```python
swing_ahead =
    left_foot_x - right_foot_x   # left swing일 때
    right_foot_x - left_foot_x   # right swing일 때

alternating_forward_step_score =
    alternating_gait_score
  * swing_ahead_score
  * no_scissor_score
  * heading_alignment_score
  * world_forward_gate
```

### 10.4 현재 V7-2 수정 후 reward 구성

| 항목 | weight | 역할 |
|---|---:|---|
| forward_velocity_score | 0.14 | world +X 방향 목표 속도 추종 |
| heading_alignment_score | 0.09 | 로봇이 뒤돌지 않고 world +X 방향을 보도록 유도 |
| alternating_gait_score | 0.13 | 좌우 발 교대 패턴 형성 |
| cross_forward_step_score | 0.10 | swing foot이 world +X 방향으로 움직이도록 유도 |
| alternating_forward_step_score | 0.10 | 좌우 발을 번갈아 지지발보다 앞쪽으로 내딛도록 유도 |
| single_support_score | 0.07 | 한 발 지지 구간 형성 |
| feet_clearance_score | 0.065 | swing foot clearance 유지 |
| upright_orientation_score | 0.085 | IMU/자이로 기준 몸통 기울어짐 억제 |
| no_feet_slide_score | 0.05 | 접촉발 미끄러짐 억제 |
| pressure_flat_contact_score | 0.065 | 8개 압력센서 기반 발바닥 flat contact 유도 |
| smooth_action_score | 0.035 | 관절 명령 변화 부드럽게 유지 |
| motor_safe_joint_usage_score | 0.02 | 모터 스펙에 맞는 관절 사용 유도 |
| roll_suppression_score | 0.05 | 발목 roll/골반 roll 과사용 억제 |

합계 weight는 1.0이며, 각 score와 최종 per-step reward는 기존처럼 0.001~1.0 범위로 clamp한다.

### 10.5 실제 하드웨어 관점

자이로/IMU로 직접 측정 가능한 값:

- roll/pitch 기울기
- roll/pitch/yaw 각속도
- yaw rate
- 시작 방향 기준 yaw 적분값

따라서 `heading_alignment_score`는 실제 하드웨어에서는 순수 자이로 yaw rate만으로는 장시간 절대 heading을 정확히 보장하기 어렵고, 시작 방향 기준 yaw 적분 또는 추후 카메라/외부 odometry 보정이 필요하다.

압력센서로 직접 활용 가능한 값:

- 각 발 네 모서리 압력
- 발바닥 중심압 COP
- 앞/뒤 압력 편차
- 좌/우 압력 편차
- 지지발 flat contact 여부

이번 수정 후 목표는 뒤로 돌아서 걷는 편법을 막고, world +X 방향을 유지한 채 왼발/오른발을 번갈아 앞으로 내딛는 보행을 유도하는 것이다.

## 11. V7-2 로봇 앞축 재정의: +X가 아니라 -Y

### 11.1 확인 배경

Forward heading reward를 적용한 뒤, 로봇 앞쪽 축이 잘못 정의되었을 가능성이 제기되었다.

URDF와 foot STL을 확인한 결과:

- 오른발/왼발은 `x = +0.1 / x = -0.1` 쪽으로 나뉜다.
- 따라서 X축은 전진축이 아니라 좌우축이다.
- pitch 관절 축도 X축을 사용하므로, 사람 관절 구조상 X축은 좌우 방향 관절축으로 보는 것이 맞다.
- 발 STL의 foot mesh bbox는 Y 방향으로 더 길다.
- 발 mesh는 `Y = -108mm` 쪽으로 더 많이 튀어나오고, `Y = +72mm` 쪽은 상대적으로 짧다.

사용자 기준인 "발이 조금 더 앞으로 나와있는 쪽이 앞"을 적용하면, 로봇의 실제 앞 방향은 **-Y**이다.

### 11.2 기존 문제

직전 수정에서는 world +X를 앞 방향으로 가정했다.

이 가정은 로봇 CAD/URDF 축과 맞지 않는다.

문제점:

- 전진 속도 보상이 잘못된 world 축을 사용함
- heading alignment가 실제 로봇 앞이 아니라 옆 방향을 바라보게 만들 수 있음
- alternating forward step이 발을 실제 앞뒤가 아니라 좌우축 기준으로 판단할 수 있음
- lateral gap 계산도 Y축을 lateral로 잘못 사용할 수 있음

### 11.3 수정 내용

로봇 좌표 기준을 다음과 같이 고정했다.

| 의미 | 축 |
|---|---|
| 좌우축 | X |
| 앞뒤축 | Y |
| 위아래축 | Z |
| 실제 앞 방향 | -Y |

코드 상수:

```python
FORWARD_AXIS = 1
FORWARD_SIGN = -1.0
LATERAL_AXIS = 0
```

적용한 항목:

- world forward velocity: `-root_lin_vel_w[:, 1]`
- swing foot forward velocity: `-foot_lin_vel_w[:, 1]`
- swing foot ahead position: `-foot_pos_w[:, 1]`
- heading alignment local forward: local `-Y`
- desired world heading: world `-Y`
- lateral gap: foot relative position의 X축 차이
- pressure sensor front/rear corner: front를 `Y < 0` 쪽으로 재정의

### 11.4 사람 무릎 관절 관점

사람 무릎은 앞으로 꺾이지 않고 뒤쪽으로 접힌다.

현재 URDF에서 무릎 pitch 축은 좌우축인 X 방향을 기준으로 설정되어 있다. 이 구조도 X축이 전진축이 아니라 좌우 관절축이라는 해석과 일치한다.

따라서 앞으로 걷기 reward는 X축이 아니라 실제 앞뒤축인 Y축, 그중 발이 더 길게 튀어나온 -Y 방향을 기준으로 계산해야 한다.

## 12. V7-2 Body-Forward Locomotion 수정

### 12.1 결정

USD/URDF를 180도 회전하는 방식은 취소했다.

Fusion/CAD 화면에서 로봇 정면이 정상적으로 보이고, 무릎 관절도 사람 무릎처럼 앞쪽으로 꺾이지 않는 구조이므로 CAD 자체를 돌리는 것보다 학습 보상 기준을 바꾸는 방식이 안전하다고 판단했다.

### 12.2 핵심 변경

기존에는 로봇이 world `-Y` 방향으로 가야 높은 보상을 받도록 고정 heading reward가 섞여 있었다.

이번 수정에서는 다음처럼 바꿨다.

- 로봇 로컬 앞 방향은 그대로 `-Y`로 둔다.
- 전진 속도는 `root_lin_vel_w`를 현재 root heading의 forward vector에 투영해서 계산한다.
- swing foot forward velocity도 현재 root heading의 forward vector 기준으로 계산한다.
- swing foot이 앞에 있는지는 world 좌표가 아니라 root frame foot position의 forward 축으로 판단한다.
- 고정 world heading alignment 보상은 제거하고, 대신 yaw rate가 커지지 않도록 yaw stability 보상을 사용한다.

즉 목표는 특정 world 축으로만 걷는 것이 아니라, 로봇이 현재 바라보는 방향 기준으로 앞으로 걷는 것이다.

### 12.3 Sim-To-Real 설정

이번 run에서는 sim-to-real 변수는 기존 v7-2 기본값인 10~20% 수준으로 유지한다.

이 run에서 좋은 best checkpoint가 나오면, 다음 run에서 해당 best를 이어받아 sim-to-real 강도를 80~100% 수준으로 올리는 계획이다.

## 13. V7-2 Initial Heading Lock / ROS Hardware-Oriented Posture Reward

### 13.1 문제

Body-forward reward만 사용하면 로봇이 시작 직후 몸을 틀고, 틀어진 몸 방향을 새로운 forward로 삼는 편법이 생긴다.

GUI 확인 결과 다음 문제가 보였다.

- 시작하자마자 몸을 틀어서 걷는다.
- 골반 높이를 낮추고 다리를 벌린 낮은 자세로 이동한다.
- 앞으로 이동은 하지만 실제 하드웨어에 넣기 좋은 보행 자세가 아니다.

### 13.2 수정 방향

실제 ROS 하드웨어 적용을 고려해, BNO085가 시작 시점 yaw를 기준 방향으로 잡는 구조를 보상에 반영했다.

즉 episode 시작 시점의 root/BNO085 heading을 `initial_forward`로 저장하고, 이후 전진 보상은 계속 이 방향 기준으로 계산한다.

이를 통해 로봇이 몸을 돌려서 새로운 forward를 만드는 편법을 막는다.

### 13.3 추가된 보상 항목

- `heading_lock_score`: 현재 몸통 forward가 시작 yaw 기준 forward와 일치하면 보상
- `pelvis_posture_score`: 골반 높이가 기준 높이 `0.719m` 근처이고 너무 낮지 않으면 보상
- `stance_width_score`: 발 좌우 간격이 약 `0.20m` 근처이면 보상

### 13.4 유지된 현실 적용 기준

이번 수정은 ROS2 실제 하드웨어 적용을 고려해 다음 센서/계산값으로 설명 가능한 항목 위주로 구성했다.

- BNO085: 시작 yaw 기준 heading lock, yaw rate 안정성, roll/pitch 안정성
- 스마트서보: 관절각, 관절속도, 명령값, 관절 오차, 관절 리미트 margin
- 역기구학 계산: 발 위치, 발 높이, 발 좌우 간격, 골반 높이 추정
- 압력센서 8개: COP, 앞뒤/좌우 압력 균형, 지지발 flat contact

학습 reward는 시뮬레이터 내부 상태를 일부 사용하지만, policy observation과 실제 제어에 필요한 정보는 위 하드웨어 구성에서 계산 가능하도록 유지한다.

### 13.5 ROS 실행 방식과 대응

실제 ROS2 하드웨어에서는 사용자가 버튼을 클릭한 순간의 BNO085 yaw를 `desired_forward_yaw`로 저장한다.

그 후 정책 실행 중에는 다음 기준을 사용한다.

- 버튼 클릭 순간 바라보던 방향 = 앞으로 가야 할 기준 방향
- 현재 BNO085 yaw와 `desired_forward_yaw` 차이 = heading error
- heading error가 커지면 회전/몸틀기 편법으로 판단
- 전진 명령은 현재 몸이 임의로 튼 방향이 아니라 버튼 클릭 시점의 기준 방향으로 계산

시뮬레이션에서는 episode reset 시점의 root heading을 버튼 클릭 순간으로 보고 같은 방식으로 학습한다.

## 14. V7-2 8068 Epoch GUI Review / 보상 수정 보류 결정

### 14.1 확인한 영상

확인 영상:

`C:\Users\hsh\OneDrive\Videos\Captures\Isaac Sim 5.1.0 2026-07-01 06-23-33.mp4`

GUI 확인 결과, 로봇은 완전히 정지하거나 쓰러지는 상태는 아니고 실제로 걷는 형태가 나오기 시작했다.

다만 다음 문제가 관찰됐다.

- 오른다리가 왼다리에 비해 swing 시 앞으로 나가는 양이 작다.
- 시작 직후 골반 자체는 앞을 보고 있지만, 양쪽 다리의 hip yaw가 이상하게 틀어진 상태로 보행을 시작한다.
- 시작하고 두 발이 안정적으로 땅에 닿기 전 다리 yaw를 이용해 유리한 자세를 만든 뒤 걷는 경향이 있다.
- 발목 roll, 골반 roll처럼 약한 관절을 비정상적으로 꺾어서 버티는 행동도 계속 주의해야 한다.

### 14.2 해당 시점 학습 상태

8068 epoch 근처에서 로그를 확인했다.

- `Episode/Episode_Reward/reward_humanoid_v7`
  - 현재값: `0.668914`
  - 최고값: `0.674729`
  - 최근 100 평균: `0.669794`
  - 최근 300 평균: `0.667850`
- `rewards/iter`
  - 현재값: `5.350616`
  - 최고값: `5.403053`
  - 최근 100 평균: `5.357269`
  - 최근 300 평균: `5.341537`
- `fall_or_bad_pose`
  - 현재값: `0.005222`
  - 최근 100 평균: `0.007531`
- `time_out`
  - 현재값: `0.994778`
  - 최근 100 평균: `0.992488`
- `episode_lengths/iter`
  - 현재값: `477.224823`
  - 최대 episode 길이 기준 약 `480` 근처까지 유지됨

해석:

로봇은 대부분 episode 끝까지 살아남고 있으며, 넘어짐 비율도 낮다. 즉 학습이 망가진 상태는 아니다. 하지만 reward가 `0.66~0.67` 근처에서 오래 머무르기 때문에, 빠르게 좋은 보행으로 뚫고 올라가는 상태라기보다는 현재 보상 지형에서 한 자세에 수렴해 가는 plateau 성격이 강하다.

### 14.3 제안했던 수정 후보

문제를 고치기 위해 다음 보상 추가를 검토했다.

- `hip_yaw_suppression_score`
  - 대상 관절: `회전 172`, `회전 192`
  - 목적: 시작 직후 다리 yaw를 과하게 틀어 걷기 전 준비 자세를 만드는 편법 억제
- `startup_leg_yaw_stability_score`
  - episode 초반 또는 양발 접지 구간에서 hip yaw가 과하게 움직이면 감점
  - 목적: 시작하자마자 두 다리를 비틀어 보행을 시작하는 행동 억제
- `right_swing_forward_balance_score`
  - 오른발 swing forward distance가 왼발보다 작게 나오는 문제 보정
  - 목적: 좌우 swing 전진량 균형 확보
- 발목 roll / 골반 roll 과사용 감점 강화
  - 목적: 약한 roll 계열 관절로 버티는 자세 억제
- 두 발이 땅에 안정적으로 닿기 전 움직임 감점
  - 목적: 시작 직후 불안정한 leg yaw twisting 억제

### 14.4 실제 적용 여부

이 시점에서는 위 수정들을 실제 학습 코드에 적용하지 않았다.

한 번 patch를 시도했지만 코드 위치가 맞지 않아 적용에 실패했고, 결과적으로 실제 IsaacLab 학습 파일과 현재 실행 중인 학습에는 아무 변경도 들어가지 않았다.

즉 현재 계속 돌고 있는 run은 기존 `V7-2 Initial Heading Lock / ROS Hardware-Oriented Posture Reward` 설정 그대로이다.

### 14.5 최종 판단

수정은 보류하고, 현재 run을 조금 더 관찰하기로 결정했다.

이유:

- GUI상으로 완전히 실패한 움직임이 아니라 실제 보행 형태가 나오기 시작했다.
- reward/iter가 다시 조금씩 오르기 시작했다.
- 현재 상태에서 보상을 바로 바꾸면, 이제 막 잡히기 시작한 보행 리듬을 깨뜨릴 수 있다.
- fall 비율이 낮고 episode가 대부분 끝까지 유지되므로, 적어도 현재 정책은 안정된 생존/보행 후보를 찾고 있다.

따라서 다음 조건 중 하나가 보이면 수정한다.

- reward가 다시 장시간 정체된다.
- 오른발 swing 부족이 계속 고착된다.
- hip yaw twisting으로 시작하는 편법이 계속 유지된다.
- GUI에서 실제 ROS 하드웨어에 넣기 어려운 비정상 보행 자세가 지속된다.

다음 수정이 필요하면 우선순위는 다음과 같다.

1. hip yaw 과사용 감점
2. 시작 초반 / 양발 접지 구간 다리 yaw twisting 억제
3. 오른발 swing forward balance 보상
4. 발목 roll / 골반 roll 과사용 감점 강화

## 15. V7-2 Lateral Drift / Startup Guard Reward 적용

### 15.1 변경 계기

2026-07-01 13:30 영상(`Isaac Sim 5.1.0 2026-07-01 13-30-12.mp4`)에서 다음 문제가 확인되었다.

- 로봇이 걷기는 하지만 로봇 기준 왼쪽으로 점점 이동한다.
- 시작 직후 골반은 비교적 앞을 보는데 두 다리 yaw를 먼저 틀어서 보행을 시작한다.
- 오른다리 swing 전진량이 왼다리에 비해 작다.
- ROS2 실제 하드웨어에서 버튼을 누른 순간의 BNO085 기준 방향을 앞으로 삼아야 하므로, 시작 후 몸을 틀어 새 방향을 만드는 보행은 적합하지 않다.

### 15.2 실제 적용한 보상 변경

기존 V7-2 `initial heading / posture` 보상에서 USD, 질량, 모터 토크, 117개 입력, 네트워크, sim-to-real 10~20% 설정은 유지하고 보상 항목만 수정했다.

추가한 항목:

- `no_lateral_drift_score`
  - 처음 저장한 heading 기준 좌우 속도와 좌우 위치 오차를 함께 감점한다.
  - 로봇 기준 왼쪽 또는 오른쪽으로 새는 보행을 모두 억제한다.
- `hip_yaw_suppression_score`
  - hip yaw 계열 관절(`172`, `192`)이 과하게 틀어지는 것을 억제한다.
  - 시작 직후 다리를 yaw로 비틀어 보행 방향을 만드는 편법을 줄인다.
- `startup_pose_guard_score`
  - episode 초반 `0.75 s` 동안 기본 관절각에서 벗어나면 감점한다.
  - 양발 접지가 안정되기 전 관절을 조금이라도 크게 꺾는 행동을 낮은 점수로 만든다.
  - hip yaw 비틀림과 양발 접지 실패도 함께 감점한다.
- `right_swing_forward_balance_score`
  - 오른발 swing phase에서 오른발이 왼발보다 충분히 앞으로 나가지 못하면 점수가 낮아진다.
  - 기존 영상에서 보인 오른발 swing 부족을 직접 보정한다.
- 기존에 계산만 하고 총 보상에 약하게 반영되지 않았던 `yaw_stability_score`를 총 보상에 포함했다.

### 15.3 새 보상 가중치

최종 per-step reward는 계속 `0.001 ~ 1.0`으로 clamp한다.

| 항목 | weight | 역할 |
|---|---:|---|
| `forward_velocity_score` | 0.090 | 목표 전진 속도 추종 |
| `heading_lock_score` | 0.080 | 시작 heading 기준 방향 유지 |
| `no_lateral_drift_score` | 0.070 | 로봇 기준 좌우 드리프트 억제 |
| `yaw_stability_score` | 0.040 | 몸통 yaw 회전 속도 억제 |
| `alternating_gait_score` | 0.075 | 좌우 발 교대 패턴 |
| `cross_forward_step_score` | 0.055 | swing foot가 앞으로 이동 |
| `alternating_forward_step_score` | 0.060 | 교대 발 전진 위치 확보 |
| `right_swing_forward_balance_score` | 0.035 | 오른발 swing 부족 보정 |
| `single_support_score` | 0.040 | 한발 지지 구간 유도 |
| `feet_clearance_score` | 0.040 | swing foot clearance 유지 |
| `upright_orientation_score` | 0.055 | BNO085/IMU 기반 몸통 upright |
| `no_feet_slide_score` | 0.030 | 접지발 미끄러짐 억제 |
| `pressure_flat_contact_score` | 0.035 | 네 모서리 압력 균형으로 발바닥 flat 유도 |
| `smooth_action_score` | 0.025 | 명령 변화 부드럽게 유지 |
| `motor_safe_joint_usage_score` | 0.015 | 모터별 안전 사용 범위 |
| `roll_suppression_score` | 0.055 | 골반/발목 roll 과사용 억제 |
| `hip_yaw_suppression_score` | 0.045 | 다리 yaw 비틀림 억제 |
| `startup_pose_guard_score` | 0.070 | 시작 초반 양발 접지 전 관절 꺾임 억제 |
| `pelvis_posture_score` | 0.045 | 골반 자세 유지 |
| `stance_width_score` | 0.040 | 발 간격 20 cm 근처 유지 |

가중치 합계는 `1.000`이다.

### 15.4 학습 설정

새 run 이름:

- `humanoid_v7_2_lateral_startup_guard_obs117_pelvisBody719_net512_8192env_mb32768_sim10to20`

시작 checkpoint:

- `humanoid_v7_2_initial_heading_posture_obs117_pelvisBody719_net512_8192env_mb32768_sim10to20.pth`

유지한 설정:

- 입력: 117개
- 네트워크: `117 -> 512 -> 256 -> 128 -> 64 -> 12`
- 병렬 환경: `8192`
- horizon length: `32`
- minibatch size: `32768`
- max epoch: `100000`
- sim-to-real 변수: 기본 10~20% 범위
- USD/질량/모터 토크/압력센서/자이로센서 구성 유지

### 15.5 기대되는 변화

이 수정 후 GUI에서 봐야 하는 정상 변화:

- 시작하자마자 다리 yaw를 틀어 방향을 바꾸는 동작이 줄어든다.
- 로봇 기준 왼쪽으로 계속 새는 현상이 줄어든다.
- 오른발 swing이 왼발보다 짧게 나오는 현상이 완화된다.
- 버튼 클릭 순간의 BNO085 heading을 기준으로 앞으로 걷는 형태에 가까워진다.

주의점:

- 초반 관절 움직임을 제한하는 보상이 추가되었기 때문에, 처음 수백 epoch 동안 reward가 이전보다 낮게 보일 수 있다.
- 목표는 단순히 reward 최고치를 높이는 것이 아니라 ROS2 실기에서 쓸 수 있는 방향 안정 보행을 만드는 것이다.

## 16. V7-2 Swing Landing Quality Reward 적용

### 16.1 수정 배경

GUI 확인 결과 보행 자체는 나오기 시작했지만, 공중발이 지지발보다 충분히 앞쪽으로 착지하지 않고 바로 옆에 붙는 경향이 있었다. 특히 화면을 뒤에서 보고 있을 때 오른발이 왼발 앞쪽으로 나가지 않는 것처럼 보였으나, 오른발만 직접 보정하면 좌우 비대칭 정책이 생길 위험이 있다.

따라서 이번 수정은 오른발 전용 조건이 아니라, 현재 swing 중인 발이 어느 발이든 동일하게 적용되는 착지 품질 보상으로 정리했다.

### 16.2 제거한 항목

이전 보상에서 오른발 swing 부족을 직접 보정하던 항목을 제거했다.

- 제거: `right_swing_forward_balance_score`

제거 이유:

- 오른발만 따로 밀어주는 보상은 왼발/오른발 동작 비대칭을 만들 수 있다.
- 카메라 시점이 뒤쪽인지 앞쪽인지에 따라 사람이 판단하는 오른발/왼발 착지 위치가 헷갈릴 수 있다.
- ROS2 실기에서는 특정 발 전용 보정보다 “현재 공중발 기준”의 일반 규칙이 더 안정적이다.

### 16.3 새로 추가한 항목

추가 항목:

- `swing_landing_quality_score`

적용 조건:

- 현재 gait phase의 후반부일 것
  - `swing_progress > 0.55`
- 현재 swing foot가 착지 접촉을 만들었을 것
- 착지 순간 swing foot가 stance foot보다 initial heading 기준 앞쪽에 있을 것
- 좌우 발 간격이 정상 stance width 근처일 것
- 다리가 안쪽으로 교차되는 scissor gait가 아닐 것
- 시작 버튼을 눌렀을 때의 BNO085/root heading에서 크게 돌아가지 않을 것

구성:

- `swing_landing_forward_score`: 공중발이 지지발보다 진행방향 앞쪽에 착지하는지 평가
- `swing_landing_lateral_score`: 착지 시 좌우 발 간격이 약 20 cm 근처인지 평가
- `no_scissor_score`: 다리가 안쪽으로 교차하지 않는지 평가
- `heading_lock_score`: 시작 heading을 유지하는지 평가
- `forward_motion_gate`: 실제 전진 중인지 평가

계산 의도:

```text
swing_landing_quality_score =
    late_swing_landing_gate
  * forward_landing_score
  * lateral_stance_width_score
  * no_scissor_score
  * heading_lock_score
  * forward_motion_gate
```

실제 코드에서는 forward landing을 더 중요하게 보기 위해 전진 착지 75%, 좌우 간격 25%로 섞었다.

### 16.4 새 보상 가중치

최종 per-step reward는 계속 `0.001 ~ 1.0`으로 clamp한다.

| 항목 | weight | 역할 |
|---|---:|---|
| `forward_velocity_score` | 0.085 | 목표 전진 속도 추종 |
| `heading_lock_score` | 0.075 | 시작 heading 기준 방향 유지 |
| `no_lateral_drift_score` | 0.065 | 로봇 기준 좌우 드리프트 억제 |
| `yaw_stability_score` | 0.040 | 몸통 yaw 회전 속도 억제 |
| `alternating_gait_score` | 0.075 | 좌우 발 교대 패턴 |
| `cross_forward_step_score` | 0.055 | swing foot가 앞으로 이동 |
| `alternating_forward_step_score` | 0.055 | 교대 발 전진 위치 확보 |
| `swing_landing_quality_score` | 0.080 | 현재 공중발이 지지발 앞쪽에 정상 간격으로 착지 |
| `single_support_score` | 0.040 | 한발 지지 구간 유도 |
| `feet_clearance_score` | 0.040 | swing foot clearance 유지 |
| `upright_orientation_score` | 0.050 | BNO085/IMU 기반 몸통 upright |
| `no_feet_slide_score` | 0.030 | 접지발 미끄러짐 억제 |
| `pressure_flat_contact_score` | 0.035 | 네 모서리 압력 균형으로 발바닥 flat 유도 |
| `smooth_action_score` | 0.025 | 명령 변화 부드럽게 유지 |
| `motor_safe_joint_usage_score` | 0.015 | 모터별 안전 사용 범위 |
| `roll_suppression_score` | 0.055 | 골반/발목 roll 과사용 억제 |
| `hip_yaw_suppression_score` | 0.045 | 다리 yaw 비틀림 억제 |
| `startup_pose_guard_score` | 0.060 | 시작 초반 양발 접지 전 관절 꺾임 억제 |
| `pelvis_posture_score` | 0.040 | 골반 자세 유지 |
| `stance_width_score` | 0.035 | 발 간격 20 cm 근처 유지 |

가중치 합계는 `1.000`이다.

### 16.5 학습 설정

새 run 이름:

- `humanoid_v7_2_swing_landing_quality_obs117_pelvisBody719_net512_8192env_mb32768_sim10to20`

시작 checkpoint:

- `humanoid_v7_2_lateral_startup_guard_obs117_pelvisBody719_net512_8192env_mb32768_sim10to20.pth`

유지한 설정:

- 입력: 117개
- 네트워크: `117 -> 512 -> 256 -> 128 -> 64 -> 12`
- 병렬 환경: `8192`
- horizon length: `32`
- minibatch size: `32768`
- max epoch: `100000`
- sim-to-real 변수: 기본 10~20% 범위
- USD/질량/모터 토크/압력센서/자이로센서 구성 유지

### 16.6 기대되는 변화

GUI에서 봐야 하는 정상 변화:

- 특정 오른발만 보정하지 않고 양쪽 swing foot 모두 착지 위치가 앞으로 잡힌다.
- 발이 지지발 바로 옆에 내려오는 현상이 줄어든다.
- 다리가 서로 안쪽으로 교차하는 scissor gait가 줄어든다.
- 로봇 기준 왼쪽으로 흐르는 보행은 기존 `no_lateral_drift_score`가 계속 억제한다.
- 시작 heading 기준 앞으로 걷는 규칙은 유지된다.

주의점:

- 착지 품질 조건이 추가되었기 때문에 초반에는 reward가 잠깐 낮아질 수 있다.
- 정상적으로 학습되면 `reward/iter`보다 GUI에서 swing foot 착지 위치가 먼저 좋아지는지 확인해야 한다.

## 17. V7-2 Lateral Best 기반 No-Backward Touchdown Reward

### 17.1 변경 계기

`swing_landing_quality` run의 GUI/영상 확인 결과, 로봇이 전진은 하지만 다음 꼼수가 확인되었다.

- 시작 직후 왼발을 뒤로 빼서 뒤쪽에 착지한다.
- 오른발을 앞으로 딛어서 전체 전진 보상을 얻는다.
- 결과적으로 오른발만 주도적으로 앞으로 가고, 왼발은 뒤쪽 안정화 발처럼 쓰인다.

따라서 이번 run은 왼발 뒤착지 습관이 생긴 최신 checkpoint가 아니라, 그 이전의 “오른발 전진량이 부족하던 lateral/startup guard best”에서 다시 시작한다.

### 17.2 시작 checkpoint

사용 checkpoint:

- `C:\tmp\v7_2\logs\rl_games\humanoid_v7_2_lateral_startup_guard_obs117_pelvisBody719_net512_8192env_mb32768_sim10to20\2026-07-01_13-45-55\nn\humanoid_v7_2_lateral_startup_guard_obs117_pelvisBody719_net512_8192env_mb32768_sim10to20.pth`

선택 이유:

- 이미 걷는 형태는 어느 정도 나왔다.
- 아직 왼발을 뒤로 빼서 착지하는 습관이 강하게 굳기 전이다.
- 오른발 전진량 부족만 보상으로 보정하기 좋은 출발점이다.

### 17.3 추가한 핵심 보상/감점

이번 수정은 오른발 전용 보상을 넣지 않는다. 모든 항목은 현재 swing foot 기준으로 좌우 공통 적용한다.

추가 항목:

- `early_no_backward_score`
  - episode 초반 `0.90 s` 동안 어느 발이든 시작 위치보다 뒤로 빠지면 점수가 낮아진다.
  - 허용 오차: `0.015 m`
- `late_swing_no_backward_score`
  - swing progress가 `0.45`를 넘은 뒤, swing foot가 support foot 뒤쪽에 있으면 점수가 낮아진다.
  - 허용 오차: `0.005 m`
- `swing_landing_no_behind_score`
  - swing foot가 착지하는 순간 support foot 뒤쪽이면 점수가 낮아진다.
- `alternating_forward_touchdown_score`
  - 왼발/오른발 어느 발이든 자기 차례에 착지할 때 support foot보다 앞쪽이면 점수가 높다.
  - 다리 교차, heading 이탈, 실제 전진 부족이 있으면 점수가 낮아진다.

유지 항목:

- `swing_landing_quality_score`
  - swing foot가 앞쪽에 착지하고 stance width를 유지하면 보상.
- `no_lateral_drift_score`
  - 좌우로 새는 움직임 억제.
- `startup_pose_guard_score`
  - 시작 직후 관절 비틀림 억제.
- `hip_yaw_suppression_score`, `roll_suppression_score`
  - hip yaw, pelvis/ankle roll 꼼수 억제.

### 17.4 새 보상 가중치

최종 per-step reward는 계속 `0.001 ~ 1.0`으로 clamp한다.

| 항목 | weight | 역할 |
|---|---:|---|
| `forward_velocity_score` | 0.085 | 목표 전진 속도 추종 |
| `heading_lock_score` | 0.075 | 시작 heading 기준 방향 유지 |
| `no_lateral_drift_score` | 0.065 | 좌우 드리프트 억제 |
| `yaw_stability_score` | 0.035 | yaw 회전 속도 억제 |
| `alternating_gait_score` | 0.060 | 좌우 교대 패턴 |
| `cross_forward_step_score` | 0.045 | swing foot 전진 속도 |
| `alternating_forward_step_score` | 0.040 | swing foot 전진 위치 |
| `swing_landing_quality_score` | 0.065 | 앞쪽 착지와 stance width |
| `alternating_forward_touchdown_score` | 0.025 | 자기 차례의 앞쪽 착지 |
| `swing_landing_no_behind_score` | 0.035 | 뒤쪽 착지 억제 |
| `late_swing_no_backward_score` | 0.025 | swing 후반 뒤쪽 유지 억제 |
| `early_no_backward_score` | 0.030 | 시작 직후 발 뒤로 빼기 억제 |
| `single_support_score` | 0.035 | 한발 지지 |
| `feet_clearance_score` | 0.035 | swing foot clearance |
| `upright_orientation_score` | 0.045 | BNO085/IMU upright |
| `no_feet_slide_score` | 0.025 | 접지발 미끄러짐 억제 |
| `pressure_flat_contact_score` | 0.030 | 압력 기반 발바닥 flat |
| `smooth_action_score` | 0.020 | 부드러운 관절 명령 |
| `motor_safe_joint_usage_score` | 0.015 | 모터 안전 사용 |
| `roll_suppression_score` | 0.050 | 골반/발목 roll 과사용 억제 |
| `hip_yaw_suppression_score` | 0.040 | 다리 yaw 비틀림 억제 |
| `startup_pose_guard_score` | 0.055 | 시작 자세 안정화 |
| `pelvis_posture_score` | 0.035 | 골반 자세 유지 |
| `stance_width_score` | 0.030 | 발 간격 20 cm 근처 유지 |

가중치 합계는 `1.000`이다.

### 17.5 새 run 이름

- `humanoid_v7_2_lateral_best_no_backward_touchdown_obs117_pelvisBody719_net512_8192env_mb32768_sim10to20`

유지 설정:

- 입력: 117개
- 네트워크: `117 -> 512 -> 256 -> 128 -> 64 -> 12`
- 병렬 환경: `8192`
- horizon length: `32`
- minibatch size: `32768`
- sim-to-real 변수: 기본 10~20%
- USD/질량/모터 토크/압력센서/자이로센서 구성 유지

## 18. V7-2 Forward Axis 교차검증

### 18.1 검증 계기

GUI에서 로봇이 보행할 때 발이 앞쪽으로 나가는지 뒤쪽으로 나가는지 시각적으로 헷갈리는 장면이 있었다. 특히 왼발을 뒤로 빼고 오른발을 앞으로 딛는 패턴이 보였기 때문에, 보상 코드의 “앞쪽” 정의가 실제 USD/URDF/발바닥 기준과 일치하는지 교차검증했다.

### 18.2 코드 기준

현재 보상 코드의 forward 정의:

```python
FORWARD_AXIS = 1
FORWARD_SIGN = -1.0
LATERAL_AXIS = 0
```

의미:

- 로봇 로컬 `Y`축을 앞뒤 축으로 사용한다.
- `-Y` 방향을 앞쪽으로 본다.
- `X`축을 좌우 방향으로 본다.

현재 reward에서 전진 속도, foot forward position, swing foot ahead 판정은 이 forward 기준을 사용한다.

### 18.3 USD geometry 검증

분석 대상 USD:

- `C:\Users\hsh\OneDrive\바탕 화면\humanoid_v7\v7-2\robot_asset\mass_6993g_robot\usd_with_sensors\humanoid_v7_mass6993_sensors.usd`

USD Python API가 한글 경로를 깨뜨리는 문제가 있어, 분석용으로만 `C:\tmp\v7_2_usd_forward_inspect\usd_with_sensors`에 복사해서 열었다. 학습용 USD 파일 자체는 수정하지 않았다.

USD stage 정보:

- default prim: `/humanoid_no_body`
- up axis: `Z`
- meters per unit: `1.0`

발 링크 원점:

| 링크 | origin X | origin Y | origin Z |
|---|---:|---:|---:|
| `left_foot_1` | `+0.1000` | `0.0000` | `0.1475` |
| `right_foot_1` | `-0.1000` | `0.0000` | `0.1475` |

좌우 발 원점 차이:

- `left - right = +0.2000 m` on X

따라서 좌우 간격 축은 `X`가 맞다.

발판 bbox:

| 링크 | foot origin 기준 Y min | foot origin 기준 Y max |
|---|---:|---:|
| `left_foot_1` | `-0.1080 m` | `+0.0720 m` |
| `right_foot_1` | `-0.1080 m` | `+0.0720 m` |

발판이 원점 기준 `-Y` 방향으로 `10.8 cm`, `+Y` 방향으로 `7.2 cm` 뻗어 있다. 발 앞쪽이 더 길게 나와 있는 쪽이라고 보면, 실제 USD geometry 기준 앞쪽은 `-Y`가 맞다.

### 18.4 압력센서 / 발바닥 corner 기준

보상 코드에서 사용하는 발바닥 corner 좌표:

```python
FOOT_CORNER_XY = (
    (-0.03, -0.055),  # front_left
    (0.03, -0.055),   # front_right
    (-0.03, 0.045),   # rear_left
    (0.03, 0.045),    # rear_right
)
```

의미:

- front pressure corner는 `Y = -0.055`
- rear pressure corner는 `Y = +0.045`

따라서 압력센서 모델도 `front = -Y`로 정의되어 있다. 이는 USD 발판 bbox와 보상 코드의 `FORWARD_SIGN = -1.0`과 일치한다.

### 18.5 URDF 기준

URDF 파일:

- `C:\Users\hsh\OneDrive\바탕 화면\humanoid_v7\v7-2\robot_asset\mass_6993g_robot\source_urdf\humanoid_no_body_description\humanoid_no_body.urdf`

확인된 foot 관련 정보:

- `right_foot_1` visual/collision mesh:
  - origin: `xyz="0.1 0.0 -0.1475"`
  - mesh: `right_foot_1.stl`
- `left_foot_1` visual/collision mesh:
  - origin: `xyz="-0.1 0.0 -0.1475"`
  - mesh: `left_foot_1.stl`
- foot inertial origin:
  - Y 약 `-0.0155`

foot inertial center도 약간 `-Y` 쪽으로 치우쳐 있어, 발 질량 중심 역시 앞쪽이 `-Y`라는 해석과 충돌하지 않는다.

### 18.6 결론

현재 기준에서 “앞쪽” 정의가 반대로 뒤집힌 증거는 없다.

검증 결과:

- 코드 forward: `-Y`
- USD 발판 긴 쪽: `-Y`
- 압력센서 front corner: `-Y`
- 좌우 축: `X`

따라서 현재 이상한 보행은 forward 축이 반대로 잡혀서 생긴 문제라기보다, reward가 아직 다음 꼼수를 완전히 막지 못해서 생기는 문제로 보는 것이 맞다.

관찰된 꼼수:

- 한쪽 발을 뒤로 빼서 안정성을 만들고
- 다른 한쪽 발만 앞으로 내딛어 전진 보상을 얻는 행동

### 18.7 남은 주의점

현재 reward의 foot ahead 판정은 발끝 위치가 아니라 `left_foot_1`, `right_foot_1` 링크 원점 위치를 기준으로 한다.

따라서 발이 yaw/roll로 비틀릴 경우, 사람이 화면에서 보는 발끝/toe 위치와 reward가 보는 foot origin 위치가 다르게 느껴질 수 있다.

다음 개선 후보:

- swing foot link origin만 보지 말고, foot front virtual point도 함께 계산한다.
- 예:
  - `foot_front_point = foot_origin + current_foot_rotation * (0, -0.08, 0)`
  - 이 front point가 support foot front point보다 앞에 있는지도 검사한다.
- 이렇게 하면 발 원점은 앞에 있지만 발끝이 옆이나 뒤로 비틀리는 경우를 더 잘 잡을 수 있다.

단, 현재 교차검증 결과만 보면 축 자체를 `+Y`로 바꾸는 것은 적절하지 않다.

## 19. V7-2 pth v7.2.5 실행 파일 누락 보완

### 19.1 목적

사용자가 “오른발이 앞으로 안 나가면서도 그나마 걸었던 best pth”를 나중에 다시 열 수 있어야 한다고 지적했다.

기존 보고서 폴더에는 `.pth` 사본은 있었지만, 해당 `.pth`를 Isaac Sim GUI로 여는 실행 `.cmd` 파일이 빠져 있었다.

이번 업데이트는 학습 코드를 바꾸는 작업이 아니라, 재현성과 추적성을 위해 pth 실행 파일을 보고서 체계에 추가한 작업이다.

### 19.2 대상 pth

대상 체크포인트:

```text
C:\tmp\v7_2\logs\rl_games\humanoid_v7_2_lateral_startup_guard_obs117_pelvisBody719_net512_8192env_mb32768_sim10to20\2026-07-01_13-45-55\nn\humanoid_v7_2_lateral_startup_guard_obs117_pelvisBody719_net512_8192env_mb32768_sim10to20.pth
```

선택 이유:

- `no_backward_touchdown` 보상을 추가하기 전 기준 best 파일이다.
- GUI 관찰상 전진은 어느 정도 했지만, 오른발이 왼발보다 충분히 앞으로 나오지 않는 문제가 있었다.
- 이후 “스윙발이 지지발보다 앞쪽에 착지하면 보상”, “뒤쪽 착지 감점”을 추가하는 기준점으로 사용했다.

### 19.3 추가한 실행 파일

보고서용 실행 파일 이름:

```text
pth v7.2.5 right_foot_not_forward_best_GUI.cmd
```

저장 위치:

```text
C:\Users\hsh\OneDrive\바탕 화면\보고서용 정리\humanoid_v7\pth_실행파일\pth v7.2.5 right_foot_not_forward_best_GUI.cmd
```

작업용 원본 위치:

```text
C:\Users\hsh\OneDrive\문서\New project\reports\pth v7.2.5 right_foot_not_forward_best_GUI.cmd
```

### 19.4 실행 방식

이 실행 파일은 학습 중인 headless 프로세스를 건드리지 않고, 별도 1개 환경 GUI play만 실행한다.

핵심 실행 조건:

- task: `Isaac-Pleas-OneFootBalance-v0`
- num_envs: `1`
- device: `cuda:0`
- checkpoint: 위 lateral startup guard best `.pth`
- real-time: enabled
- GUI 우회:
  - `isaaclab.python.rendering.kit`
  - `--reset-user`
  - renderer active GPU 0
  - multi-GPU off
  - DLSSG off
  - driver version verify off

### 19.5 이후 보상 변경과의 관계

이 pth는 현재 돌리는 `no_backward_touchdown` run보다 이전 기준이다.

현재 run에서 추가된 주요 항목:

- 시작 직후 발을 뒤로 빼는 행동 감점
- swing foot이 support foot 뒤에 남는 행동 감점
- touchdown 시 swing foot이 support foot보다 앞쪽에 착지하면 보상
- 좌우 특정 발 고정 보상이 아니라 current swing foot 기준 좌우 대칭 보상

따라서 `pth v7.2.5`는 “오른발이 앞으로 덜 나가던 문제를 확인하기 위한 기준 pth”로 보존한다.

## 20. V7-2 pth v7.2.5 기반 sim-to-real 30-50% 재학습

### 20.1 목적

사용자가 `pth v7.2.5 right_foot_not_forward_best` 정책을 기준으로, 해당 파일을 학습했던 코드 흐름은 그대로 유지하고 sim-to-real 변수만 조정해서 학습을 다시 시작하자고 요청했다.

이번 변경의 원칙:

- 보상체계는 `lateral_startup_guard` 기준 그대로 사용한다.
- 네트워크 크기와 observation/action 구조는 그대로 유지한다.
- USD, 질량, 모터 토크, 관절 매칭은 그대로 유지한다.
- 변경하는 것은 sim-to-real 관련 현실 오차 변수뿐이다.

### 20.2 기준 체크포인트

이어받는 pth:

```text
C:\tmp\v7_2\logs\rl_games\humanoid_v7_2_lateral_startup_guard_obs117_pelvisBody719_net512_8192env_mb32768_sim10to20\2026-07-01_13-45-55\nn\humanoid_v7_2_lateral_startup_guard_obs117_pelvisBody719_net512_8192env_mb32768_sim10to20.pth
```

이 pth는 GUI 관찰에서 “전진은 하지만 오른발이 충분히 앞으로 나오지 않는” 동작을 보였던 기준 파일이다.

### 20.3 새 run 이름

```text
humanoid_v7_2_lateral_startup_guard_sim30to50_from_v725_obs117_pelvisBody719_net512_8192env_mb32768
```

### 20.4 변경한 sim-to-real 변수

기존 10~20% 수준에서 중간 강화 30~50% 수준으로 조정했다.

센서/백래쉬:

| 항목 | 기존 | 변경 |
|---|---:|---:|
| IMU angular velocity noise std | `0.010` | `0.020` |
| IMU angular velocity bias std | `0.002` | `0.004` |
| projected gravity noise std | `0.010` | `0.020` |
| joint position backlash std | `0.006` | `0.012` |
| joint position backlash bias std | `0.003` | `0.006` |

서보 지연:

| 항목 | 기존 | 변경 |
|---|---:|---:|
| `SERVO_MIN_DELAY_STEPS` | `1` | `2` |
| `SERVO_MAX_DELAY_STEPS` | `3` | `5` |

마찰/재질 랜덤화:

| 항목 | 기존 | 변경 |
|---|---:|---:|
| static friction range | `(1.02, 1.20)` | `(0.90, 1.35)` |
| dynamic friction range | `(0.82, 1.05)` | `(0.70, 1.20)` |
| restitution range | `(0.0, 0.02)` | `(0.0, 0.04)` |

액추에이터/관절 랜덤화:

| 항목 | 기존 | 변경 |
|---|---:|---:|
| stiffness scale | `(0.90, 1.10)` | `(0.75, 1.25)` |
| damping scale | `(0.85, 1.15)` | `(0.70, 1.30)` |
| joint friction scale | `(0.85, 1.15)` | `(0.70, 1.30)` |
| armature scale | `(0.90, 1.10)` | `(0.80, 1.20)` |

### 20.5 그대로 유지한 것

보상체계:

- `forward_velocity_score`
- `heading_lock_score`
- `no_lateral_drift_score`
- `yaw_stability_score`
- `alternating_gait_score`
- `cross_forward_step_score`
- `alternating_forward_step_score`
- `right_swing_forward_balance_score`
- `single_support_score`
- `feet_clearance_score`
- `upright_orientation_score`
- `no_feet_slide_score`
- `pressure_flat_contact_score`
- `smooth_action_score`
- `motor_safe_joint_usage_score`
- `roll_suppression_score`
- `hip_yaw_suppression_score`
- `startup_pose_guard_score`
- `pelvis_posture_score`
- `stance_width_score`

학습 설정:

- num_envs: `8192`
- horizon_length: `32`
- minibatch_size: `32768`
- max_iterations: `100000`
- network: `117 -> 512 -> 256 -> 128 -> 64 -> 12`
- normalize_input: `True`
- normalize_value: `True`

### 20.6 실행 파일

작업용 실행 파일:

```text
C:\tmp\v7_2\RUN_v7_2_lateral_startup_guard_sim30to50_from_v725_headless_8192.cmd
```

보고서용 실행 파일:

```text
C:\Users\hsh\OneDrive\바탕 화면\보고서용 정리\humanoid_v7\rl_강화학습코드\rl v7.2.9\RUN_v7_2_lateral_startup_guard_sim30to50_from_v725_headless_8192.cmd
```

### 20.7 주의

이번 run은 행동 보상 자체를 바꾼 것이 아니다. 따라서 gait 패턴이 좋아지는 목적보다는, 기존 `v7.2.5` 정책이 더 강한 현실 오차 조건에서도 버티는지 확인하는 성격이 강하다.

## 21. V7-2 sim-to-real 10-20%와 30-50% 보행 차이 관찰

### 21.1 관찰 내용

사용자가 GUI로 확인한 결과, sim-to-real 강도에 따라 발 착지 패턴이 다르게 나타났다.

10~20% sim-to-real 조건에서 관찰된 패턴:

- 오른발이 스윙할 때 왼발 바로 옆에 착지했다.
- 전진은 하지만 오른발이 충분히 앞쪽으로 나가지 않는 경향이 있었다.
- 이 동작은 `pth v7.2.5 right_foot_not_forward_best`로 따로 보존했다.

30~50% sim-to-real 조건에서 관찰된 패턴:

- 로봇 기준 왼발이 스윙할 때, 오른발 착지발과 거의 동일선상에 착지했다.
- 즉 더 멀리 앞으로 보내기보다, 지지발 선 근처에서 안정성을 먼저 확보하는 형태가 나타났다.
- fall 비율은 낮고 episode length는 높게 유지되어, 정책이 안정성을 우선하는 쪽으로 적응 중인 것으로 판단했다.

### 21.2 추정 원인

30~50% run에서는 보상체계 자체는 바꾸지 않고 sim-to-real 변수만 강화했다.

강화한 항목:

- IMU angular velocity noise 증가
- projected gravity noise 증가
- joint position backlash 증가
- servo delay 증가
- 마찰 랜덤화 범위 증가
- actuator stiffness/damping 랜덤화 범위 증가
- joint friction/armature 랜덤화 범위 증가

이 때문에 로봇이 발을 멀리 스윙할수록 넘어질 위험이 커진다. PPO 입장에서는 다음 두 선택지를 비교하게 된다.

- 발을 더 앞으로 보내서 전진 보상을 조금 더 얻기
- 발을 짧게 보내서 넘어지지 않고 episode length와 생존 보상을 유지하기

30~50% 조건에서는 후자가 더 안정적인 전략으로 선택될 수 있다.

### 21.3 정책 비대칭과의 관계

현재 30~50% run은 완전히 새로 시작한 것이 아니라 `pth v7.2.5`에서 이어받았다.

`pth v7.2.5` 자체의 특징:

- 어느 정도 전진은 가능했다.
- 하지만 오른발 스윙이 충분히 앞으로 나오지 않는 비대칭이 있었다.

sim-to-real 변수를 강화하면 이 비대칭이 바로 사라지기보다, 더 안전한 방향으로 눌릴 수 있다. 그래서 한쪽 발은 옆착지 또는 동일선상 착지처럼 보이고, 다른 한쪽도 멀리 보내기보다 짧은 스텝으로 안정성을 확보하는 패턴이 나타날 수 있다.

### 21.4 현재 판단

현재 수치상으로는 학습이 망가진 상태는 아니다.

확인된 상태:

- reward는 다시 상승하여 `0.63` 근처 best를 만들었다.
- episode length는 `473~476 / 480` 수준으로 높다.
- fall 비율은 대략 `1~2%` 수준까지 낮아졌다.

따라서 즉시 보상체계를 바꾸기보다, 더 돌려서 동일선상 착지가 앞으로 조금씩 넘어가는지 확인하는 것이 좋다.

### 21.5 다음 수정 후보

만약 충분히 더 돌린 뒤에도 스윙발이 계속 지지발과 동일선상에만 착지한다면, 다음 보상 항목을 아주 약하게 추가할 수 있다.

```text
swing_foot_slightly_ahead_touchdown_score
```

의도:

- 스윙발이 지지발보다 현재 로봇 전방 기준으로 `1~3 cm` 정도만 앞에 착지하면 작은 보상.
- 너무 큰 보상으로 넣지 않는다.
- 강하게 넣으면 다시 한쪽 발만 쓰기, 뒤로 빼기, 비정상적인 발 비틀기 같은 꼼수가 생길 수 있다.

현재 결정:

- 아직은 보상체계를 바꾸지 않는다.
- `sim30to50_from_v725` run을 계속 진행하면서 GUI와 reward 추이를 더 본다.

## 22. V7-2 GUI best 기준 양발 무릎 사용 보상 강화 + sim-to-real 50-80% 재학습

### 22.1 GUI 관찰 기준

방금 GUI로 확인한 기준 파일:

```text
C:\tmp\v7_2\logs\rl_games\humanoid_v7_2_lateral_startup_guard_sim30to50_from_v725_obs117_pelvisBody719_net512_8192env_mb32768\2026-07-02_02-56-05\nn\humanoid_v7_2_lateral_startup_guard_sim30to50_from_v725_obs117_pelvisBody719_net512_8192env_mb32768.pth
```

GUI 관찰:

- 이전보다 보행 형태는 유지되지만, 갑자기 왼발 스윙 때 왼쪽 무릎 사용이 줄어드는 모습이 보였다.
- 오른발/왼발 어느 한쪽만 무릎을 쓰는 비대칭으로 굳으면 실제 하드웨어에서 발목 roll이나 골반 roll로 버티는 보상이 생길 수 있다.
- 따라서 GUI로 본 best를 이어받되, 좌우 대칭 무릎 사용 보상을 추가한다.

### 22.2 추가한 보상

추가 항목:

```text
knee_usage_score
weight = 0.045
```

조인트 매핑:

- 오른쪽 무릎 pitch: `176`
- 왼쪽 무릎 pitch: `196`

조건:

- 스윙 중인 다리 무릎은 약 `0.22 rad` 근처로 굽히면 보상.
- 지지 중인 다리 무릎은 약 `0.07 rad` 정도 살짝 굽힌 자세를 보상.
- 현재 스윙발 기준으로 좌우를 바꿔 계산한다.
- 오른발 스윙 때는 오른무릎이 주 보상, 왼무릎이 지지 보상.
- 왼발 스윙 때는 왼무릎이 주 보상, 오른무릎이 지지 보상.

의도:

- 왼발만 무릎을 안 쓰는 현상을 줄인다.
- 발목 roll/골반 roll로 버티는 대신 무릎 pitch를 쓰게 만든다.
- 실제 스마트서보 하드웨어에서 비교적 강한 pitch 계열 관절을 중심으로 걷게 유도한다.

가중치 정리:

- `knee_usage_score`를 `0.045`로 추가했다.
- 전체 reward weight 합이 1.0을 유지하도록 `heading_lock_score`를 `0.080 -> 0.075`로 조정했다.
- 최종 per-step reward는 기존처럼 `0.001 ~ 1.0`으로 clamp된다.

### 22.3 sim-to-real 50-80% 변경

이전 조건:

- sim-to-real 30~50%

이번 조건:

- sim-to-real 50~80%

변경값:

| 항목 | 이전 | 이번 |
|---|---:|---:|
| servo delay | 2~5 steps | 3~7 steps |
| IMU angular velocity noise std | 0.020 | 0.035 |
| IMU angular velocity bias std | 0.004 | 0.007 |
| projected gravity noise std | 0.020 | 0.035 |
| joint backlash std | 0.012 | 0.020 |
| joint backlash bias std | 0.006 | 0.010 |
| static friction randomization | 0.90~1.35 | 0.75~1.50 |
| dynamic friction randomization | 0.70~1.20 | 0.55~1.35 |
| restitution randomization | 0.00~0.04 | 0.00~0.06 |
| actuator stiffness scale | 0.75~1.25 | 0.60~1.40 |
| actuator damping scale | 0.70~1.30 | 0.55~1.45 |
| joint friction scale | 0.70~1.30 | 0.55~1.45 |
| armature scale | 0.80~1.20 | 0.70~1.30 |

### 22.4 새 학습 run

새 run 이름:

```text
humanoid_v7_2_knee_usage_sim50to80_from_sim30to50best_obs117_pelvisBody719_net512_8192env_mb32768
```

실행 CMD:

```text
C:\tmp\v7_2\RUN_v7_2_knee_usage_sim50to80_from_sim30to50best_headless_8192.cmd
```

고정 학습 설정:

- `num_envs = 8192`
- `horizon_length = 32`
- `minibatch_size = 32768`
- `max_iterations = 100000`
- observation = `117`
- action = `12`
- network = `117 -> 512 -> 256 -> 128 -> 64 -> 12`

기대 변화:

- 좌우 무릎 사용 비대칭이 줄어야 한다.
- 왼발 스윙 때도 왼무릎이 더 분명하게 굽혀져야 한다.
- sim-to-real 강도가 올라가므로 초반 reward가 일시적으로 떨어질 수 있다.

### 22.5 실행 직후 초기 로그

실행 시작:

```text
2026-07-02 17:10:59
```

초기 확인 시점:

```text
80 iter
```

초기 지표:

| 항목 | 값 |
|---|---:|
| `Episode/Episode_Reward/reward_humanoid_v7` | `0.493271` |
| `rewards/iter` | `3.965047` |
| `shaped_rewards/iter` | `4.853274` |
| `episode_lengths/iter` | `477.550720 / 480` |
| `fall_or_bad_pose` | `0.025208` |
| `time_out` | `0.974915` |
| `performance/step_fps` | 약 `73,854 fps` |

판단:

- sim-to-real을 50~80%까지 올렸지만 초기부터 episode length가 거의 끝까지 유지된다.
- fall 비율도 약 2.5%로, 시작 직후 기준으로는 안정적이다.
- reward는 30~50% run보다 낮게 시작하는 것이 정상이며, 무릎 사용 보상을 새로 넣었기 때문에 몇백 iter 동안 적응 구간이 필요하다.

### 22.6 2026-07-02 18:50 GUI best 확인

학습 상황 확인 후 GUI를 별도 1 env play로 실행했다.

확인 시점:

- 학습 epoch: 약 `1600`
- 최신 저장: `ep_1600_rew_3.9661303`
- `Episode/Episode_Reward/reward_humanoid_v7`: 약 `0.495`
- `rewards/iter`: 약 `3.99`
- episode length: `469~478 / 480`
- fall 비율: 약 `2.4%`

GUI 기준 checkpoint:

```text
C:\tmp\v7_2\logs\rl_games\humanoid_v7_2_knee_usage_sim50to80_from_sim30to50best_obs117_pelvisBody719_net512_8192env_mb32768\2026-07-02_17-10-59\nn\humanoid_v7_2_knee_usage_sim50to80_from_sim30to50best_obs117_pelvisBody719_net512_8192env_mb32768.pth
```

GUI 실행 파일:

```text
C:\tmp\v7_2\RUN_v7_2_knee_usage_sim50to80_best_gui_1env_play.cmd
C:\Users\hsh\OneDrive\바탕 화면\보고서용 정리\humanoid_v7\pth_실행파일\pth v7.2.10 knee_usage_sim50to80_best_GUI.cmd
```

실행 방식:

- headless 학습은 종료하지 않았다.
- GUI는 별도 `play.py --num_envs 1`로 실행했다.
- 기존 NVIDIA 단일 GPU 렌더링 우회 옵션을 그대로 사용했다.

### 22.7 v7.2.11 Y축 발 간격 + 골반 기울기 소폭 억제 + sim-to-real 80~100%

목표:

- v7.2.10에서 나온 best 정책을 이어받아 현실 불확실성을 더 강하게 적용한다.
- 기존에 이미 들어 있던 발 간격/스윙 전진 보상을 크게 새로 만들지 않고, 발 간격 판정축만 사용자가 지정한 방식으로 정리한다.
- 골반/몸체가 어느 축으로든 과하게 기울어지는 행동을 아주 약하게 더 억제한다.

기반 checkpoint:

```text
C:\tmp\v7_2\logs\rl_games\humanoid_v7_2_knee_usage_sim50to80_from_sim30to50best_obs117_pelvisBody719_net512_8192env_mb32768\2026-07-02_17-10-59\nn\humanoid_v7_2_knee_usage_sim50to80_from_sim30to50best_obs117_pelvisBody719_net512_8192env_mb32768.pth
```

새 run 이름:

```text
humanoid_v7_2_pelvis_soft_tilt_y_gap10to20_sim80to100_from_v7210best_obs117_pelvisBody719_net512_8192env_mb32768
```

보상체계 변경:

- 발 간격 보상은 더 이상 X축 차이를 보지 않는다.
- `left_foot_1`과 `right_foot_1`의 로봇 로컬 Y축 좌표 차이 절대값만 사용한다.
- Y축 발 간격 목표 범위는 `0.10 m ~ 0.20 m`이다.
- `0.10 m`보다 좁으면 감점, `0.20 m`보다 넓어도 감점이 들어간다.
- 기존 스윙/전진/교차 보상 구조는 유지했다.
- `upright_orientation_score` weight를 `0.055 -> 0.065`로 올려 골반/몸체 기울기 억제를 아주 약하게 강화했다.
- 전체 per-step reward는 계속 `0.001 ~ 1.0`으로 clamp된다.

sim-to-real 변경:

- rigid body material 랜덤화:
  - static friction: `(0.75, 1.50) -> (0.55, 1.80)`
  - dynamic friction: `(0.55, 1.35) -> (0.40, 1.60)`
  - restitution: `(0.0, 0.06) -> (0.0, 0.08)`
- actuator gain 랜덤화:
  - stiffness scale: `(0.60, 1.40) -> (0.50, 1.50)`
  - damping scale: `(0.55, 1.45) -> (0.50, 1.55)`
- joint parameter 랜덤화:
  - friction scale: `(0.55, 1.45) -> (0.45, 1.65)`
  - armature scale: `(0.70, 1.30) -> (0.60, 1.40)`

고정 유지:

- USD/URDF asset은 변경하지 않았다.
- observation은 `117`개 유지.
- action은 `12`개 유지.
- network는 `117 -> 512 -> 256 -> 128 -> 64 -> 12` 유지.
- `num_envs = 8192`, `horizon_length = 32`, `minibatch_size = 32768`, `max_iterations = 100000` 유지.

실행 파일:

```text
C:\tmp\v7_2\RUN_v7_2_pelvis_y_gap10to20_sim80to100_from_v7210best_headless_8192.cmd
```

### 22.8 보고서용 pth GUI 실행파일 정리

`.pth` 파일은 PyTorch checkpoint이므로 Windows에서 직접 더블클릭해 실행하는 파일이 아니다. 따라서 보고서 폴더 안에는 각 `.pth`를 Isaac Sim GUI playback으로 여는 Windows 실행용 `.cmd` 파일을 별도로 생성했다.

저장 위치:

```text
C:\Users\hsh\OneDrive\바탕 화면\보고서용 정리\humanoid_v7\pth_실행파일
```

실행 방식:

- `.pth` 원본은 그대로 보존한다.
- 사용자는 같은 폴더의 `pth ... GUI.cmd` 파일을 더블클릭한다.
- 실행 시 `play.py --num_envs 1`로 1마리만 GUI에 띄운다.
- 기존 headless 학습은 종료하지 않는다.
- NVIDIA 단일 GPU 렌더링 우회 옵션을 동일하게 사용한다.

생성된 GUI 실행파일:

```text
pth v7.2.1 GUI.cmd
pth v7.2.2 GUI.cmd
pth v7.2.3 GUI.cmd
pth v7.2.4 GUI.cmd
pth v7.2.5 right_foot_not_forward_best_GUI.cmd
pth v7.2.8 lateral_startup_guard_best_GUI.cmd
pth v7.2.9 sim30to50_best_GUI.cmd
pth v7.2.10 knee_usage_sim50to80_best_GUI.cmd
```

주의:

- 파일 이름이 `.pth`인 checkpoint는 모델 가중치 파일이다.
- GUI 확인은 `.cmd` 파일을 열어야 한다.
- 새 학습 run에서 best `.pth`가 새로 생성되면 같은 방식으로 `pth v7.2.xx ... GUI.cmd`를 추가한다.
- Windows CMD에서 한글 경로가 깨질 수 있으므로 `v7.2.1~v7.2.4.pth` 사본은 아래 안전 경로에도 복사했다.

```text
C:\tmp\v7_2\report_pth_checkpoints
```

- `pth v7.2.1 GUI.cmd`부터 `pth v7.2.4 GUI.cmd`까지는 위 `C:\tmp` 사본을 checkpoint로 사용한다.
- 모든 GUI 실행 `.cmd`는 checkpoint 존재 여부를 검증했다.

### 22.9 v7.2.12 double-stance Y-gap only 실행

목표:

- v7.2.10 sim-to-real 50~80% best checkpoint를 기반으로 다시 학습한다.
- 발 Y축 간격 보상이 스윙 중에도 계속 적용되어 발이 앞으로 나가지 못하고 두 발 간격을 유지하려는 문제를 줄인다.
- 발 간격 보상은 양발이 모두 지면에 닿은 double support 순간에만 적용한다.

기반 checkpoint:

```text
C:\tmp\v7_2\logs\rl_games\humanoid_v7_2_knee_usage_sim50to80_from_sim30to50best_obs117_pelvisBody719_net512_8192env_mb32768\2026-07-02_17-10-59\nn\humanoid_v7_2_knee_usage_sim50to80_from_sim30to50best_obs117_pelvisBody719_net512_8192env_mb32768.pth
```

새 run 이름:

```text
humanoid_v7_2_double_stance_y_gap10to20_sim80to100_from_v7210best_obs117_pelvisBody719_net512_8192env_mb32768
```

실행 파일:

```text
C:\tmp\v7_2\RUN_v7_2_double_stance_y_gap10to20_sim80to100_from_v7210best_headless_8192.cmd
```

핵심 보상 변경:

- 발 간격 보상은 로봇 로컬 Y축 거리 절대값만 사용한다.
- 목표 Y축 발 간격은 `0.10 m ~ 0.20 m`이다.
- X축 발 간격은 이 보상에서 사용하지 않는다.
- double support, 즉 왼발과 오른발이 동시에 접촉 중일 때만 `stance_width_score`를 적용한다.
- 스윙 중에는 Y축 발 간격 보상이 사실상 중립이 되어 스윙발이 앞으로 나갈 자유도를 확보한다.
- v7.2.11의 sim-to-real 80~100% 설정과 신경망 구조는 유지했다.

초기 로그:

```text
run start: 2026-07-03 20:16:35
checked iter: 38
```

| 항목 | 값 |
|---|---:|
| `Episode/Episode_Reward/reward_humanoid_v7` | `0.442441` |
| `rewards/iter` | `3.594623` |
| `shaped_rewards/iter` | `4.308528` |
| `episode_lengths/iter` | `468.266327 / 480` |
| `fall_or_bad_pose` | `0.123081` |
| `time_out` | `0.877285` |
| `performance/step_fps` | 약 `72,875 fps` |

판단:

- 초반에는 fall 비율이 높았지만, 38 iter까지 빠르게 감소하고 있다.
- episode length가 다시 468/480까지 올라왔으므로 학습 프로세스는 정상적으로 진행 중이다.
- sim-to-real 80~100% 조건이기 때문에 초기 reward가 v7.2.10보다 낮게 시작하는 것은 정상이다.

### 22.10 v7.2.13 / v7.2.14 double support 발 간격 보정

문제 관찰:

- GUI 확인 중 양발이 모두 지면에 닿은 double support 순간에 두 가지 착지 패턴이 번갈아 나타났다.
- 한 스텝은 두 발이 정상적으로 앞뒤로 벌어진 자세였지만, 다음 스텝에서는 두 발이 거의 같은 선상에 놓이는 자세가 반복되었다.
- 기존 v7.2.12는 double support에서 Y축 앞뒤 간격 `0.10 m ~ 0.20 m`을 보상했지만, weight가 작고 sigma가 커서 같은 선상 착지에 대한 점수 손실이 충분하지 않았다.

축 기준:

```text
X축 = 로봇 기준 좌우
Y축 = 로봇 기준 앞뒤
Z축 = 위아래
앞 방향 = -Y
```

v7.2.13 변경:

- double support 순간에만 X축 좌우 간격 보상을 추가했다.
- 목표 X축 좌우 간격은 `0.14 m ~ 0.26 m`이다.
- 이 보상은 사진처럼 양발이 좌우로 너무 붙거나 겹치는 착지를 막기 위한 항목이다.
- 정기구학 기반 골반 앞쏠림 감점도 실험적으로 준비했지만, 사용자 판단에 따라 실제 실행 버전에서는 제거했다.
- 기반 checkpoint는 v7.2.10 sim-to-real 50~80% best로 맞추어야 한다.

v7.2.14 변경:

- X축 좌우 간격 보상은 v7.2.13과 동일하게 유지했다.
- double support 순간의 Y축 앞뒤 간격 보상을 강화했다.
- 목표 Y축 앞뒤 간격은 그대로 `0.10 m ~ 0.20 m`로 유지했다.
- Y축 앞뒤 간격 전용 sigma를 새로 분리했다.

```text
double_support_forward_gap_sigma = 0.045
stance_width_score weight: 0.020 -> 0.055
double_support_side_gap_score weight: 0.030 유지
```

의도:

- 스윙 중에는 발 간격 보상을 강제하지 않는다.
- 양발이 모두 땅에 닿은 순간에만 앞뒤 간격과 좌우 간격을 검사한다.
- 같은 선상 착지는 낮은 점수를 받게 하고, 정상적인 앞뒤 step 착지는 더 높은 점수를 받게 한다.
- 골반 앞쏠림 감점은 이번 run에 넣지 않는다.

기반 checkpoint:

```text
C:\tmp\v7_2\logs\rl_games\humanoid_v7_2_knee_usage_sim50to80_from_sim30to50best_obs117_pelvisBody719_net512_8192env_mb32768\2026-07-02_17-10-59\nn\humanoid_v7_2_knee_usage_sim50to80_from_sim30to50best_obs117_pelvisBody719_net512_8192env_mb32768.pth
```

v7.2.14 run 이름:

```text
humanoid_v7_2_14_stronger_double_support_ygap_sim80to100_from_v7210best_obs117_pelvisBody719_net512_8192env_mb32768
```

실행 파일:

```text
C:\tmp\v7_2\RUN_v7_2_14_stronger_ygap_sim80to100_from_v7210best_headless_8192.cmd
```

학습 조건:

| 항목 | 값 |
|---|---:|
| 병렬 환경 수 | `8192` |
| horizon length | `32` |
| minibatch size | `32768` |
| max iterations | `100000` |
| 입력 | `117` |
| 신경망 | `117 -> 512 -> 256 -> 128 -> 64 -> 12` |
| sim-to-real | `80~100%` |

실행 상태:

```text
CMD PID: 35972
kit.exe PID: 17820
```

### 22.11 v7.2.15 FK 기반 앞발 순서 보상 추가

문제 관찰:

- v7.2.14는 double support 순간의 Y축 앞뒤 간격 보상을 강화했지만, 보행 중 한 스텝은 정상적으로 앞뒤가 벌어지고 다음 스텝은 두 발이 같은 선상에 가까워지는 패턴이 남아 있었다.
- 단순히 두 발의 앞뒤 거리만 보상하면, 어느 발이 앞에 있어야 하는지에 대한 좌우 교대 순서가 충분히 강제되지 않는다.

추가한 개념:

- 정기구학으로 계산 가능한 왼발/오른발의 로봇 기준 앞뒤 위치를 사용한다.
- IsaacLab 안에서는 `left_rel_b`, `right_rel_b`에서 로봇 root frame 기준 발 위치를 계산한다.
- 실제 ROS2 하드웨어에서는 스마트서보 각도, 링크 길이, 관절 배치를 이용한 정기구학으로 같은 값을 계산할 수 있다.

보상 적용 조건:

- 스윙 중에는 적용하지 않는다.
- 양발이 모두 지면에 닿은 double support 순간에만 적용한다.

보상 구조:

```text
lead_foot_gap_target = 0.08 m
lead_foot_gap_sigma = 0.04

lead_foot_order_score weight = +0.055
lead_foot_order_penalty weight = -0.035
```

동작:

- 왼발 차례에는 왼발이 오른발보다 앞에 있으면 보상한다.
- 오른발 차례에는 오른발이 왼발보다 앞에 있으면 보상한다.
- 기대한 발이 앞에 있지 않으면 `wrong_lead_penalty`를 준다.
- 두 발 앞뒤 차이가 너무 작아 같은 선상에 가까우면 `same_line_penalty`를 준다.
- 최종 reward는 기존처럼 `0.001 ~ 1.0` 범위로 clamp된다.

유지한 항목:

- double support Y축 앞뒤 간격 보상 `0.10 m ~ 0.20 m`, weight `0.055`
- double support X축 좌우 간격 보상 `0.14 m ~ 0.26 m`, weight `0.030`
- sim-to-real `80~100%`
- 입력 `117`
- 신경망 `117 -> 512 -> 256 -> 128 -> 64 -> 12`
- 병렬 환경 수 `8192`

기반 checkpoint:

```text
C:\tmp\v7_2\logs\rl_games\humanoid_v7_2_14_stronger_double_support_ygap_sim80to100_from_v7210best_obs117_pelvisBody719_net512_8192env_mb32768\2026-07-04_10-54-59\nn\humanoid_v7_2_14_stronger_double_support_ygap_sim80to100_from_v7210best_obs117_pelvisBody719_net512_8192env_mb32768.pth
```

v7.2.15 run 이름:

```text
humanoid_v7_2_15_fk_alternating_lead_foot_sim80to100_from_v7214best_obs117_pelvisBody719_net512_8192env_mb32768
```

실행 파일:

```text
C:\tmp\v7_2\RUN_v7_2_15_fk_alternating_lead_foot_sim80to100_from_v7214best_headless_8192.cmd
```

실행 상태:

```text
CMD PID: 10604
kit.exe PID: 32952
```

### 22.12 v7.2.17 발 중심축 기준 과보폭 감점 추가

기반 checkpoint:

```text
C:\tmp\v7_2\logs\rl_games\humanoid_v7_2_knee_usage_sim50to80_from_sim30to50best_obs117_pelvisBody719_net512_8192env_mb32768\2026-07-02_17-10-59\nn\humanoid_v7_2_knee_usage_sim50to80_from_sim30to50best_obs117_pelvisBody719_net512_8192env_mb32768.pth
```

기반 조건:

- v7.2.10 sim-to-real `50~80%` best 정책을 이어받는다.
- 새 학습 run에서는 sim-to-real 변수를 `80~100%`로 올린다.
- 보상체계는 기존 v7.2.10 계열 보상체계를 유지하고, 발 중심축 기준 과보폭 감점만 추가한다.

문제 관찰:

- GUI 확인 중 양발의 앞뒤 간격이 순간적으로 과하게 벌어지는 자세가 나타났다.
- 이러한 자세는 시뮬레이션에서는 앞으로 가는 행동처럼 보일 수 있지만, 실제 하드웨어에서는 골반 Pitch, 무릎 Pitch, 발목 관절에 큰 부담을 줄 수 있다.
- 특히 발 중심이 로봇 중심축에서 앞뒤로 너무 멀어지면 실제 로봇은 무게중심 회복이 어려워지고, 착지 충격과 서보모터 부하가 커진다.

추가한 보상/감점:

```text
max_foot_forward_from_center = 0.25 m
foot_forward_overreach_sigma = 0.05
foot_forward_overreach_penalty weight = -0.040
```

적용 방식:

- 발 위치는 발 링크 원점이 아니라 압력센서 네 모서리의 평균 위치, 즉 발바닥 중심점을 기준으로 계산한다.
- 왼발과 오른발 각각에 대해 로봇 중심축 기준 앞뒤 위치를 계산한다.
- 두 발 중 하나라도 중심축에서 앞뒤로 `25 cm` 이상 멀어지면 초과량에 비례해 감점을 준다.
- 감점은 hard termination이 아니라 soft penalty로 적용하여, 정상적인 보폭은 허용하되 과도한 보폭만 손해를 보도록 했다.

의도:

- 로봇이 앞으로 가기 위해 다리를 지나치게 길게 뻗는 행동을 억제한다.
- 실제 하드웨어에서 골반과 무릎에 걸리는 토크 부담을 줄인다.
- 보행이 `큰 보폭으로 버티는 자세`가 아니라, 좌우 발을 번갈아 내딛는 안정적인 보행 패턴으로 수렴하도록 유도한다.

### 22.13 v7.2.18 연속 보행 유도 보상 추가

기반 checkpoint:

```text
C:\tmp\v7_2\logs\rl_games\humanoid_v7_2_17_foot_forward_overreach_from_v7210best_sim80to100_obs117_pelvisBody719_net512_8192env_mb32768\2026-07-04_21-38-56\nn\humanoid_v7_2_17_foot_forward_overreach_from_v7210best_sim80to100_obs117_pelvisBody719_net512_8192env_mb32768.pth
```

문제 관찰:

- v7.2.17 GUI 확인 결과, 로봇이 한 발을 내딛은 뒤 두 발 지지 자세에서 잠깐 멈추고 다시 한 발을 내딛는 패턴을 보였다.
- 원인은 `double_support` 상태에서 발 간격, 발 순서, 안정 자세 보상을 여러 개 받을 수 있는 반면, 다음 발을 바로 이어서 내딛는 보상이 상대적으로 약했기 때문으로 판단했다.
- v7.2.17에서 추가한 `25 cm` 과보폭 감점은 필요하지만, 단독으로는 `너무 멀리 뻗지 말라`는 제약만 강해져서 보수적으로 멈추는 전략을 강화할 수 있다.

추가/수정한 항목:

```text
step_continuation_score weight = +0.060
alternating_gait_score weight = +0.075 -> +0.095
cross_forward_step_score weight = +0.055 -> +0.075
alternating_forward_step_score weight = +0.060 -> +0.095
single_support_score weight = +0.040 -> +0.070

stance_width_score weight = +0.055 -> +0.035
lead_foot_order_score weight = +0.055 -> +0.040

double_support_hold_penalty weight = -0.070
stall_penalty weight = -0.050
target_forward_step_length = 0.06 m -> 0.12 m
lead_foot_gap_target = 0.08 m -> 0.10 m
```

`step_continuation_score` 정의:

- 현재 gait phase에서 스윙해야 하는 발이 실제로 지면에서 떨어진다.
- 스윙발이 로봇 기준 앞 방향으로 속도를 낸다.
- 몸체가 목표 진행 방향으로 계속 전진한다.
- heading lock이 유지된다.

의도:

- `한 스텝 후 멈춤` 전략보다 `다음 발을 바로 이어서 내딛는` 전략이 더 높은 점수를 받게 한다.
- 두 발 지지 자세는 착지 안정용으로만 짧게 허용하고, 오래 유지하면 감점한다.
- 보폭 목표를 12 cm로 올려 제자리에서 짧게 까딱이는 보행을 줄인다.
- 과보폭 감점은 유지하되, 그 대신 연속 스윙 보상을 키워 `과하게 뻗지 않으면서도 계속 걷는` 방향으로 유도한다.

### 22.14 v7.2.23 접촉 기반 좌우 보행 전환 보상 추가

기반 checkpoint:

```text
C:\tmp\v7_2\logs\rl_games\humanoid_v7_2_22_balanced_limp_penalty_from_v7221best_sim10to20_obs117_pelvisBody719_net512_8192env_mb32768\2026-07-26_14-01-35\nn\humanoid_v7_2_22_balanced_limp_penalty_from_v7221best_sim10to20_obs117_pelvisBody719_net512_8192env_mb32768.pth
```

문제 관찰:

- v7.2.21과 v7.2.22에서 한쪽 발 절름발이 현상을 줄이기 위해 좌우 대칭 스윙 보상과 좌우 EMA 균형 보상을 추가하였다.
- 그러나 보상 하나를 조정할 때마다 왼발 절름발이가 오른발 절름발이로, 또는 반대로 옮겨가는 현상이 반복되었다.
- 실제 관찰 순서는 다음과 같았다.
  1. 처음에는 오른발이 충분히 앞으로 나가지 못하고 절름발이처럼 따라오는 문제가 나타났다.
  2. 오른발 전진/착지 보상을 조정한 뒤에는 오른발 문제는 완화되었지만, 반대로 왼발이 충분히 스윙하지 못하는 문제가 나타났다.
  3. 왼발 쪽 문제를 다시 보정하자 이번에는 다시 오른발 절름발이 현상이 재발하였다.
  4. 따라서 단순히 특정 발의 보상을 키우는 방식은 문제를 반대쪽 발로 옮길 뿐, 좌우 보행 리듬 자체를 안정화하지 못한다고 판단하였다.
- 원인은 `sin(phase)` 기반으로 “지금은 왼발 차례/오른발 차례”를 강하게 정해 둔 보상과 실제 로봇의 접촉 리듬이 서로 어긋나기 때문으로 판단하였다.
- 실제 하드웨어에서는 ROS2에서 phase 시간표 자체를 정확히 재현하기보다, 압력센서 접촉 상태와 IMU 자세를 기준으로 현재 어느 발이 지지발/스윙발인지 판단하는 편이 더 자연스럽다.

추가/수정한 항목:

```text
actual_single_support_score
  = 실제 접촉력 기준으로 한 발만 지면에 닿아 있는지 평가

swing_alternation_score weight = +0.055
  = 실제 스윙 이벤트가 발생했을 때 직전 스윙발과 반대발이면 보상

same_foot_repeat_penalty weight = -0.075
  = 같은 발이 연속으로 스윙 이벤트를 만들면 감점

balanced_limp_score weight = +0.070 -> +0.080
  = 왼발/오른발 스윙 활동량과 전진 스윙량의 장기 균형 보상 강화

knee_usage_score
  = phase 기준 무릎 보상에서 실제 공중발 기준 무릎 보상으로 변경

lead_foot_order_penalty
  = 특정 phase에서 반드시 특정 발이 앞서야 한다는 감점 제거
  = 두 발이 같은 앞뒤 선상에 놓이는 경우만 감점
```

의도:

- “정해진 시간표대로 왼발/오른발을 들어라”가 아니라, 실제 접촉 상태를 기준으로 보행 리듬을 판단하게 한다.
- 한쪽 발만 계속 앞으로 보내는 절름발이 전략을 줄이고, 실제로 한 발을 내딛은 다음에는 반대발이 이어서 내딛는 행동을 더 높은 보상으로 만든다.
- 무릎 사용 보상도 실제 공중에 있는 발 기준으로 계산하여, phase가 틀어졌을 때 반대쪽 무릎에 보상이 들어가는 문제를 줄인다.
- ROS2 실제 구동 시에도 발바닥 압력센서 8개로 접촉 상태를 판별할 수 있으므로, sim-to-real 관점에서 더 재현 가능한 보상 구조로 바꾼다.

### 22.15 v7.2.24 sim-to-real 20~40% 강화

기반 checkpoint:

```text
C:\tmp\v7_2\logs\rl_games\humanoid_v7_2_23_contact_driven_gait_from_v7222best_sim10to20_obs117_pelvisBody719_net512_8192env_mb32768\2026-07-26_16-26-37\nn\humanoid_v7_2_23_contact_driven_gait_from_v7222best_sim10to20_obs117_pelvisBody719_net512_8192env_mb32768.pth
```

변경 이유:

- v7.2.23에서 접촉 기반 좌우 교대 보상으로 절름발이 문제가 완화되고, 학습 reward가 안정적으로 올라갔다.
- 따라서 보상체계와 신경망 구조는 유지한 채, 실제 하드웨어와의 차이를 줄이기 위해 sim-to-real 변수만 `10~20%`에서 `20~40%` 수준으로 올렸다.
- 이 단계는 성공 정책을 더 거친 현실 오차 조건에 적응시키는 중간 강화 단계이다.

유지한 항목:

```text
reward = v7.2.23 contact-driven gait reward 유지
observation size = 117
network = 117 -> 512 -> 256 -> 128 -> 64 -> 12
num_envs = 8192
horizon_length = 32
minibatch_size = 32768
max_epochs = 100000
```

sim-to-real 증가 항목:

```text
IMU angular velocity noise:
  std 0.010 -> 0.020
  bias_std 0.002 -> 0.004

IMU projected gravity noise:
  std 0.010 -> 0.020

joint backlash / encoder residual noise:
  backlash_std 0.006 -> 0.012
  backlash_bias_std 0.003 -> 0.006

servo delay:
  min_delay 1 -> 2 steps
  max_delay 3 -> 5 steps

robot material friction randomization:
  static_friction_range 1.02~1.20 -> 0.90~1.35
  dynamic_friction_range 0.82~1.05 -> 0.70~1.20
  restitution_range 0.00~0.02 -> 0.00~0.04

actuator gain randomization:
  stiffness scale 0.90~1.10 -> 0.80~1.20
  damping scale 0.85~1.15 -> 0.70~1.30

joint friction / armature randomization:
  friction scale 0.85~1.15 -> 0.70~1.30
  armature scale 0.90~1.10 -> 0.80~1.20

reset disturbance:
  base xy/yaw offset and initial velocity variation increased by about 2x
  joint initial position/velocity offset increased by about 2x
```

초기 확인 결과:

```text
step 20:
  rewards/iter = 1.0968
  reward_humanoid_v7 = 0.1338
  fall_or_bad_pose = 0.9990

step 50:
  rewards/iter = 4.6000
  reward_humanoid_v7 = 0.5706
  fall_or_bad_pose = 0.0173
  episode_length = 478.28
```

해석:

- sim-to-real 조건을 올린 직후에는 기존 정책이 크게 흔들려 fall 비율이 높게 시작했다.
- 그러나 50 iter에서 reward와 episode length가 빠르게 회복되었고, fall 비율도 크게 낮아졌다.
- 따라서 v7.2.24는 너무 강하게 망가진 run이 아니라, 더 현실적인 오차 조건에 적응하기 시작한 run으로 판단하였다.

### 22.16 v7.2.25 sim-to-real 40~60% 강화

기반 checkpoint:

```text
C:\tmp\v7_2\logs\rl_games\humanoid_v7_2_24_sim20to40_from_v7223best_obs117_pelvisBody719_net512_8192env_mb32768\2026-07-27_03-09-25\nn\humanoid_v7_2_24_sim20to40_from_v7223best_obs117_pelvisBody719_net512_8192env_mb32768.pth
```

변경 이유:

- v7.2.24에서 sim-to-real 20~40% 조건을 적용해도 reward가 안정적으로 회복되었다.
- 따라서 같은 보상체계와 같은 신경망을 유지한 상태에서, 실제 하드웨어 오차에 더 가까운 `40~60%` 조건으로 한 단계 더 올렸다.
- 이 단계는 바로 실제 투입용 최종 단계라기보다, 80~100% sim-to-real 강화학습으로 가기 전 중간 적응 단계이다.

유지한 항목:

```text
reward = v7.2.23 contact-driven gait reward 유지
observation size = 117
network = 117 -> 512 -> 256 -> 128 -> 64 -> 12
num_envs = 8192
horizon_length = 32
minibatch_size = 32768
max_epochs = 100000
```

sim-to-real 증가 항목:

```text
IMU angular velocity noise:
  std 0.020 -> 0.030
  bias_std 0.004 -> 0.006

IMU projected gravity noise:
  std 0.020 -> 0.030

joint backlash / encoder residual noise:
  backlash_std 0.012 -> 0.018
  backlash_bias_std 0.006 -> 0.009

servo delay:
  min_delay 2 -> 3 steps
  max_delay 5 -> 7 steps

robot material friction randomization:
  static_friction_range 0.90~1.35 -> 0.75~1.50
  dynamic_friction_range 0.70~1.20 -> 0.55~1.35
  restitution_range 0.00~0.04 -> 0.00~0.06

actuator gain randomization:
  stiffness scale 0.80~1.20 -> 0.65~1.35
  damping scale 0.70~1.30 -> 0.55~1.45

joint friction / armature randomization:
  friction scale 0.70~1.30 -> 0.55~1.45
  armature scale 0.80~1.20 -> 0.65~1.35

reset disturbance:
  base xy/yaw offset and initial velocity variation increased again
  joint initial position/velocity offset increased again
```

초기 확인 결과:

```text
step 5:
  rewards/iter = 0.2137
  reward_humanoid_v7 = 0.0243
  fall_or_bad_pose = 0.9774
  episode_length = 73.36

step 35:
  rewards/iter = 3.3570
  reward_humanoid_v7 = 0.4274
  fall_or_bad_pose = 0.5596
  episode_length = 464.37

step 55:
  rewards/iter = 4.2497
  reward_humanoid_v7 = 0.5319
  fall_or_bad_pose = 0.0596
  episode_length = 461.27
```

해석:

- 40~60% 조건에서는 시작 직후 정책이 크게 흔들렸고, step 5에서 fall 비율이 매우 높게 나타났다.
- 그러나 step 35에서 episode length가 회복되기 시작했고, step 55에서는 fall 비율이 크게 내려갔다.
- 따라서 v7.2.25는 난이도가 확실히 올라간 run이지만, 기존 정책이 적응할 수 있는 범위 안에 있는 것으로 판단하였다.


