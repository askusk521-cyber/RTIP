# RTIP/JAX 全部判据与判定条件汇总


---

## 一、配置默认参数

**来源：** `rtipmd/jax/src/rtip_jax/config.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `a0` | 0.002 Hartree | RTIP 偏置基础振幅 |
| `max_step` | 1500 | 最大 MD 步数 |
| `dt` | 0.5 fs | 积分时间步长 |
| `temp_bath` | 1000.0 K | 恒温器目标温度 |
| `tau` | 500.0 fs | 恒温器耦合时间常数 |

---

## 二、MD 模拟的停止和偏置关闭条件

**来源：** `rtipmd/jax/src/rtip_jax/workflows/md.py`

### 2.1 温度保护（所有模式通用）

| 条件 | 动作 |
|------|------|
| 瞬时温度 > 1500 K | 偏置势和偏置力按 `1500 / T` 缩放 |

### 2.2 Attractive 模式（吸引偏置，拉向目标）

| 条件 | 阈值 | 动作 |
|------|------|------|
| `sigma_min`（RTI 距离）< 1.0 Bohr | < 1.0 | 关闭偏置 |
| 偏置力 > 1000 Hartree/Bohr | > 1000 | 关闭偏置 |
| RMS 真实力 < `f_epsilon`（偏置已关）| < 0.001 Ha/Bohr | **停止 MD，收敛** |
| step == max_step | 1500 | **达到最大步数，停止** |

### 2.3 Repulsive / Synthesis 模式（排斥偏置，推离起点）

| 条件 | 阈值 | 动作 |
|------|------|------|
| 真实势能下降 > 0.02 Hartree（相对历史最高值） | 下降 > 0.02 Ha | 关闭偏置 |
| 真实势能爬升 > 0.185 Hartree（相对历史最低值） | 爬升 > 0.185 Ha (~115 kcal/mol) | **停止 MD** |
| 偏置力 > 1000 Hartree/Bohr | > 1000 | **停止 MD** |
| RMS 真实力 < 0.001 Ha/Bohr（偏置已关） | < 0.001 | **停止 MD，收敛** |

**注意：** `stop_pot_climb` 的优先级高于 `stop_large_bias_force`。

### 2.4 所有 state_decision 值

| 值 | 含义 |
|----|------|
| `md_running` | MD 正常运行中 |
| `bias_off_target_reached` | 吸引模式：到达目标，偏置已关 |
| `bias_off_large_bias_force` | 偏置力过大，偏置已关 |
| `bias_off_after_pot_drop` | 排斥模式：势能下降，偏置已关 |
| `bias_off_after_synthesis_drop` | 合成模式：势能下降，偏置已关 |
| `stop_converged` | **成功收敛（偏置已关 + 力收敛）** |
| `stop_pot_climb` | 势能爬升过大，停止 |
| `stop_large_bias_force` | 偏置力过大，停止 |
| `max_step` | 达到最大步数，停止 |

---

## 三、Pathway Sampling 收敛条件

**来源：** `rtipmd/jax/src/rtip_jax/workflows/pathway_sampling.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 线搜索收敛容差 | `pot_epsilon × natom` = **0.00005 × 原子数** Hartree | 一维线搜索收敛判据 |
| 线搜索初始步长 | 0.01 Bohr | 黄金分割搜索起始步长 |
| 最小步长保护 | > 1e-15 Bohr | 防止无限搜索 |
| 搜索上界 | < 0.1 Bohr | 黄金分割搜索边界 |

停止逻辑与 MD 一致，区别是用 `running` / `bias_on` / `bias_off` 替代了 `md_running`。

---

## 四、"Promising" 成键判据（研究分析用）

**来源：** `research/ic5c02384/scripts/rcmd_lib.py`

### 4.1 Promising 判定

两个条件**必须同时满足**：

| 条件 | 阈值 | 含义 |
|------|------|------|
| `min_N_C < 2.0 Å` | < 2.0 Å | N2 上的 N 到小分子上的 C 进入成键范围 |
| `min_B_X < 2.0 Å` | < 2.0 Å | B 到 X（O/S/N 等）进入成键范围 |

```text
promising = (min_N_C < 2.0) AND (min_B_X < 2.0)
```

### 4.2 最佳成键帧选择

选择 `bond_score = min_N_C + min_B_O` **最小**的那一帧作为最佳成键帧。

### 4.3 反应参考距离的确定

- N-C 目标距离 = TS/产物参考结构中 N（N2 上）到 C（小分子上）的**最近原子对距离**
- B-X 目标距离 = TS/产物参考结构中 B 到 X（小分子的 O/S/N 等）的**最近原子对距离**
- N2 配对检测 = 所有 N-N 对中距离最短的一对（N≡N 三键 ~1.2 Å，显著短于其他 N-N 距离）

---

## 五、数值阈值总表

| 阈值 | 数值 | 单位 | 用途 |
|------|------|------|------|
| `pot_climb` | 0.185 | Hartree (~115 kcal/mol) | 势能爬升上限，超则停止 |
| `pot_drop` | 0.02 | Hartree (~12.6 kcal/mol) | 势能下降，超则关偏置 |
| `pot_epsilon × natom` | 0.00005 × N | Hartree | 线搜索收敛容差 |
| `f_epsilon` | 0.001 | Hartree/Bohr | RMS 力收敛标准 |
| `sigma_min` 目标 | 1.0 | Bohr (~0.53 Å) | RTI 距离：到达目标判定 |
| `f_bias_max` | 1000.0 | Hartree/Bohr | 偏置力上限，超则停止 |
| `T_max` | 1500.0 | K | 温度上限，超则缩放偏置 |
| `T_floor` | 1.0 | K | 恒温器温度下限保护 |
| `bond_threshold` | 2.0 | Å | 成键 promising 判定 |
| `dt` | 0.5 | fs | MD 时间步长 |
| `temp_bath` | 1000.0 | K | 恒温器目标温度 |
| `tau` | 500.0 | fs | 恒温器耦合时间 |

---

## 六、判据关系图

```text
MD 每步流程:
  ┌─ 位置更新（leapfrog 半步）
  ├─ 恒温器调温（目标 1000K）
  ├─ 算真实力场（DeePMD）
  ├─ 算偏置力场（RTIP）  ←── 若 T > 1500K，偏置缩放
  ├─ 合力 = 真实 + 偏置
  ├─ 速度更新（leapfrog 半步 + Berendsen λ）
  └─ 判定:
       ├─ pot_real 下降 > 0.02 Ha？ → 关偏置
       ├─ pot_real 爬升 > 0.185 Ha？ → 停止（stop_pot_climb）
       ├─ 偏置力 > 1000？ → 停止（stop_large_bias_force）
       ├─ 偏置已关 + RMS力 < 0.001？ → 停止（stop_converged）✅
       └─ step == max_step？ → 停止（max_step）

事后分析:
  读 .out 文件 → 每帧计算 min_N_C, min_B_X
  → promising = (min_N_C < 2.0 Å) AND (min_B_X < 2.0 Å)
  → 最佳帧 = bond_score 最小的帧
```
