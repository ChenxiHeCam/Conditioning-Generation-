# Agent Handoff Document — 21cm Latent Diffusion Generator

> 给接手的 agent 看的完整上下文文档。读完这个文件就能无缝继续工作。
> 最后更新：2026-04-15

---

## 用户信息

- **姓名：** Chenxi He（贺晨曦）
- **身份：** IoA Cambridge 新博士生，刚加入 ASR21cm 项目
- **导师：** Anastasia Fialkov（FIALKOV，IoA Cambridge，21cm 宇宙学）
- **团队：** Simon Pochinda（lead）、Peter Sims、Jiten Dhandha（数据）
- **语言偏好：** 中文回复

---

## 项目目标

为 SKA 望远镜构建 21cm 亮温场（T21）的生成式模拟器：

```
给定：初始条件场 (δ, v_bv) + 天体物理参数 (fstar, Vc, fX, delay, z)
生成：T21 三维亮温场 patch (64³ Mpc)
```

替代 21cmFAST 全物理模拟（耗时数小时），生成时间约 1 秒。

---

## 代码位置

### 本地（Windows）
```
D:\Astro\
├── models/
│   ├── vae.py              # VAE3D
│   └── flow_matching.py    # FlowUNet3D + ICEncoder + ICCrossAttn3D + ConditionalFlowMatcher
├── utils/
│   └── power_spectrum.py   # 评估用物理功率谱
├── dataset.py              # T21Dataset（自动识别 varying_IC / varying_astro）
├── train_vae.py            # VAE 训练
├── train_flow.py           # Flow 训练
├── evaluate_vae.py         # VAE 评估
├── evaluate_flow.py        # Flow 评估
├── plot_loss.py            # 训练曲线
├── submit_vae.sh           # CSD3 SLURM 脚本
├── submit_flow.sh          # CSD3 SLURM 脚本
├── submit_reflow.sh        # CSD3 reflow 精调脚本
├── README.md               # 技术文档
├── future_improvements.md  # 改进方向 + 论文
├── HANDOFF.md              # 本文件
└── conversation_history.jsonl  # 完整对话记录
```

### CSD3（HPC 训练）
```
/home/ch2067/rds/hpc-work/21cm_gen/
├── models/           # 同上（已上传）
├── utils/
├── dataset.py
├── train_vae.py
├── train_flow.py
├── submit_vae.sh
├── submit_flow.sh
├── checkpoints/
│   ├── vae/          # VAE checkpoints（vae_epoch0299.pt 目标）
│   └── flow/         # Flow checkpoints
└── logs/             # SLURM stdout/stderr
```

### 数据（CSD3）
```
/home/ch2067/rds/hpc-work/ASR21cm/datasets/
├── varying_IC/           # 主训练集（已传完）
│   ├── T21_cubes/        # T21_cube_z10__Npix256_IC{N}.mat  (~3609 个)
│   └── IC_cubes/         # delta_Npix256_IC{N}.mat + vbv_Npix256_IC{N}.mat  (204 对)
└── varying_astro/        # 备用（固定 IC，变参数）
    ├── T21_cubes/
    ├── IC_cubes/         # delta1000.mat + vbv1000.mat（共享）
    └── parameters/       # *.params 文件
```

---

## SSH 连接方法

### CSD3
```python
# 需要 keyboard-interactive（密码 + TOTP），不支持 SSH key 直接认证
# 用 paramiko Transport 方式（每次需要新的 TOTP）

import paramiko

password = 'hcxhaoshuaI123456/'
totp = '??????'   # 向用户要新的 TOTP（30秒有效）

responses = [password, totp]
idx = [0]

def handler(title, instructions, prompts):
    ans = []
    for p, echo in prompts:
        if idx[0] < len(responses):
            ans.append(responses[idx[0]])
            idx[0] += 1
        else:
            ans.append('')
    return ans

transport = paramiko.Transport(('login-cpu.hpc.cam.ac.uk', 22))
transport.connect()
transport.auth_interactive('ch2067', handler)

client = paramiko.SSHClient()
client._transport = transport

# 一次连接内完成所有操作，最后 transport.close()
```

**重要：** 一次连接内完成所有操作（上传文件 + 执行命令），避免多次要 TOTP。

### highz（IoA 集群）
```bash
# ~/.ssh/config 已配置，但 key 有 passphrase，需用户提供
Host highz
    HostName cap001a.ast.cam.ac.uk
    User ch2067
    IdentityFile ~/.ssh/highz_key
```

---

## 当前 SLURM 任务状态（2026-04-15）

| Job ID | 任务 | 分区 | 状态 | 说明 |
|--------|------|------|------|------|
| 27669867 | VAE 训练 300 epochs | ampere | Pending | 等待资源 |
| 27736175 | Flow 训练 500 epochs | ampere | Pending | 等 VAE 完成（实际是独立提交，不是 dependency） |

账号：`FIALKOV-SL3-GPU`（导师 Fialkov 名下免费额度，低优先级）

提升优先级的方法：让导师通过 SAFE 门户申请 SL2 GPU 小时。

---

## 模型架构

### 阶段一：VAE3D
```
输入: (B, 1, 64, 64, 64) T21 patch
  → Encoder: ResBlock×2 + 步幅卷积 × 2（64→32→16）
  → 潜向量 z: (B, 4, 16, 16, 16)  [重参数化采样]
  → Decoder: 转置卷积 × 2（16→32→64）+ ResBlock×2
输出: (B, 1, 64, 64, 64) 重建
损失: MSE + β×KL  (β=1e-4)
```

### 阶段二：条件流匹配（CFM）
```
给定: z1（真实潜向量，VAE编码）
      z0 ~ N(0,1)
      t ~ logit-normal（集中在 t≈0.5）

训练目标:  v_θ(z_t, t, IC, params) ≈ z1 - z0
           其中 z_t = (1-t)·z0 + t·z1

条件注入:
  - 时间 t:      正弦嵌入 → AdaGN（所有 ResBlock）
  - 参数 params: Fourier特征 → MLP → AdaGN（所有 ResBlock）
  - IC 场:       ICEncoder(delta, vbv) → (B,4,16,16,16)
                 → ICCrossAttn3D 注入每层 decoder（Q=latent, K/V=IC）

物理损失:  L_ps = MSE(log Δ²(k_pred), log Δ²(k_true))  权重0.01

采样（100步 Euler）:
  z_0 ~ N(0,1) → ... → z_1 → VAE decode → T21
  CFG: v = v_uncond + 3.0 × (v_cond - v_uncond)
```

### 为什么用 IC 交叉注意力而非 concat？
concat 只在输入层融合 IC 信息，多层卷积后空间对应关系退化。  
cross-attention 让 decoder 每层每个位置直接查询 IC 场，保留物理对应关系。  
K/V 分辨率上限 8³=512 tokens，控制计算量。

### 参数维度（param_dim=5）
```python
PARAM_KEYS = ['MyStar_II', 'MyVc', 'MyFX', 'DelayParam', 'redshift']
LOG_PARAMS  = {'MyStar_II', 'MyFX'}   # 先取 log10 再归一化

# varying_IC 固定参数（从 MATLAB 脚本默认值）
VARYING_IC_FIXED_PARAMS = {'MyStar_II': 0.05, 'MyVc': 4.2, 'MyFX': 1.0, 'DelayParam': 0.75}
```

---

## 数据集格式（自动检测）

```python
# dataset.py 自动识别：
# 文件名含 _IC\d+\.mat → varying_IC 格式
# 文件名含 diffusion_  → varying_astro 格式

# varying_IC IC 文件对应关系（按 IC 编号）：
# T21_cube_z10__Npix256_IC42.mat ←→ delta_Npix256_IC42.mat + vbv_Npix256_IC42.mat

# IC LRU 缓存：最多同时载入 60 个 IC（OrderedDict）
# 参数归一化：自动计算 mean/std
# T21 归一化：从前 50 个文件采样
```

---

## 下一步任务（按优先级）

### 立即（等 CSD3 任务跑完）
1. **检查 VAE 任务**：`squeue -u ch2067` → 等 27669867 完成
2. **评估 VAE**：
   ```bash
   python evaluate_vae.py --ckpt checkpoints/vae/vae_epoch0299.pt
   # 目标：相对MSE < 0.05，PS ratio 0.95~1.05
   ```
3. **等 Flow 任务完成**（27736175）
4. **评估 Flow**：
   ```bash
   python evaluate_flow.py --flow_ckpt checkpoints/flow/flow_epoch0499.pt \
                            --vae_ckpt  checkpoints/vae/vae_epoch0299.pt
   ```
5. **Reflow 精调**（Flow 收敛后）：
   ```bash
   sbatch submit_reflow.sh
   ```

### 近期代码改进
6. **加 EMA**（train_flow.py 里加 torch_ema 或手动实现）
7. **图像空间 PS 损失**：`--ps_image` flag 已实现，用于精调
8. **多红移**：传其他红移数据到 CSD3，加 `--redshifts 7 8 9 10`

### 代码上传到 CSD3
```python
# 每次更新代码后，用 paramiko SFTP 上传
files_to_upload = [
    'dataset.py', 'train_flow.py', 'train_vae.py',
    'evaluate_vae.py', 'evaluate_flow.py', 'plot_loss.py',
    'submit_flow.sh', 'models/flow_matching.py',
    'models/vae.py', 'utils/power_spectrum.py',
]
remote_base = '/home/ch2067/rds/hpc-work/21cm_gen'
```

---

## 关键论文

| arXiv | 标题 | 优先级 |
|-------|------|--------|
| [2507.11842](https://arxiv.org/abs/2507.11842) | CosmoFlow — 最接近本项目的架构 | ⭐⭐⭐ |
| [2502.17087](https://arxiv.org/html/2502.17087v1) | Diffusion vs FM for 3D density fields | ⭐⭐⭐ |
| CVPR 2024 EDM2 | 训练稳定性改进（EMA + 超球面约束） | ⭐⭐ |
| NeurIPS 2024 Autoguidance | 比 CFG 更好的引导方式 | ⭐⭐ |
| [2505.18825](https://arxiv.org/abs/2505.18825) | Flow Maps 蒸馏（1-4步推理） | ⭐ |

完整改进列表见 `future_improvements.md`。

---

## 已知问题 / 踩过的坑

1. **CSD3 TOTP**：`~/.ssh/authorized_keys` 写了 SSH 公钥但 CSD3 不接受，仍需 TOTP。每次连接向用户要新的 TOTP，立刻运行，不要拖延（30秒过期）。

2. **highz SSH key 有 passphrase**：`~/.ssh/highz_key` 加密，需用户提供 passphrase 才能通过 paramiko 连接。

3. **varying_IC 只传了 z=10**：其他红移（z=7,8,9,12...）的 T21 数据还在 highz，需要时再传。

4. **submit_flow.sh 中 `--param_dim 5`**：varying_IC 数据固定参数，param_dim=5（含 redshift）。如果加入 varying_astro 数据需确认参数维度一致。

5. **Flow job 27736175 不是 dependency**：独立提交的，VAE 跑完后需手动检查 VAE checkpoint 路径是否匹配。

---

## 用户偏好

- 中文回复
- 不用总结"我刚才做了什么"（直接看 diff 就知道）
- 一次连接完成所有 CSD3 操作，不要多次要 TOTP
- 不需要解释基础概念，PhD 学生，直接说结论
