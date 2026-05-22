# Code Walkthrough — 21cm Latent Diffusion Generator

> 完整代码走读文档，覆盖所有源文件。
> 最后更新：2026-04-22

---

## 目录

1. [项目概览](#1-项目概览)
2. [文件结构](#2-文件结构)
3. [数据层 — dataset.py](#3-数据层--datasetpy)
4. [模型层 — VAE](#4-模型层--vae)
5. [模型层 — Flow Matching](#5-模型层--flow-matching)
6. [训练脚本](#6-训练脚本)
7. [评估脚本](#7-评估脚本)
8. [工具函数](#8-工具函数)
9. [SLURM 提交脚本](#9-slurm-提交脚本)
10. [依赖关系图](#10-依赖关系图)
11. [关键数学公式](#11-关键数学公式)
12. [完整运行流程](#12-完整运行流程)

---

## 1. 项目概览

**目标：** 为 SKA 望远镜构建 21cm 亮温场（T21）的生成式模拟器，替代 21cmFAST 全物理模拟（数小时 → ~1秒）。

**输入输出：**
```
输入: 初始条件场 (δ, v_bv) + 天体物理参数 (fstar, Vc, fX, delay, z)
输出: T21 三维亮温场 64³ patch
```

**两阶段架构：**
```
阶段一 VAE3D:
  T21 (B,1,64,64,64) ──Encoder──→ z (B,4,16,16,16) ──Decoder──→ T21_recon

阶段二 条件流匹配 (CFM):
  z₀ ~ N(0,1) ──100步 Euler ODE──→ z₁ ──VAE Decode──→ T21_generated
                    ↑ 条件注入
          时间 t (AdaGN) + 参数 params (Fourier→AdaGN) + IC 场 (交叉注意力)
```

---

## 2. 文件结构

```
D:\Astro\
├── models/
│   ├── vae.py                 # VAE3D 编解码器（124行）
│   └── flow_matching.py       # FlowUNet3D + ICEncoder + CFM（510行）
├── utils/
│   └── power_spectrum.py      # 球平均功率谱 Δ²(k)（65行）
├── dataset.py                 # T21Dataset 数据加载（365行）
├── train_vae.py               # VAE 训练循环（141行）
├── train_flow.py              # Flow 训练循环（206行）
├── evaluate_vae.py            # VAE 评估（165行）
├── evaluate_flow.py           # Flow 评估（253行）
├── evaluate.py                # 联合评估（182行）
├── plot_loss.py               # 训练曲线可视化（59行）
├── submit_vae.sh              # SLURM VAE 提交
├── submit_flow.sh             # SLURM Flow 提交
├── submit_reflow.sh           # SLURM Reflow 精调提交
└── submit_eval.sh             # SLURM 评估提交
```

---

## 3. 数据层 — dataset.py

### 3.1 两种数据格式

| 格式 | 特点 | T21 文件名 | IC | 参数 |
|------|------|-----------|-----|------|
| `varying_IC` | 变初始条件，固定天体物理参数 | `T21_cube_z10__Npix256_IC42.mat` | 每个 IC 独立：`delta_Npix256_IC42.mat` | 固定值（见下） |
| `varying_astro` | 固定 IC，变天体物理参数 | `T21_cube_z10__diffusion_0001.mat` | 共享：`delta1000.mat` | `.params` 文件 |

**自动格式检测**（`_detect_format`, L158-165）：扫描文件名，含 `_IC\d+\.mat` → varying_IC，含 `diffusion_` → varying_astro。

### 3.2 参数规范

```python
PARAM_KEYS = ['MyStar_II', 'MyVc', 'MyFX', 'DelayParam', 'redshift']  # L42
LOG_PARAMS = {'MyStar_II', 'MyFX'}   # 先取 log10 再归一化              # L46

# varying_IC 固定参数（来自 MATLAB 脚本默认值）
VARYING_IC_FIXED_PARAMS = {
    'MyStar_II': 0.05, 'MyVc': 4.2, 'MyFX': 1.0, 'DelayParam': 0.75
}  # L50-55
```

### 3.3 数据加载流程

**`T21Dataset.__init__`**（L101-152）：
1. 扫描 `T21_cubes/` 目录，按红移过滤
2. 自动检测格式
3. varying_IC 按 `Npix` 过滤
4. 随机 train/val 切分（`val_frac=0.1`, `seed=42`）
5. 加载所有参数 → 计算归一化统计量

**`__getitem__`**（L302-364）：
1. 加载 T21 cube → 归一化 `(cube - mean) / std`
2. 随机裁剪 64³ patch（`i,j,k = randint(0, N-64+1)`）
3. 加载对应位置的 IC patch（delta + vbv）
4. 返回 `{'patch', 'ic_delta', 'ic_vbv', 'params'}`

### 3.4 IC 缓存机制

varying_IC 每个模拟有独立 IC 文件，全部加载会爆内存。用 **LRU 缓存**（`OrderedDict`，L121-229）：
- 最多同时缓存 `ic_cache_size=60` 个 IC 场
- 命中时 `move_to_end`，未命中时 `popitem(last=False)` 淘汰最旧

### 3.5 MAT 文件兼容

`load_mat`（L62-80）：先尝试 `scipy.io.loadmat`（v5 格式），失败则用 `h5py`（v7.3 HDF5 格式）。

---

## 4. 模型层 — VAE

**文件：** `models/vae.py`（124行）

### 4.1 整体结构

```
VAE3D
├── Encoder3D: (B,1,64,64,64) → mean, logvar (B,4,16,16,16)
└── Decoder3D: (B,4,16,16,16) → (B,1,64,64,64)
```

### 4.2 Encoder3D（L29-56）

```
init_conv: Conv3d(1→64, k=3)
stage[0]: ResBlock(64→64) × 2 + Conv3d(stride=2)   # 64³ → 32³
stage[1]: ResBlock(64→128) × 2 + Conv3d(stride=2)   # 32³ → 16³
mid:      ResBlock(128→128) × 2
to_latent: Conv3d(128→8, k=1)  → chunk → mean(4ch) + logvar(4ch)
```

- `logvar` 被 clamp 到 `[-30, 20]`（L55），防止数值不稳定

### 4.3 Decoder3D（L59-87）

Encoder 的镜像：
```
init_conv: Conv3d(4→128, k=3)
mid:       ResBlock(128→128) × 2
stage[0]:  ConvTranspose3d(stride=2) + ResBlock(128→128) × 2   # 16³ → 32³
stage[1]:  ConvTranspose3d(stride=2) + ResBlock(64→64) × 2     # 32³ → 64³
out:       GroupNorm → SiLU → Conv3d(64→1, k=3)
```

### 4.4 重参数化采样（L97-100）

```python
z = mean + randn_like(mean) * exp(0.5 * logvar)
```

### 4.5 损失函数（L119-123）

```python
L = MSE(recon, x) + kl_weight × KL
KL = -0.5 × mean(1 + logvar - mean² - exp(logvar))
```

### 4.6 ResBlock3D（L14-26）

```
GroupNorm → SiLU → Conv3d(3×3×3) → GroupNorm → SiLU → Conv3d(3×3×3) + skip
```
- GroupNorm 分组数 = `min(8, ch)`
- 输入输出通道不同时用 1×1 卷积做 skip

---

## 5. 模型层 — Flow Matching

**文件：** `models/flow_matching.py`（510行）

### 5.1 组件总览

```
ConditionalFlowMatcher
├── ICEncoder:    (δ, v_bv) → IC 特征 (B,4,16,16,16)
├── FlowUNet3D:  预测速度场 v(z_t, t, params, IC)
│   ├── SinusoidalEmbedding:  时间 t → 正弦嵌入
│   ├── Fourier 参数嵌入:     params → 随机 Fourier 特征 → MLP
│   ├── Encoder:              2 级下采样 + ResBlock(AdaGN)
│   ├── Bottleneck:           ResBlock 或 DiTBlock3D（可选）
│   └── Decoder:              2 级上采样 + ResBlock(AdaGN) + ICCrossAttn3D
└── 训练/采样/Reflow 方法
```

### 5.2 条件注入机制

**三种条件信号，三种注入方式：**

| 条件 | 注入方式 | 位置 |
|------|---------|------|
| 时间 t | SinusoidalEmbedding → MLP → AdaGN | 所有 ResBlock |
| 天体物理参数 | Fourier 特征 → MLP → AdaGN | 所有 ResBlock |
| IC 场 (δ, v_bv) | ICEncoder → ICCrossAttn3D | Decoder 每层 |

### 5.3 SinusoidalEmbedding（L20-31）

```python
freq = exp(-log(10000) × arange(dim/2) / (dim/2 - 1))
emb = cat([sin(t × freq), cos(t × freq)])   # (B, dim)
```

### 5.4 AdaGN — 自适应 GroupNorm（L34-45）

```python
scale, shift = Linear(cond_dim → ch×2).chunk(2)
output = GroupNorm(x) × (1 + scale) + shift
```
- 权重零初始化（L40-41），训练初期等价于普通 GroupNorm

### 5.5 ResBlock3D（L48-61）

与 VAE 的 ResBlock 不同，这里每个 norm 层都是 **AdaGN**，接受条件向量：
```
Conv3d → AdaGN(cond) → SiLU → Conv3d → AdaGN(cond) → SiLU + skip
```

### 5.6 ICCrossAttn3D — IC 交叉注意力（L68-118）

**为什么用交叉注意力而非 concat：**
- concat 只在输入层融合 IC，多层卷积后空间对应关系退化
- 交叉注意力让 decoder 每层每个位置直接查询 IC 场，保留物理 IC→T21 映射

**实现：**
```
Q = flatten(latent_feature)     # (B, D×H×W, C)
K = V = flatten(IC_feature)     # (B, max_ic_res³, ic_ch)
→ LayerNorm → MultiheadAttention → Linear → reshape + residual
```

**内存控制：** IC K/V 分辨率上限 `max_ic_res=8`（8³=512 tokens），用 `adaptive_avg_pool3d` 降采样（L105-106）。

**零初始化输出投影**（L93-95）：训练初期 cross-attn 输出为零，block 等价于 identity，不干扰主干训练。

### 5.7 DiTBlock3D — Transformer Bottleneck（L125-160）

可选的 UViT 风格 bottleneck，用 **adaLN-Zero** 条件化：

```
条件 → SiLU → Linear → 6 个投影 (shift_a, scale_a, gate_a, shift_f, scale_f, gate_f)

自注意力: h = LayerNorm(tok) × (1+scale_a) + shift_a → MultiheadAttn → tok + gate_a × h
前馈:     h = LayerNorm(tok) × (1+scale_f) + shift_f → FFN → tok + gate_f × h
```

- 6 个投影全部零初始化（L142-143），训练初期 DiT block 是 identity
- 通过 `use_dit_mid=True` 启用

### 5.8 ICEncoder（L178-197）

```
cat(δ, v_bv) → (B,2,64,64,64)
→ Conv3d(2→64) → SiLU → ResBlock×2
→ Conv3d(stride=2) → SiLU → ResBlock×2    # 64³ → 32³
→ Conv3d(stride=2)                          # 32³ → 16³
→ (B,4,16,16,16)
```

### 5.9 FlowUNet3D（L200-357）

**参数嵌入 — Fourier 特征**（L228-238）：
```python
# 每个标量参数 → 与随机频率相乘 → sin/cos
proj = params[:,:,None] * param_freqs[None]     # (B, 5, 64)
fourier = cat([sin(proj), cos(proj)])            # (B, 5, 128)
fourier = flatten → MLP(640 → 256 → 256 → 256)  # (B, 256)
```
- `param_freqs` 是 `register_buffer` 的随机矩阵（`randn × 2.0`），训练时固定
- 比直接用原始参数值更好，因为 Fourier 特征能捕捉高频变化

**CFG null token**（L240）：`nn.Parameter(zeros(256))`，dropout 时替换参数嵌入。

**UNet 主体：**
```
Encoder:
  init_conv(latent_ch → 128)
  stage[0]: ResBlock(128→128, cond)×2 + Conv3d(stride=2)   # 16³ → 8³
  stage[1]: ResBlock(128→256, cond)×2 + Conv3d(stride=2)   # 8³ → 4³

Bottleneck:
  ResBlock(256→256, cond) × 2   或   DiTBlock3D × 2

Decoder:
  stage[0]: ConvTranspose3d + cat(skip) + ResBlock(512→256, cond)×2 + ICCrossAttn3D
  stage[1]: ConvTranspose3d + cat(skip) + ResBlock(256→128, cond)×2 + ICCrossAttn3D

out_conv: GroupNorm → SiLU → Conv3d(128→4)   # 零初始化
```

**输出层零初始化**（L301-302）：训练初期预测速度为零，等价于不动。

**forward**（L305-357）：
1. 如果 `ic_cond_mode='concat'`，IC 在输入层 concat
2. 时间嵌入 + Fourier 参数嵌入 → 拼接为 `cond`（512维）
3. CFG dropout：`drop_cond=True` 时用 null token 替换参数嵌入
4. Encoder → Bottleneck → Decoder（每层 IC cross-attn）→ 输出速度场

### 5.10 ConditionalFlowMatcher（L364-509）

**training_loss**（L412-460）：
```python
z0 = randn_like(z1)                          # 噪声
t  = sigmoid(randn(B))                       # logit-normal 采样
zt = (1-t) × z0 + t × z1                     # 线性插值
vt = z1 - z0                                  # 目标速度

ic_enc = ICEncoder(δ, v_bv)
drop   = rand(B) < cfg_dropout                # 10% 概率 drop
v_pred = UNet(zt, t, params, ic_enc, drop)

loss = MSE(v_pred, vt)

# 可选功率谱损失
if ps_weight > 0:
    z1_pred = zt + (1-t) × v_pred             # 估计的干净潜向量
    loss += ps_weight × MSE(log PS(z1_pred), log PS(z1))
```

**sample**（L486-509）— 100步 Euler ODE + CFG：
```python
z = randn(latent_shape)
for i in range(100):
    t = i / 100
    v_uncond = UNet(z, t, params, ic, drop_all=True)
    v_cond   = UNet(z, t, params, ic, drop_all=False)
    v = v_uncond + cfg_scale × (v_cond - v_uncond)   # CFG
    z = z + v × dt
return z
```

**reflow_sample_z1**（L466-481）— 生成直线轨迹对：
```python
# 从 z0 积分到 z1_hat（无 CFG），用于 reflow 训练
z = z0
for i in range(num_steps):
    v = UNet(z, t=i/N, params, ic, drop=False)
    z = z + v × dt
return z   # 与 z0 配对，形成更直的轨迹
```

**_ps_loss**（L377-407）— 可微分功率谱损失：
- 对潜向量做 3D rFFT → 径向分 bin → log 空间 MSE
- `z_true` 用 `torch.no_grad()` 包裹，梯度只流过 `z_pred`

---

## 6. 训练脚本

### 6.1 train_vae.py（141行）

**关键参数：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 300 | 训练轮数 |
| `batch_size` | 4 | 64³ 体素较大，batch 不能太大 |
| `lr` | 1e-4 | AdamW 学习率 |
| `kl_weight` | 1e-4 | KL 散度权重（很小，重建优先） |
| `kl_anneal` | 50 | KL 线性退火轮数 |
| `base_ch` | 64 | 基础通道数 |
| `latent_ch` | 4 | 潜空间通道数 |

**训练循环**（L84-135）：
1. KL 退火：前 50 个 epoch 线性增加 `kl_weight`（L85, L35-37）
2. 前向：`model.loss(x, kl_weight)` → `(total_loss, recon_loss, kl_loss)`
3. 梯度裁剪 `clip_grad_norm_(1.0)`（L96）
4. CosineAnnealingLR 调度器（L68）
5. 每 25 epoch 保存 checkpoint（L130-134）
6. CSV 日志：`epoch, train_loss, train_recon, train_kl, val_loss, val_recon, val_kl`

**Resume 支持**（L70-76）：`--resume PATH` 恢复 model + optimizer + epoch。

### 6.2 train_flow.py（206行）

**关键参数：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 500 | 训练轮数 |
| `batch_size` | 8 | 潜空间操作，可以更大 |
| `lr` | 1e-4 | AdamW 学习率 |
| `base_ch` | 128 | UNet 基础通道（比 VAE 大） |
| `cfg_dropout` | 0.1 | CFG 条件 dropout 概率 |
| `time_mode` | lognormal | logit-normal 时间采样 |
| `ps_weight` | 0.01 | 功率谱损失权重 |
| `ic_cond_mode` | cross_attn | IC 注入方式 |
| `reflow_steps` | 0 | >0 时启用 reflow 训练 |

**训练流程**（L134-199）：
1. 加载冻结 VAE（L66-73）：`requires_grad_(False)`
2. 支持 varying_astro + varying_IC 联合训练（`ConcatDataset`，L76-86）
3. 每个 batch：
   - VAE encode `x → z`（冻结，`no_grad`）
   - 如果 `reflow_steps > 0`：用当前模型生成 `(z0, z1_hat)` 对替换真实 `z1`
   - `flow.training_loss(z1, params, ic_delta, ic_vbv)`
4. 梯度裁剪 1.0，CosineAnnealingLR
5. 每 50 epoch 保存 checkpoint
6. CSV 日志：`epoch, train_loss, val_loss`

---

## 7. 评估脚本

### 7.1 evaluate_vae.py（165行）

**评估指标：**
1. **重建 MSE** + **相对 MSE**（`||x-xhat||² / ||x||²`，目标 < 0.05）
2. **潜空间分布**：mean ≈ 0, std ≈ 1
3. **功率谱比值**：三个尺度（大/中/小 k），目标 0.95~1.05

**6 面板图**（L93-157）：
- 原始切片 × 3 + 重建切片 × 3
- 功率谱对比（log-log）
- PS 比值（目标线 1.0 ± 0.05）
- 潜空间直方图 vs N(0,1)
- 像素散点图（原始 vs 重建）
- 每通道潜空间分布
- 残差热力图

### 7.2 evaluate_flow.py（253行）

**流程：**
1. 加载 VAE + Flow 模型
2. 从验证集取条件（params, IC）
3. `flow.sample()` 生成潜向量 → `vae.decode()` 得到 T21
4. 反归一化到物理单位
5. 计算功率谱

**输出：**
- `power_spectrum.png`：Δ²(k) 对比 + 比值图
- `slices.png`：真实 vs 生成的中间切片
- `loss_curves.png`：训练损失曲线
- `power_spectra.npz`：原始 PS 数据

**PS 比值报告**（L236-241）：大尺度 (k<0.1) / 中尺度 / 小尺度 (k>0.5)

### 7.3 evaluate.py（182行）

**Simon 风格的联合评估**，指标：
1. **Pixel RMSE**：`sqrt(mean((gen - real)²))`
2. **PS RMSE**：`sqrt(mean((log10 Δ²_gen - log10 Δ²_real)²))`

**3 面板图**（L126-167）：
- 功率谱对比（mean ± 1σ）
- 像素 PDF 直方图
- 中间切片对比（Real | Generated）

---

## 8. 工具函数

### 8.1 utils/power_spectrum.py（65行）

**`power_spectrum(data, Lpix=3.0, kbins=50)`**（L9-58）：

球平均 3D 功率谱计算：
1. 输入归一化为 `(B, N, N, N)`
2. 减去均值（去直流分量）
3. `rfftn` → `|F|² × L³`（功率密度）
4. 几何等间距 k bins（`geomspace`）
5. 径向平均
6. 返回 `Δ²(k) = k³P(k)/(2π²)`

**`ps_batch_stats`**（L61-64）：返回 batch 的 mean 和 std。

### 8.2 plot_loss.py（59行）

读取 CSV 日志（`epoch, train_loss, val_loss`），画线性 + 对数两个子图。
报告最新 epoch、最新 train/val loss、最佳 val loss 及对应 epoch。

---

## 9. SLURM 提交脚本

### 9.1 公共配置

所有脚本共享：
```bash
#SBATCH -A FIALKOV-SL3-GPU    # 导师名下免费 GPU 额度（低优先级）
#SBATCH -p ampere              # A100 GPU 分区
#SBATCH --nodes=1
#SBATCH --gres=gpu:1           # 1 块 A100
#SBATCH --cpus-per-task=4
```

环境加载：
```bash
module load python/3.11.9/gcc/abrhyqg7
module load cuda/11.8
source .../venv/bin/activate
```

### 9.2 submit_vae.sh

| 配置 | 值 |
|------|-----|
| Job 名 | `vae_21cm` |
| 内存 | 32G |
| 时间 | 12h |
| 数据 | `varying_astro` |
| 关键参数 | `epochs=300, batch_size=4, base_ch=64, kl_weight=1e-4` |

### 9.3 submit_flow.sh

| 配置 | 值 |
|------|-----|
| Job 名 | `flow_21cm` |
| 内存 | 32G |
| 时间 | 12h |
| 数据 | `varying_IC` |
| VAE checkpoint | `vae_epoch0299.pt` |
| 关键参数 | `epochs=500, batch_size=8, base_ch=128, cfg_dropout=0.1, ps_weight=0.01, ic_cond_mode=cross_attn` |

### 9.4 submit_reflow.sh

| 配置 | 值 |
|------|-----|
| Job 名 | `reflow_21cm` |
| 内存 | 32G |
| 时间 | 6h |
| 数据 | `varying_astro` |
| 关键参数 | `epochs=100, lr=3e-5, reflow_steps=5, --resume flow_epoch0499.pt` |
| 输出目录 | `checkpoints/flow_reflow/` |

**用途：** Flow 训练收敛后，用 ODE 生成的 (z0, z1_hat) 对做精调，使轨迹更直，目标 100步 → 10-20步。

### 9.5 submit_eval.sh

| 配置 | 值 |
|------|-----|
| Job 名 | `eval_21cm` |
| 内存 | 16G |
| 时间 | 1h |
| 关键参数 | `n_samples=32, num_steps=100, cfg_scale=3.0, param_dim=5` |

---

## 10. 依赖关系图

```
dataset.py ─────────────────────────────────────────────────────┐
    │                                                           │
    ├──→ train_vae.py ──→ models/vae.py                         │
    │         │                                                 │
    │         ↓ (checkpoint)                                    │
    │    train_flow.py ──→ models/flow_matching.py              │
    │         │                 ├── FlowUNet3D                  │
    │         │                 ├── ICEncoder                   │
    │         │                 ├── ICCrossAttn3D               │
    │         │                 ├── DiTBlock3D                  │
    │         │                 └── ConditionalFlowMatcher      │
    │         │                                                 │
    │         ↓ (checkpoint)                                    │
    ├──→ evaluate_vae.py ──→ utils/power_spectrum.py            │
    ├──→ evaluate_flow.py ──→ utils/power_spectrum.py           │
    └──→ evaluate.py ──→ utils/power_spectrum.py                │
                                                                │
plot_loss.py ← (读取 log.csv，独立运行)                          │
                                                                │
submit_*.sh ← (SLURM 调度，调用上述 Python 脚本) ──────────────┘
```

**训练依赖链：**
```
VAE 训练 (300 epochs)
    ↓ vae_epoch0299.pt
Flow 训练 (500 epochs, 冻结 VAE)
    ↓ flow_epoch0499.pt
Reflow 精调 (100 epochs, --resume flow checkpoint)
    ↓ flow_reflow checkpoint
评估
```

---

## 11. 关键数学公式

### 流匹配（Conditional Flow Matching）

**线性插值路径：**
$$z_t = (1-t) \cdot z_0 + t \cdot z_1, \quad z_0 \sim \mathcal{N}(0,I), \quad t \in [0,1]$$

**目标速度场：**
$$v^*(z_t, t) = z_1 - z_0$$

**训练损失：**
$$\mathcal{L} = \mathbb{E}_{t,z_0,z_1} \left[ \| v_\theta(z_t, t, c) - (z_1 - z_0) \|^2 \right]$$

### Logit-Normal 时间采样

$$t = \sigma(\epsilon), \quad \epsilon \sim \mathcal{N}(0,1)$$

集中在 $t \approx 0.5$，这是速度场最难学的区域。

### Classifier-Free Guidance

$$v = v_{\text{uncond}} + s \cdot (v_{\text{cond}} - v_{\text{uncond}})$$

$s = 3.0$，训练时 10% 概率 drop 参数条件。

### 功率谱损失

$$\mathcal{L}_{\text{PS}} = \text{MSE}\left(\log \Delta^2_{\text{pred}}(k), \; \log \Delta^2_{\text{true}}(k)\right)$$

$$\Delta^2(k) = \frac{k^3 P(k)}{2\pi^2}$$

### VAE 损失

$$\mathcal{L}_{\text{VAE}} = \text{MSE}(x, \hat{x}) + \beta \cdot D_{\text{KL}}(q(z|x) \| \mathcal{N}(0,I))$$

$$D_{\text{KL}} = -\frac{1}{2} \sum (1 + \log\sigma^2 - \mu^2 - \sigma^2)$$

$\beta = 10^{-4}$，前 50 epoch 线性退火。

---

## 12. 完整运行流程

### 第一步：VAE 训练
```bash
sbatch submit_vae.sh
# 等待完成，检查 checkpoints/vae/vae_epoch0299.pt
```

### 第二步：VAE 评估
```bash
python evaluate_vae.py --ckpt checkpoints/vae/vae_epoch0299.pt
# 目标：相对 MSE < 0.05，PS ratio 0.95~1.05
```

### 第三步：Flow 训练
```bash
sbatch submit_flow.sh
# 确认 --vae_ckpt 路径正确指向 VAE checkpoint
```

### 第四步：Flow 评估
```bash
python evaluate_flow.py \
  --vae_ckpt  checkpoints/vae/vae_epoch0299.pt \
  --flow_ckpt checkpoints/flow/flow_epoch0499.pt
```

### 第五步：Reflow 精调（可选）
```bash
sbatch submit_reflow.sh
# 目标：100步 → 10-20步推理，质量基本不变
```

### 第六步：最终评估
```bash
sbatch submit_eval.sh
# 或本地：python evaluate.py --vae_ckpt ... --flow_ckpt ...
```
