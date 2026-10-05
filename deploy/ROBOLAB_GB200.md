# 在 GB200/B200 上运行 FLUX Action + RoboLab

这套部署已经在 Grace Blackwell GB200 节点上完成了 RoboLab 闭环评测。需要先澄清：
模型推理是严格的单卡推理，但整个仿真闭环使用两张彼此隔离的 GPU——一张运行
FLUX Action，另一张运行 Isaac Sim 的物理和渲染。它不是把模型并行到两张卡，也不是
让模型和 Isaac Sim 争抢同一张卡。

## 实际运行拓扑

```text
GB200 节点（Grace ARM64，4 × NVIDIA GB200，驱动 580.126.20）

GPU 0                                         GPU 1
FLUX Action policy server                     RoboLab + Isaac Lab/Isaac Sim
Base FP8r, 4 NFE                              headless Vulkan + PhysX + cameras
原生 Python 环境                              ARM64 Enroot 容器
       │                                             │
       └──── 127.0.0.1:8000 / OpenPI WebSocket ─────┘
             observation → action chunk
```

Slurm 脚本申请整台四卡节点（`--gpus-per-node=4 --exclusive`），这是集群的节点分配
方式；当前闭环只使用 GPU 0 和 GPU 1。GPU 2、3 不参与模型推理或仿真。因此：

- FLUX Action 的延迟仍然是单 B200/GB200 数字；
- RoboLab 的闭环吞吐不能称为“整个系统单卡”；
- 这种隔离避免 RTX 渲染/物理计算改变模型的时钟、显存和 kernel 调度。

## 为什么这套配置能在 GB200 上启动

### 1. 使用原生 ARM64 Isaac 镜像

GB200 节点的主机 CPU 是 Grace（ARM64），不能直接拿常见的 x86-64 Isaac Sim 镜像来跑。
实际镜像是：

```text
/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/.cache/enroot/isaaclab-2.3.0-arm64.sqsh
```

已从镜像内部核对到：

| 组件 | 实际版本 |
|---|---|
| 容器架构 | ARM64/aarch64 |
| 容器系统 | Ubuntu 24.04.2 |
| Isaac Lab | 2.3.0 |
| Isaac Sim | 5.1.0-rc.19 |
| Isaac Python | 3.11 |
| CUDA 用户态库 | 12.8 系列 |
| 镜像大小 | 21,474,885,632 bytes |

这里没有使用 QEMU，也没有在 ARM 主机上模拟 x86。Isaac Sim、Kit、PhysX、Vulkan
插件和 Python 扩展均来自同一套原生 aarch64 镜像。

需要注意的是，这不是一个可以泛化为“所有 B200 都受到 Isaac Sim 5.1 官方支持”的
结论。Isaac Sim 5.1 的公开系统要求列出的推荐 GPU 是 RTX 系列，而且其 aarch64
公开支持范围主要是 DGX Spark。我们当前使用的是已经通过实测的 Grace-Blackwell
ARM64 镜像和驱动组合，应把它视为经过验证的固定软件栈，而不是随意替换版本。

### 2. 驱动由宿主机提供，用户态栈由容器固定

容器通过 Slurm Pyxis/Enroot 启动。宿主机提供 580.126.20 NVIDIA 驱动和 GPU 设备，
容器固定 Isaac Sim/Isaac Lab/CUDA 用户态依赖。这避免在登录节点或 FLUX Action 的
Python 环境中直接安装完整 Isaac 栈。

ARM64 镜像本身没有宿主驱动中的 `libnvidia-ngx.so.1`（以及它指向的版本化文件）和
`/usr/share/nvidia/nvoptix.bin`。启动器把这三项复制到作业专属目录，再显式挂入
Enroot，并把额外库目录加入 `LD_LIBRARY_PATH`。同时 Kit 使用
`--/rtx/indirectDiffuse/enabled=true`，与固定的 Isaac Lab balanced rendering 配置保持
一致。缺少这些挂载虽然可能不立即终止 headless 进程，但会触发 NGX/NRD 初始化错误并
改变相机亮度，因此不能用于策略精度对比。

Isaac Sim 5.1 文档给出的 Linux 测试驱动基线是 580.65.06；当前驱动高于这个版本。
但“版本号够新”本身仍不保证成功，镜像架构、Vulkan、Python ABI 和缓存目录也必须
同时正确。

### 3. 只跑 headless rendering experience

启动命令使用：

```bash
/workspace/isaaclab/_isaac_sim/python.sh policies/cosmos3/run.py \
  --headless --device cuda:1 ...
```

日志显示实际加载的是：

```text
/workspace/isaaclab/apps/isaaclab.python.headless.rendering.kit
```

因此不依赖 X Server 或桌面窗口，但仍保留相机所需的 Vulkan/RTX 离屏渲染。日志中的
`GLFW initialization failed` 和 `failed to open the default display` 在这种运行方式下
是预期的无显示告警，不代表物理或相机流水线没有运行。

### 4. 不改坏 Isaac 自带的 Python 环境

RoboLab 客户端还需要 `tyro`、`websockets`、`msgpack-numpy` 等少量包。它们没有被
直接安装进镜像，而是放在宿主目录 `robolab/.container-site`，然后置于
`PYTHONPATH` 的最前面：

```bash
export PYTHONPATH=/workspace/agent_deploy/robolab/.container-site:\
/workspace/agent_deploy/robolab
```

这个 sidecar 依赖目录使用 CPython 3.11 的 aarch64 wheels，并刻意不覆盖 Isaac
镜像中的 NumPy。当前目录还显式提供 `typing_extensions==4.16.0`，解决 Isaac 自带
旧版 `typing_extensions` 缺少 `TypeForm`、导致新版 `tyro` 无法导入的问题。

这是一个很容易踩到的坑：从宿主机普通虚拟环境复制 Python 3.13 或 x86 wheel，或者
用 NumPy 2 覆盖 Isaac 内置 NumPy，都会让 Python 扩展在 Kit 初始化前就失败。

### 5. 持久化 Kit/shader cache

以下目录从 Lustre 挂入容器：

```text
$CACHE_ROOT/isaac-root  -> /root/.cache
$CACHE_ROOT/isaac-kit   -> /isaac-sim/kit/cache
```

首次启动需要生成大量 shader/Kit 缓存，明显慢于后续启动。持久化缓存既减少启动时间，
也避免只读 SquashFS 或临时目录导致 shader cache 无法写入。容器本身以
`--container-writable` 启动，但长期缓存仍写回显式挂载目录。

### 6. 先让模型完成编译，再启动仿真

FLUX Action 首先在 GPU 0 启动并执行两次 warm-up。启动脚本轮询
`127.0.0.1:8000`，最长等待 2700 秒；只有端口 ready 后才以 `srun --overlap`
启动 Isaac Sim。这样避免 Isaac 已经开始计时，而模型还在进行 TorchInductor、Triton
或 Quack kernel 编译。

仿真和模型服务在同一节点上通过 loopback WebSocket 通信，不依赖外部网络，不需要把
Isaac Python 依赖和模型 Python 依赖装进同一个环境。

## 启动过程

入口脚本是 `deploy/flux3_action_robolab.sbatch`。一个可复现的 smoke test 为：

```bash
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/agent_deploy/flux-action

export ISAAC_IMAGE=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/.cache/enroot/isaaclab-2.3.0-arm64.sqsh

sbatch -A nvr_elm_llm \
  --export=ALL,FLUX_ACTION_CHECKPOINT=outputs/droid/variants/fp8r,FLUX_ACTION_NFE=4,FLUX_ACTION_COMPILE_DIT=1,ROBOLAB_TASKS=BananaInBowlTask,ROBOLAB_NUM_ENVS=1,ROBOLAB_NUM_RUNS=1,ROBOLAB_VIDEO_MODE=none,ROBOLAB_RUN_NAME=flux3_smoke_fp8r_4nfe \
  deploy/flux3_action_robolab.sbatch
```

脚本内部依次完成：

1. 在 GPU 0 加载 Base FP8r、4 NFE 的 FLUX Action，并完成编译/warm-up；
2. 探测本机 WebSocket 端口，确认 policy server ready；
3. 用 Enroot 在 GPU 1 启动 RoboLab/Isaac Sim；
4. RoboLab 渲染多相机观测并通过 OpenPI 协议发送给 policy server；
5. policy server 返回 action chunk，RoboLab 将动作施加到仿真机器人；
6. RoboLab 写出 episode、成功判定和 timing，服务端另存每次请求的输入与计时。

扩大到十个并行环境时只需设置 `ROBOLAB_NUM_ENVS=10`。这会增加 Isaac 侧的渲染和
物理负载，但不会把 FLUX Action 变成多卡推理。

## 已验证结果

以下作业均在日志和 Slurm accounting 中显示 `COMPLETED`、退出码 `0:0`：

| Slurm job | 任务 | 并行环境 | 结果 |
|---|---|---:|---|
| 7487877 | `BananaInBowlTask` | 1 | 1/1 episode success |
| 7487954 | `BananaInBowlTask` | 10 | 10/10 episode success |
| 7488069 | `RubiksCubeAndBananaTask` | 10 | 10/10 episode success |

作业 7488069 的节点日志明确识别出 4 张 `NVIDIA GB200`、每张 189471 MiB，驱动
580.126.20；Isaac Sim 完成环境初始化、相机渲染、817 个仿真 step（最慢环境）和最终
JSON 写出。因此这里的“跑起来”是完整的闭环 benchmark，不只是导入 Python 或打开空场景。

## 日志里的报错该如何判断

成功运行中仍会看到几类高噪声日志：

- `GLFW initialization failed` / `failed to open the default display`：headless 节点没有
  X display，不阻断离屏渲染；
- `Could not get NGX parameters` / `Failed to create NGX context`：驱动资产挂载不完整，
  该次运行不能用于正式图像或策略比较；
- `Failed to compile ... PackForNRD.cs.hlsl`：同样按渲染栈失败处理，不纳入正式指标；
- `CUDA_VISIBLE_DEVICES ... enumeration are different`：Kit/Vulkan 和 CUDA 的设备编号
  体系不同，所以脚本同时固定服务进程的可见卡和 Isaac 的 `--device cuda:1`；
- USD unresolved reference、actuator deprecated 参数：当前资产仍能加载和运行，但换场景
  或比较物理一致性时需要单独审计。

判断作业是否成功不能只 grep `Error`。至少要同时检查：

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 100 flux-action/logs/flux3-robolab-JOB_ID.out
tail -n 200 robolab/logs/flux3-robolab-JOB_ID.log
find robolab/output/RUN_NAME -name 'log_*.json'
```

真正的成功条件是 Slurm 退出码为 0、episode JSON 全部写出，并且其中的 `success`
字段符合预期。

## 当前限制

- 这是对当前 ARM64 image + driver + Isaac 版本组合的验证，不代表任意 Isaac Sim 版本
  都能在 B200 上直接运行；
- 只验证了当前任务使用的离屏 RGB 渲染路径，不代表所有 RTX rendering feature 都已
  覆盖；正式作业要求日志中没有上述 NGX/NRD 错误；
- Isaac Sim 5.0 和 5.1 的 PhysX 行为可能不同，benchmark 结果只能与相同版本栈直接比较；
- 录制视频会额外引入渲染、编码和 I/O 开销。测策略稳态延迟时应使用
  `ROBOLAB_VIDEO_MODE=none`，需要展示视频时再单独进行完整录制；
- 十个并行环境报告的 policy timing 是批量闭环统计，不等价于单 observation replay 的
  单卡端到端延迟。后者应继续用 `deploy/benchmark_service.py` 测量。

## 相关文件

- `deploy/flux3_action_robolab.sbatch`：完整 Slurm/Enroot 启动器；
- `src/flux_action/serving/robolab.py`：OpenPI WebSocket adapter；
- `robolab/policies/cosmos3/run.py`：RoboLab 侧策略客户端入口；
- `robolab/output/<run-name>/`：episode 和成功判定；
- `flux-action/outputs/robolab-sim/<job-id>/`：服务端记录的 observation/request；
- `flux-action/logs/robolab-server-<job-id>.log`：模型服务日志；
- `robolab/logs/flux3-robolab-<job-id>.log`：Isaac/RoboLab 日志。

公开参考：

- [Isaac Sim 5.1 system requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [Isaac Sim 5.1 container installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html)
- [Isaac Lab container guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html)
