<h1 align="left">
  <img src="docs/source/_static/images/small.png" alt="RLightning 标志" width="30" />
  RLightning
</h1>

<p align="center"><a href="README.md">English</a> | 中文</p>

<p align="center"><strong>面向具身智能、统一原型开发与规模化扩展的灵活强化学习框架</strong></p>

<p align="center">
  <img src="docs/source/_static/images/big2.png" alt="RLightning 展示图" width="500" />
</p>
<p align="center">
  <a href="https://rlightning.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/badge/Docs-Website-0F766E?style=for-the-badge" alt="项目文档" />
  </a>
</p>


## RLightning 是什么？

RLightning 是一个面向具身智能的强化学习框架，适用于从人形机器人运动控制到机器人操作等场景。其核心理念是“本地原型开发，无缝规模化扩展”：研究人员可以在标准的单进程工作流中开发和调试算法，随后只需修改配置文件，无须改动代码，即可扩展到分布式、多节点、多 GPU 训练。

## 核心设计

**统一编程接口** — 通过透明的运行时适配层，统一单进程与分布式执行环境下的编程接口，使本地原型能够零代码迁移至多节点集群。

**控制平面与数据平面解耦** — 研究人员通过控制引擎编排粗粒度工作流，底层数据平面则借助异步 I/O 和流水线机制处理节点间数据传输与任务调度，在不暴露分布式复杂性的前提下最大化吞吐量。

**细粒度资源管理** — 支持计算模块独立扩展，以及灵活的共置与调度策略。GPU 级进程共置可降低高频交互的通信开销，动态加载/卸载则可让顺序执行的组件复用计算资源。

**模块化异构生态集成** — 松耦合的模块化设计通过可扩展接口，实现主流模拟器（IsaacLab、MuJoCo、ManiSkill）、真实机器人硬件和经典算法库（RSL-RL）的标准化集成。


<p align="center">
  <img src="docs/source/_static/images/system_architecture.png" alt="系统架构" width="500" />
</p>


## 性能亮点

**吞吐量扩展最高达 15 倍** — 人形机器人全身控制训练只需修改配置，即可从单 GPU 扩展至 64 个 GPU（8 个节点），数据吞吐量最高提升 15 倍。在 8 节点规模下，异步 I/O 与流水线优化还可额外带来 3.75 倍的吞吐量提升。
<p align="center">
  <img src="docs/source/_static/images/humanoid_throughput.PNG" alt="通过规模化扩展获得更高吞吐量" width="650" />
</p>

**VLA 任务吞吐量提升 30% 以上** — 在计算密集型 OpenVLA 强化学习任务上，RLightning 相比基线将训练吞吐量提升了 30% 以上，同时保持相同的收敛曲线，准确率不受影响。
<p align="center">
  <img src="docs/source/_static/images/openvla-performance.png" alt="加速 OpenVLA PPO" width="650" />
</p>


## 支持的功能

| 类别 | 组件 | 说明 |
|----------|-----------|-------------|
| **强化学习组件** | 数据缓冲区 | `RolloutBuffer`（同策略）、`ReplayBuffer`（异策略） |
| | 策略 | 策略模型以及训练/推理算法的接口 |
| | 环境 | ManiSkill、MuJoCo、IsaacLab、Libero、远程环境 |
| **多维度扩展** | 环境 | 向量化环境数量、环境实例、异构模拟器 |
| | 任务 | 在单次训练中运行多个任务 |
| | 评估策略（Actor） | 多个评估工作进程，支持有状态路由与负载均衡路由 |
| | 训练策略（Learner） | 单进程训练或 DDP 分布式训练 |
| | 缓冲区 | 统一或分片存储，支持全局采样与数据路由 |
| **任务调度** | 同步 | 用于同策略算法（如 PPO）的 `SyncRLEngine` |
| | 异步 | 用于异策略算法的 `AsyncRLEngine` |
| **执行模式** | 单进程 | 在本地进行算法原型开发与调试 |
| | 分布式 | 支持数据并行训练的多进程、多 GPU、多节点模式 |
| **资源调度** | 默认 / 解耦 / 共置 / 手动 | 灵活的资源池策略 |
| **权重同步** | 双缓冲 / CPU 缓冲 / 分片 | 针对不同内存与吞吐量权衡的多种策略 |
| **可观测性** | 日志与性能分析 | TensorBoard、Wandb、SwanLab；内置计时性能分析器 |


## 内置示例

| 算法 | 模拟器 | 引擎 | 示例 |
|-----------|-----------|--------|---------|
| OpenVLA PPO | ManiSkill | `syncrl` | `examples/openvla_ppo/` |
| OpenPI PPO | Libero | `syncrl` | `examples/openpi_ppo/` |
| WBC Tracking | IsaacLab | `rsl` / `async_rsl` | `examples/wbc_tracking/` |

## 构建自己的算法

从模板开始，实现你的自定义策略：

```bash
cp -r examples/algorithm_template/ /path/to/your/project
cd /path/to/your/project && uv sync
```

一个最简的 `train.py` 如下：

```python
from pathlib import Path
from rlightning.utils.config import MainConfig
from rlightning.utils.launch import launch
from rlightning.utils.builders import (
    build_env_group, build_policy_group, build_data_buffer, build_engine
)

def main(config: MainConfig):
    env_group = build_env_group(config.env)
    policy_group = build_policy_group(config.policy.type, config.policy, config.cluster)
    buffer = build_data_buffer(config.buffer.type, config.buffer)
    engine = build_engine(config, env_group, policy_group, buffer)
    engine.run()

if __name__ == "__main__":
    launch(main_func=main, config_path=Path(__file__).parent / "conf")
```

继承 `BasePolicy`，实现 `rollout()` 和 `learn()`，使用 `@POLICY.register("my_algo")` 注册，然后运行 `bash launch_train.sh` 启动。

## 文档

在线文档：[https://rlightning.readthedocs.io/en/latest/](https://rlightning.readthedocs.io/en/latest/)

完整文档源码位于 [`docs/source/`](docs/source/)，包括：

- [安装](docs/source/getting_started/installation.rst)
- [快速入门](docs/source/getting_started/quickstart.rst)
- [系统架构](docs/source/user_guide/system_architecture.rst)
- [配置](docs/source/user_guide/configuration.rst)
- [构建自己的强化学习项目](docs/source/getting_started/build_your_own_rl.rst)
- [调试与规模化扩展](docs/source/user_guide/debug_scaling_up.rst)
- [贡献指南](docs/source/contribution/contributing_guide.rst)

## 贡献

有关开发环境配置和代码审查规范，请参阅[贡献指南](docs/source/contribution/contributing_guide.rst)。

## 许可证

本项目采用 Apache License 2.0 许可证。详情请参阅 [LICENSE](LICENSE) 文件。
第三方代码以及复制或改编的文件记录在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 中。
