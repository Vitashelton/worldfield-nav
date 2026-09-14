# RelationNav：Habitat-GS 关系执行 Pilot（组会版）

## 30 秒摘要

本轮不是训练大模型，而是先验证一个容易被普通导航指标掩盖的问题：机器人到达某个 waypoint，并不等于完成了任务关系。基于 Habitat-GS 的 RGB-D、绝对位姿和 NavMesh，我们复用 400 条固定 episode、1,408 个关系阶段，对比 arrival-only、同目标重试、关系验证、关系保持恢复和 oracle。结果显示，arrival-only 在 held-out episodes 上名义成功率为 100%，但关系完成率仅 64.9%；关系保持恢复将关系完成率提高到 97.4%，语义任务完成率达到 92.0%。这是一项 pilot diagnostic，结论是“需要关系级完成验证与恢复”，不是已经完成的新 planner 或 VLM 结果。

## 1. 问题

传统距离阈值容易把“到了门前”“停在门槛”“接近地标”误报为任务完成。本 pilot 将阶段写成实体—关系约束，例如 `APPROACH`、`CROSS`、`ENTER`、`OBSERVE`，并用 Habitat-GS 的几何真值检查关系是否真的发生。

## 2. 实验协议

- 4 个已有 Habitat-GS 室内 scene；400 个 deterministic episodes；1,408 个阶段。
- train 300 episodes，held-out 100 episodes。
- 同一 episode、同一固定 executor、同一目标关系，改变的只有完成判据和恢复策略。
- Habitat-GS 负责 RGB-D、pose、NavMesh 与关系谓词；本轮未调用 Qwen，也未训练 DINO/导航网络。

## 3. Held-out 结果（pilot）

| 方法 | 名义到达 | 关系完成 | 语义任务完成 | 错误阶段推进 | 恢复成功 |
|---|---:|---:|---:|---:|---:|
| Arrival-only | 100.0% | 64.9% | 0.0% | 35.1% | 0.0% |
| Same-goal retry | 100.0% | 64.9% | 0.0% | 35.1% | 0.0% |
| Relation-verified | 0.0% | 50.0% | 0.0% | — | — |
| Relation-preserving recovery | 92.0% | 97.4% | 92.0% | 0.0% | 92.6% |
| Oracle | 92.0% | 97.4% | 92.0% | 0.0% | — |

关键观察：名义 SR 不能替代关系完成率；同一个失败目标重复执行没有帮助；保持当前关系、在同一关系集合内重选，才可能恢复。

## 4. 直接展示的资产

- 结果图：`paper_assets/figures/relationnav_pilot_results.png`
- 同 episode 完成矩阵：`paper_assets/figures/relationnav_episode_completion_matrix.png`
- Habitat-GS 案例视频：`paper_assets/videos/relationnav_habitat_pilot.mp4`
- 完整 CSV：`paper_assets/tables/relationnav_pilot_summary.csv`
- 结果说明：`docs/results/RELATIONNAV_PILOT_RESULTS.md`

讲图时先放视频，再放完成矩阵，最后放柱状图。重点句：**同一批 episode 中，距离到达可以看起来成功，但关系没有完成；关系保持恢复把错误阶段推进清掉了。**

## 5. 必须主动说明的限制

当前结果是内部 pilot diagnostic。其浅层 boundary realization 仍需在正式 benchmark 中改为由 scene topology 自动生成；不能把本轮数值直接写成最终论文主结果。当前也没有 VLM 或可学习模块的增益结论，避免将 oracle/pilot 结果冒充模型结果。

## 6. 后续正式论文实验

1. 用室内 scene 的 portal/room/landmark 拓扑自动生成关系任务，scene-disjoint 划分。
2. 固定低频 Qwen 只输出关系/实体语义，不输出坐标或底层动作；所有响应缓存。
3. 用 DINOv3 作为目标图像与当前观测的视觉一致性特征，输入同一候选关系集合。
4. 对比 arrival-only、关系验证、关系保持恢复，以及 `Qwen semantic prior + fixed executor`；学习模块若无法超过规则基线则如实收缩论文主张。
5. 在 Habitat-GS 动态 avatar 场景验证通道阻塞后的关系保持重选，再选少量 Ranger Mini 门口/走廊案例做真机可行性验证。

## 7. 当前论文边界

可以主张：关系级任务完成不能由单点到达替代；Habitat-GS 能提供可复现的关系完成真值；关系保持恢复是一个可部署的系统设计方向。

暂不能主张：新的 VLM、新的 Nav2 planner、端到端具身模型、或已完成的 sim-to-real 泛化。

## 导师可能追问

**问：这是不是规则？**

答：当前 pilot 是 reference implementation，目标是先证明评价缺口；方法创新暂不夸大，正式论文将把重点放在关系任务协议、可验证完成判据和动态失败恢复的系统评测。

**问：为什么不直接用距离？**

答：距离只能表示接近，不能表示已经跨过 portal、进入目标区域或保持地标可见；pilot 中 arrival-only 的 100% 名义到达与 64.9% 关系完成率正好量化了这个差异。

**问：Qwen 在哪里？**

答：这次 pilot 不调用 Qwen，避免把 API 输出与执行能力混在一起；正式实验只在任务开始、岔路和失败事件低频调用并缓存语义关系。

**问：这是不是完整具身闭环？**

答：不是。本轮是关系执行协议的仿真验证；完整闭环需要动态 avatar、真实 VLM prior 和 Ranger Mini/Nav2 的小规模验证。
