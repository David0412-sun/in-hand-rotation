# Vision Hand 资产

本目录来自相邻项目 `rapidhand_isaacgym/assets`，包含默认的 `vision_hand` 及其可选碰撞简化版。原始 `urdf/vision_hand.urdf` 和 41 个 STL 网格按字节复制，没有修改关节、几何、惯量或限位。资产采用相对路径，运行时不需要相邻项目或原 Isaac Lab 项目。

`manifest.json` 保留原始来源、配置与依赖哈希，以及 `hands.vision_hand` 元数据。`integration.copied_files_sha256` 记录本目录实际复制的 42 个文件的 SHA-256；`hands.vision_hand.source_sha256` 则是导出前原始 URDF 的来源哈希，两者含义不同。

手部根节点及掌部节点均为 `BASE`，共有 20 个可动关节、20 个固定关节和 41 个 link。加载时须设置 `collapse_fixed_joints=False`，并按实际导入后的关节名称映射控制索引，不要假设 manifest 中的顺序就是 Isaac Gym 的 DOF 顺序。

以下 15 个 link 的名称包含 `TACTILE`，均保留碰撞几何，应通过实际刚体名称建立触觉索引：

- 拇指：`TDIP_TACTILE_DOWN`、`TDIP_TACTILE_UP`、`TPIP_TACTILE`
- 食指：`IDIP_TACTILE`、`IPIP_TACTILE`、`IMCPS_TACTILE`
- 中指：`MDIP_TACTILE`、`MPIP_TACTILE`、`MMCPS_TACTILE`
- 无名指：`RDIP_TACTILE`、`RPIP_TACTILE`、`RMCPS_TACTILE`
- 小指：`LDIP_TACTILE`、`LPIP_TACTILE`、`LMCPS_TACTILE`

`*_CAMERA` link 是保留的外壳几何；复制资产不会创建 Isaac Gym 相机传感器，也不会自动产生触觉读数。触觉读取由训练任务从接触力张量中提取上述刚体的接触信息。

源演示采用固定基座、位置驱动，默认刚度 3.0、阻尼 0.5、关节摩擦 0.01。Vision Hand 的 URDF 为所有 20 个可动关节设置了 1 Nm 力矩限位和 1 rad/s 速度限位，各关节角度范围以 URDF 为准。源演示初始目标为经过关节限位裁剪的零位；这些演示设置不代表已经完成旋转任务或硬件动力学标定。

## 可选的简化碰撞 URDF

`urdf/vision_hand_simple_collision.urdf` 复用原始 STL 作为外观，将 11 个结构件的 `<collision>` 替换为盒体：拇指的 `TMCPF`、`TMCPS`、`TDIP`，以及其余四指的 `MCPF` 和 `MCPS`。共保留 30 个碰撞网格，包括掌部、其余指节、全部 5 个 camera 外壳和 15 个 tactile 碰撞面。

这些盒体覆盖原始网格，但会填充圆角周围的空间，碰撞响应并不完全等价。没有强行替换全部 link：掌部和部分指节的包围盒会引入额外自碰撞，camera 的盒体也会侵犯邻指间隙，因此这些部件保留原始碰撞网格。

几何筛查中，默认姿态及 200 个带复位噪声的姿态没有新增自碰撞对，55 mm 双球均无初始穿透，最小几何间隙约 1.81 mm。该检查基于原始 STL 的数学凸包，不能覆盖完整关节活动范围。盒体角部相对原凸包的最大外扩约 13.6 mm，发生在四指 MCPS，因此应把该资产视为可选近似版本。

两个版本的 41 个 link、20 个可动关节、20 个固定关节、惯量、限位和外观完全一致。默认仍加载原始 URDF；在现有训练或回放命令中加入以下参数即可选择简化版，不需要新建任务或改变网络：

```bash
task.env.handAssetFile=urdf/vision_hand_simple_collision.urdf
```

训练与回放应使用相同的 URDF。旧权重的输入输出维度仍兼容，但碰撞形状变化可能改变动作效果。输出目录仍按物体和旋转轴组织，切换资产不会自动创建新的模型目录。

可用现有接触检测脚本逐个检查触觉的触发、释放与不同环境之间的隔离：

```bash
python scripts/check_rapidhand_tactile.py \
  --asset-file urdf/vision_hand_simple_collision.urdf --headless \
  --output /tmp/rapidhand_simple_collision_tactile.json
```

此方案简化的是碰撞形状，刚体、关节和形状数量没有减少；不能据此保证 4096 个环境的长期训练不再发生显存或 PhysX 错误。原始导入文件的哈希仍由 `manifest.json` 记录，简化版是独立的派生资产。

2026-10-23 在 RTX 4080 / Isaac Gym Preview 4 上验证通过：15 个触觉的接触、释放和环境隔离检查；55 mm 双球的 256 环境、10 轮可视化训练；保存模型后 16 环境、128 个控制步的可视化回放；4096 环境、2 轮短训。验证使用原有自碰撞设置，训练、回放数据及模型参数均为有限数值。4096 环境对照中，新旧版显存采样处于相近水平，尚无明显节省证据；短训也不代表 4000 轮稳定性已经验证。测试产物保存在 `/tmp/rapidhand_simple_collision_validation/`，没有覆盖 `runs` 下的模型。
