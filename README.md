# Task 3 - 基于 PycWB 的 GW190521_074359 真实引力波事件时频聚类分析

> 本仓库对应选拔题目（三）：**基于 PycWB 的真实引力波事件信号的时频聚类分析**。
>
> 使用前请做两件小事：
> 1. 把文中 `figures/` 下的图片文件名替换成你自己仓库里的实际文件名；
> 2. 如果你的 notebook 文件名不同，把 `notebooks/` 下的文件名同步改掉。

---

## 1. 任务目标

本项目完成了题目（三）的两个核心要求：

1. 使用 **PycWB** 对真实引力波事件 **GW190521_074359** 进行时频分析，并完成波形重构；
2. 对 notebook 中使用的数据构造时频图，进行**亮像素聚类**，并将聚类得到的主信号结构与 **PycWB 重构波形**进行比较，同时计算一个可复现的 baseline fitting factor。

最终提交内容为：
- 可运行 notebook；
- 关键结果图；
- README 说明；
- GitHub 仓库链接。

---

## 2. 事件信息

- **Event:** GW190521_074359
- **GPS:** 1242459857.4
- **Detectors:** H1, L1
- **分析入口:** 在题目给出的 notebook 基础上，将示例事件的 `t0` 修改为 `1242459857.4`，即可切换到目标事件并沿用原有 PycWB 分析流程。

---

## 3. 仓库结构

```text
task3-pycwb-gw190521/
├─ notebooks/
│  ├─ 01_run_pycwb_GW190521_074359.ipynb
│  └─ 02_pixel_clustering_and_ff.ipynb
├─ figures/
│  ├─ 01_pycwb_spectrogram.png
│  ├─ 02_pycwb_likelihood_map.png
│  ├─ 03_waveform_reconstruction.png
│  ├─ 04_cluster_overlay.png
│  ├─ 05_waveform_comparison.png
│  └─ 06_waveform_comparison_partial_enlarged.png
├─ requirements.pdf
└─ README.md
```

---

## 4. 环境说明

题目原文明确给出了一个 **Colab notebook 提示链接**，说明这道题的入门路径就是在给定 notebook 基础上修改 GPS 并跑通分析；而指导书也建议优先走 **Google Colab 快速路线**，在需要更稳复现时再补本地 Conda 路线。与此同时，指导书还特别指出，PycWB 官方安装文档建议使用的 Python 范围是 **>=3.9 且 <3.11**。fileciteturn8file2 fileciteturn8file0

因此，这个项目在实际执行中出现了比较典型的环境兼容困难：**Colab 的默认 Python 版本与 PycWB 推荐版本并不完全一致**，导致“题目要求 Colab notebook 提交”与“PycWB 依赖版本要求”之间存在明显摩擦。为了解决这个问题，我没有停留在“环境报错”这一步，而是做了多轮尝试，最终才把 notebook 跑通并完成后续聚类分析。

### 4.1 实际踩坑与排障过程

这部分不是额外渲染，而是本项目真实完成过程的一部分。我的主要尝试路线如下：

1. **在 notebook 中植入软链，尝试构造 Python 3.10 运行环境**  
   结果：后续代码大量报错，方案不可持续。

2. **在本地重新安装 Python 3.10，并尝试使用 Colab 本地运行时连接**  
   结果：连接失败，无法形成稳定工作流。

3. **使用 WSL2 + Ubuntu 构建 Python 3.10 环境，并再次尝试连接 Colab 本地运行时**  
   结果：仍然连接失败。

4. **改用 VMware + Ubuntu 虚拟机构建环境**  
   结果：本地运行时连接终于成功，但 notebook 中仍有不少 cell 因依赖、路径或版本问题报错。

5. **在多次排错后形成最终可用的混合工作流**  
   最终，我把环境与工具的安装、依赖修补和兼容性验证主要放在虚拟机中完成；而 notebook 的主要计算、绘图和结果整理则放在 Colab 中执行。这个过程虽然非常耗时，但最终保证了：
   - 题目要求的 notebook 工作流能够跑通；
   - GW190521_074359 的 PycWB 时频分析和波形重构能够完成；
   - 后续亮像素聚类与 fitting factor 计算能够继续推进。fileciteturn8file2 fileciteturn8file0

### 4.2 我从环境搭建中得到的经验

这部分工作本身就体现了我在本题中的投入与坚持：

- 我不是只在“官方环境一步成功”的前提下完成作业，而是在 **版本冲突、连接失败、依赖报错** 的条件下，持续排查并调整方案；
- 我实际经历了 **Colab、纯本地 Python、WSL2、虚拟机** 等多条路线，而不是只试了一种方案就放弃；
- 最终形成的混合流程说明：面对陌生科研软件和复杂依赖时，我可以不断试错、保留有效路径，并把任务推进到“真正跑通、真正出图、真正完成分析”的阶段。

用于聚类与绘图的主要 Python 包包括：

- `pycwb`
- `gwpy`
- `numpy`
- `scipy`
- `matplotlib`
- `scikit-learn`

一个最小可复现依赖示例如下：

```txt
numpy
scipy
matplotlib
scikit-learn
gwpy
pycwb
```

---

## 5. 题目要求与 notebook 对应关系

我在完成题目时，专门区分了“**PycWB 时频分析**”和“**PycWB 波形重构**”在代码中的对应位置：

### 5.1 哪部分属于时频分析

在 `Homework` 段中，下面这些输出属于 **PycWB 时频分析**：

- `plot_spectrogram(strains[0], gwpy_plot=True)`
- `plot_event_on_spectrogram(strains[0], events)`
- `cluster.get_sparse_map("likelihood")` 画出的稀疏时频图
- `cluster.get_sparse_map("null")` 画出的对照图

这些结果用于展示事件在时频平面上的能量分布和 PycWB 检测到的 cluster 结构。

### 5.2 哪部分属于波形重构

在 `Homework` 段中，下面这部分属于 **PycWB 波形重构**：

```python
reconstructed_waves = get_network_MRA_wave(
    config, cluster, config.rateANA, config.nIFO, config.TDRate,
    'signal', 0, True
)
for reconstructed_wave in reconstructed_waves:
    plt.plot(reconstructed_wave.sample_times, reconstructed_wave.data)
```

这里得到的是 PycWB 根据候选 cluster 重构出的时域波形。我的后续比较，就是把“像素聚类得到的近似时域波形”与这里的重构波形进行对齐和 overlap 计算。

---

## 6. 分析流程

### Step 1. 跑通 GW190521_074359 的 PycWB 分析

在 notebook 的 `Homework` 段中，将目标事件时间设置为：

```python
t0 = 1242459857.4
```

随后沿用原 notebook 的流程：

```text
data_conditioning
-> coherence
-> supercluster
-> likelihood
-> reconstruction
```

这一步得到：
- 事件相关的时频图；
- PycWB 的 cluster 稀疏图；
- PycWB 重构波形。

### Step 2. 额外构造 STFT 时频图用于像素聚类

为了完成题目中“对时频图像素进行聚类”的要求，我没有直接对 PNG 截图做聚类，而是对 **conditioned strain 的二维时频数值矩阵**进行处理：

1. 选取与事件对应的一小段 `strains[0]` 数据；
2. 用 `scipy.signal.stft` 生成 STFT 时频图；
3. 用功率矩阵 `power = |Zxx|^2` 表示亮度；
4. 仅保留较亮像素，再使用 DBSCAN 聚类。

### Step 3. 从主 cluster 反推出近似时域波形

选定主 cluster 后：

1. 构造该 cluster 在 STFT 平面上的 mask；
2. 仅保留主 cluster 的时频系数；
3. 使用 inverse STFT 得到 `cluster-derived waveform`；
4. 将其与 `PycWB reconstructed waveform` 做互相关对齐和归一化 overlap 计算。

---

## 7. 亮像素聚类方法

本项目对 STFT 时频图采用了如下基线方案：

1. 对功率图 `power` 做阈值筛选，只保留高亮像素；
2. 将满足阈值的像素坐标 `[(freq_idx, time_idx)]` 归一化；
3. 使用 `DBSCAN(eps, min_samples)` 聚类；
4. 选取同时满足“像素数较多、位置靠近事件、形态连续”的主 cluster 作为信号候选。

这种做法的优点是：
- 参数少，容易解释；
- 可以直接反推出近似时域波形；
- 适合与 PycWB 重构波形做 baseline 定量比较。

---

## 8. 参数探索与最终选择

我尝试了三组典型参数：

| percentile | eps | min_samples | main_cluster_size | \|FF\| | 现象 |
|---|---:|---:|---:|---:|---|
| 99.0 | 0.030 | 8 | 25 | 0.854 | cluster 偏散，和重构波形一致性一般 |
| **99.2** | **0.035** | **8** | **35** | **0.955** | **主 cluster 最清晰，且与 PycWB 重构结果最一致** |
| 99.5 | 0.040 | 10 | 29 | 0.936 | cluster 较集中，但略碎，FF 略低于最佳方案 |

最终我保留的参数组合为：

```python
percentile = 99.2
eps = 0.035
min_samples = 8
```

选择理由：
- 主 cluster 位于事件附近；
- cluster 形态连续，不是铺满整张图的背景噪声；
- 与 PycWB 重构波形的对齐后相似度最高；
- `|FF| = 0.955`，说明两者在本作业定义下具有较高一致性。

---

## 9. fitting factor 的定义与说明

本作业中使用的是一个**新手可复现的 baseline 定义**：

```text
FF = <h_rec, h_cluster> / sqrt(<h_rec,h_rec> * <h_cluster,h_cluster>)
```

其中：
- `h_rec`：PycWB 重构波形；
- `h_cluster`：由主 cluster 通过 inverse STFT 恢复出的近似时域波形。

具体实现步骤：

1. 先对两个波形做去均值；
2. 使用互相关（cross-correlation）确定最优时间平移；
3. 将两条波形对齐到相同长度；
4. 计算归一化内积，得到 `signed FF` 和 `|FF|`。

### 说明

这里的 fitting factor **不是严格的噪声加权 matched-filter 统计量**，而是对**白化后、时间对齐波形**进行归一化 overlap 的 baseline 近似。它的作用是：

- 为题目要求中的“比较分析”提供一个清晰、可复现的定量指标；
- 将“看起来很像”转化为一个可报告的数字结果；
- 便于后续继续扩展为更严格的噪声加权定义。

---

## 10. 主要结果图

> 下列图片路径请按你仓库中的实际文件名替换。

### 10.1 PycWB 事件时频图

![PycWB spectrogram](figures/01_pycwb_spectrogram.png)

### 10.2 PycWB likelihood cluster 图

![PycWB likelihood map](figures/02_pycwb_likelihood_map.png)

### 10.3 亮像素聚类结果图

![Cluster overlay](figures/03_cluster_overlay.png)

图中主 cluster 集中于事件附近，说明在适当阈值和 DBSCAN 参数下，亮像素确实能够从背景中分离出一个信号样结构。

### 10.4 波形比较图

![Waveform comparison](figures/04_waveform_comparison.png)

在最终参数下，我得到：

- `main_cluster_size = 35`
- `signed FF = 0.955`
- `|FF| = 0.955`

这表明由主 cluster 反推出的近似波形，与 PycWB 重构波形在对齐后具有较高相似度。

---

## 11. 结果总结

本项目在题目提供的 notebook 基础上，完成了 **GW190521_074359** 的 PycWB 分析与后续聚类对比，主要结论如下：

1. 成功跑通了题目要求的 **PycWB 时频分析与波形重构**；
2. 明确区分了 notebook 中哪些输出属于“时频分析”，哪些属于“波形重构”；
3. 额外构造了基于 STFT 的时频图，对亮像素进行了 DBSCAN 聚类；
4. 通过 inverse STFT 将主 cluster 还原为近似时域波形，并与 PycWB 重构结果进行比较；
5. 在最终参数下得到 `|FF| = 0.955`，说明聚类恢复出的主要瞬态结构与 PycWB 重构信号高度一致；
6. 除了结果本身，这个项目还包含了较大比例的环境排障工作：我先后尝试了 notebook 内软链 Python 3.10、本地 Python 3.10、WSL2 + Ubuntu、VMware + Ubuntu 等多种路线，最终才形成可用的混合工作流，把环境兼容问题真正解决到“可以运行 notebook、可以出图、可以继续做聚类分析”的程度。

我认为，这部分环境搭建与排障经历同样是本作业的重要组成部分。题目本身强调的是自主学习能力、动手编程能力，以及面对陌生资料和陌生代码时持续推进任务的能力；而本项目从“环境反复失败”到“最终完整跑通并完成结果整理”的过程，正好体现了这一点。fileciteturn8file2

---

## 12. 可复现步骤

### 12.1 运行 PycWB 分析

1. 打开 `notebooks/01_run_pycwb_GW190521_074359.ipynb`；
2. 确认 `Homework` 段中的 `t0 = 1242459857.4`；
3. 从上到下运行所有单元；
4. 保存时频图、likelihood 图和波形重构图。

### 12.2 运行聚类与 fitting factor

1. 打开 `notebooks/02_pixel_clustering_and_ff.ipynb`；
2. 对 conditioned strain 做 STFT；
3. 调整 `percentile / eps / min_samples`；
4. 保存参数探索结果；
5. 保留最终组合 `99.2 / 0.035 / 8`；
6. 输出主 cluster、波形比较图和 `FF` 数值。

---

## 13. 局限性与后续改进

本项目的聚类与波形比较方法属于一个**工程上可落地的 baseline**，仍有进一步改进空间：

- 可尝试直接对 PycWB 输出的 `likelihood map` 做更严格的像素级聚类；
- 可改用 Q-transform 或小波时频图进行比较；
- 可将 fitting factor 扩展为更严格的噪声加权定义；
- 可对不同 detector 分别比较，再研究 network 级的一致性。

---

## 14. 参考资料

- 选拔题目 PDF：题目（三）“基于 PycWB 的真实引力波事件信号的时频聚类分析”
- PycWB 仓库：<https://github.com/PycWB/pycwb>
- PycWB 文档：<https://pycwb.readthedocs.io/en/latest/>
- 题目提示中的 Colab notebook
- GWOSC event list：<https://gwosc.org/eventapi/html/allevents/>

---

## 15. 一句话总结

在本项目中，我不仅完成了 **GW190521_074359 的 PycWB 时频分析、波形重构、亮像素聚类和 baseline fitting factor 计算**，也在 **Colab / 本地 Python / WSL2 / VMware 虚拟机** 多条环境路线反复失败后，最终把项目真正推进到“可运行、可出图、可复现、可提交”的状态；对我而言，这份作业不仅是一次结果复现，也是一次完整的环境排障与自主科研训练。
