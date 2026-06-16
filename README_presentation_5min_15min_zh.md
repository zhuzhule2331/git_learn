# 床面板异物识别项目分享讲稿（15分钟主讲版）

本文件按 15 分钟组内分享设计，重点突出三件事：

1. 为什么路线从单模型演进到 U-2-Net + Dinomaly。
2. 整个工程流程如何从数据到部署闭环。
3. 自己写的自动化脚本如何显著减轻人工负担。



## 二、15分钟逐字讲稿（组内分享版，可直接照读）

### 00:00 - 01:00 开场与目标

大家好，今天我做一个组内项目分享，主题是床面板异物识别。
这个项目的目标是，在真实临床场景下稳定识别床面板异物，并且能在 Jetson AGX 上实时部署运行。
这次分享更偏工程复盘，我重点讲三部分：路线演进、全流程工程化、以及我自己写的一套自动化脚本如何把人工工作量显著降下来。

### 01:00 - 02:30 问题定义与核心难点

我们面对的不是标准实验室场景，而是动态、复杂、光照变化明显的现场。
早期遇到的问题是样本采集时。未采集足量正常样本。样本分布不平衡，正常样本偏少，异常样本偏多，模型很难学到稳定的正常分布。
另外在部署阶段，除了精度，还必须考虑实时性、吞吐和资源约束。


### 02:30 - 04:30 路线探索到最终选型

项目早期我们并行评估两条路线。
第一条是语义分割路线，核心判断是床面板区域出现与背景不一致的分割块即认为异常，考虑过过 sam3、unet、sam2。
第二条是异常识别路线，主要调研和实验了 PatchCore、UniADet、Dinomaly2、AdaptCLIP。

最后发现，单独分割在复杂背景下稳定性不足需要标注所有异常，对于未见过的异常无法保证稳定识别，稳定性难以保障，单独异常识别又容易受到整图读入时不必要的复杂环境输入波动影响。
![最初的版本](readme_image/ealier_version1_Di.png)  
所以最终路线定为两阶段组合：U-2-Net 先提取关心区域，Dinomaly 再做异常评分。

图片示例：  
![早期分割](readme_image/ealier_vision2.png)
![早期异常识别](readme_image/ealier_version2_DI.png)

![早期标注](readme_image/ealier_vision4labelme.png)


  
![早期分割](readme_image/ealier_version3unet.png)
![早期异常识别](readme_image/ealier_version3Di.png)

![早期标注](readme_image/ealier_version4labelme2.png)

![后期标注](readme_image/final_version1labelme.png)

### 04:30 - 06:30 全流程架构（参考原README）

我们的完整流程是这样的。
第一步，采集侧：工业相机视频流进入系统。
第二步，分割侧：U-2-Net 输出前景 mask，去除背景干扰。
第三步，ROI 侧：按 mask 的外接框裁切，并做等比例缩放加居中填充。
第四步，检测侧：Dinomaly 输出 anomaly map 与样本分数。
第五步，判定侧：根据阈值文件给出 normal、warning、abnormal 状态。
第六步，部署侧：Jetson AGX + TensorRT 做实时推理闭环。

这条流程和原 README 的训练、转换、推理、部署链路是一致的

### 06:30 - 09:30 关键迭代与效果提升逻辑

第一阶段是静态采图，初期正常样本太少，效果不稳定。
第二阶段补充正常样本后，模型表现明显提升，这也验证了异常检测任务首先要学稳正常分布。
第三阶段我们重构了 ROI，不再分割整床，只保留床面板和前方关键区域，提高输入相似度。
第四阶段做动态视频采集，并通过降低曝光多次采集，增强对不同光照环境的鲁棒性。
第五阶段优化输入尺寸，从 448x448 调整到 224x448，匹配“高而不长”的有效区域，减少黑边和冗余计算。

![标注示意](readme_image/fenge1_labelme.png)

![448方案](readme_image/improve1_448x448.png)
![224x448方案](readme_image/improve2_224x448.png)

### 09:30 - 12:30 自动化脚本体系（本次重点）

下面是我认为这个项目工程价值最高的部分，就是自动化脚本体系。

第一类，数据采集与筛选自动化。
通过视频抽帧、分割、特征阈值筛选，把原来人工逐帧挑图的工作变成批处理流程。
代表脚本包括：

`data_deal_support/video_dandom_picture_for_Unet.py`、
  抽取用于U2net标注的帧率
`data_deal_support/vedio_to_Dinomaly_train_data_auto.py`、
  利用训练好的模型结合绘制好的样本进行简单的宽高比例面积特征学习，自动区分视频中的帧分割结果是否可用   
  
![数据筛选](readme_image/shujujishaixuan.png)
`data_deal_support/vedio_Unet_result_move_to_train_good.py`。
  将上一个程序区分完成的帧搬运到Dinomaly train good 用于开始Dinamly训练

第二类，标注到训练集结构自动化。
把 json 标注自动转成 U-2-Net 与 Dinomaly 所需目录和掩码格式，减少手工改名和目录组织错误。
代表脚本包括：
`data_deal_support/json_choose_to_Unet_train.py`、
`data_deal_support/json_to_Dinomaly_danger.py`。

第三类，danger 样本构建自动化。
使用 YOLO + Box2Poly MLP 自动得到分割后 danger 图与对齐的 ground truth mask，并自动校验一一对应关系。
代表脚本包括：
`data_deal_support/danger_original_to_box2poly_dinomaly_224x448.py`、
`data_deal_support/merge_box2poly_danger_to_dataset.py`。

第四类，抗漂移增强自动化。
新脚本 `improve/video_anti_unet_drift_augment_224x448.py` 把视频帧、扰动区域、U2Net 推理、合法帧过滤、批量增强、异步写盘串成一条自动流水线，专门用于构建“抗分割漂移”的训练样本。
核心收益是显著降低人工挑帧和人工造扰动的时间成本。

### 12:30 - 14:20 Jetson 部署细节（文档+程序+tools 对齐）

这一段我重点讲我们在 Jetson 上是如何把方案稳定跑起来的。

第一，统一入口。  
我们现在统一用 tools/workflow 目录下的脚本作为入口，而不是手敲长命令。  
日常实时推理入口是 run_realtime_infer.sh，三联画录制入口是 record_triple_preview.sh，本地稳定性回归入口是 run_local_stability_check.sh。

第二，模型与精度策略。
U2Net 用 FP16，Dinomaly 用 mixed，这是目前兼顾速度和稳定性的默认组合。   
脚本里 precision-mode 支持 full 和 mixed，mixed 会优先使用 dinomaly_runtime_mixed.trt。  

第三，尺寸一致性。  
这是最容易踩坑的地方。  
我们当前运行链路是 UNet 输入 320，Dinomaly 输入高 448、宽 224。  
这个尺寸必须和 TRT 引擎一致，不能只改环境变量不重建引擎，否则会出现 shape 不匹配报错。  
重建引擎时记得利用系统python，并且删除目标文件夹下同名引擎，不然会因为存在同名引擎而不重构引擎

第四，相机与视场策略。  
MVS 侧默认用 scale 模式，不走硬件 ROI 裁切。主要是因为床面板移动距离较长，所以使用全画幅    
核心原因是先保留全视场，再做软件等比归一化，避免中心裁切导致视野丢失。  
在程序里 normalize_frame_size 先判断比例一致再缩放，这个顺序是为了优先保留全画幅信息。

第五，主程序运行形态。  
camera_debug_infer.py 里支持 sync 和 async 两种后端，支持 double 和 triple 预览布局。  
同时支持 save-policy 的 manual、abnormal、all 三种保存策略。  
所以我们可以把同一个程序用于本地调试、现场巡检和留痕录制。  

第六，稳定性检测机制。  
我们不是只看单次结果，而是做重复运行稳定性检查。  
full-chain 稳定性模式会对同一批图片多次重跑，记录分数漂移、耗时和结果一致性。  
这保证我们每次模型、阈值、脚本改动后都能快速回归，不会靠主观观感判断。

第七，现场可操作命令。  
实时推理我们直接用：bash tools/workflow/run_realtime_infer.sh --precision-mode mixed。  
三联画录制用：bash tools/workflow/record_triple_preview.sh --duration 30 --fps 8   --precision-mode mixed。  
全链路稳定性回归用：bash tools/workflow/run_local_stability_check.sh --mode full-chain --show。  

第八，python-can 与接口号注意事项。  
我们在 AGX 上用 python-can 时默认走 can2 接口，所以接口号一定要对齐。  
如果机器上 can2 不存在，脚本里的 can 初始化会失败，表现为程序可跑但总线通信不通。  
组内复现时要先确认接口名，再决定是否传 --can-channel 覆盖默认值。  

第九，Jetson Nano 兼容限制。  
Nano 使用这套程序时，受 TensorRT 版本差异影响，当前不能使用 mixed 混合精度，只能走 FP32。  
另外 Nano 上目前这套流程还不能稳定进行 CAN 通讯，建议先关闭 CAN 通道，只保留视觉推理链路。  
需要抽象出一个can口。  

第十，输出与复盘材料。  
每次运行可以保存 overlay、mask、result.json 和 csv 日志。  
复盘时我们重点看 sample_score、阈值和状态变化，再结合图像定位误报与漏报原因。  


总结一下，这个项目的核心不是单点模型，而是工程闭环。  
通过 U-2-Net + Dinomaly 的组合策略，我们把复杂场景拆解成稳定 ROI 和异常判别两段。  
通过自动化脚本体系，我们把数据构建、标注转换、样本筛选和增强从高人工成本流程升级成可重复流水线。  




## 四、Jetson 讲解备用页（可直接贴PPT）

### 4.1 Jetson 统一工作流（建议作为一页）

1. 构建引擎：U2Net FP16 + Dinomaly mixed。
2. 启动实时：run_realtime_infer.sh。
3. 演示录制：record_triple_preview.sh。
4. 稳定回归：run_local_stability_check.sh。
5. 结果留痕：overlay + result.json + csv。

### 4.2 Jetson 常见坑（建议作为一页）

1. 只改尺寸变量不重建 TRT，引发 shape mismatch。
2. 相机误用 ROI 模式导致视场变窄、误判增加。
3. 混用旧阈值文件，导致状态判定漂移。
4. 只看单次推理，不做 full-chain 稳定性回归。
5. AGX 默认 python-can 走 can2，接口号配错会导致 CAN 不通。
6. Nano 因 TensorRT 版本限制不能走 mixed，只能 FP32，且当前不建议启用 CAN 通讯。

### 4.3 Jetson 运行口播版（20秒）

Jetson 部署我们已经做成统一脚本入口，平时主要跑 mixed 精度链路，兼顾速度和稳定性。
所有改动都通过 full-chain 稳定性脚本回归，确保不是偶然结果。
AGX 上 python-can 默认用 can2，要先确认接口号；Nano 目前按 FP32 视觉链路运行，不启用 CAN。
线上复盘统一看图像留痕加分数日志，定位问题会更快。

## 五、组内交流备用问题（围绕自动化价值）

1. 为什么强调自动化脚本？

因为这个项目瓶颈不在“能不能训练”，而在“能不能持续高质量产数和快速迭代”。自动化脚本直接降低了采集、筛选、组织数据的时间和错误率。

2. 为什么从 448x448 改到 224x448？

因为床面板 ROI 是高而不长，224x448 更匹配目标形态，减少黑边冗余，提升有效信息占比和速度表现。

3. 今天新进展如何定义成功？

成功标准是“抗漂移数据链路可稳定运行 + 训练链路可稳定复现 + 指标可持续对比”，而不是一次性偶然高分。

4. 为什么要保留脚本化而不是手工处理？

手工流程在样本规模变大时不可维护，脚本化才可以跨场景迁移、复现实验、做版本对比。
