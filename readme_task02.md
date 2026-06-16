# 病床异常检测
## 1. 总体架构
- **U²-Net** 进行分割，将无关部分裁掉（这里直接用黑色覆盖）
  >源码地址：https://github.com/xuebinqin/U-2-Net.git
- **Dinomaly** 对分割结果进行异常识别，判断是否有异物出现
  >源码地址：https://github.com/guojiajeremy/Dinomaly.git

## 2. 模型使用
1. 输入图像/视频 → U²-Net 分割病床有效区域，无关区域黑色覆盖
2. 分割后图像 → Dinomaly 异常检测，输出正常/异常结果
3. Jetson 部署：加载 ONNX 模型 + TensorRT 加速运行
4. 推理流程：先 test 获取阈值，再 pre 进行预测

## 3. 模型训练
整体分为两部分：U²-Net 分割训练 + Dinomaly 异常检测训练

## 4. 转为 ONNX 并验证
1. PyTorch .pth 权重 → 转换为 ONNX 格式
2. 运行对应测试脚本，验证分割/异常检测效果
3. 保存异常判断阈值为 json 文件

## 5. 部署在 Jetson 上并用 TensorRT 运行
1. 模型转换：ONNX → TensorRT 引擎
2. 部署环境配置，启用 GPU 加速推理
3. 轻量化部署：仅需模型文件 + 阈值 json + 少量验证数据
4. 实现端到端病床异物实时检测

## 6. 应用到新的场景进行迁移训练
### 材料准备
- 至少三段不同光照下的床 100mm 速度进出全过程视频
- 20 张异常图片，像素最好是 14 的倍数
- 曝光时间使用手动并记录下来

### 6.1 U²-Net 的迁移训练
1. 从新场景中选出一定数量（>36）的代表图片，可从视频中抽帧
   最好同时使用带有异常的图片和正常图片
2. 利用 labelme 进行待分割区域的标注
![fenge](./readme_image/fenge1_labelme.png)
3. 使用 labelme 环境运行 `./data_deal_support/json_to_dataset.py`
   进行 json 标注文件的转化，得到训练 U²-Net 所需要的掩码图像
4. 使用 `U-2-Net/u2net_train_second.py` 进行迁移训练，得到 .pth 权重文件
5. 使用 `U-2-Net/onxx_part/onx_change.py` 进行文件转化，得到 .onnx 文件
6. 使用 `U-2-Net/onxx_part/onxx_pre.py` 进行试验，确定 onnx 文件切割效果

### 6.2 Dinomaly 训练数据获取
从视频中获取所需要的图片帧用于训练
- 方法 1：直接使用 `data_deal_support/video_to_picture.py` 进行手动抽帧，然后使用 U²-Net 进行截取
- 方法 2：使用 `./data_deal_support/vedio_to_Dinomaly_train_data_auto.py`
  利用训练好的 U²-Net onnx 模型，和选取的用于 U²-Net 迁移训练的图片
  自行抽取有效帧，将符合标注的保留

**有效帧判断标准（从训练集掩码特征统计得到）：**
- 面积特征
- 形状特征：宽高比范围、最小外接矩形占比均值
- 质量特征：最大连通域占比最小值、密实度最小值
- 位置特征：中心距离最大值

过滤规则：
- 被床控制面板遮挡的部分移除
- 过近导致无法进行有效判断的部分帧移除
- 部分图片提取效果有误，invalid 里夹杂 valid，手动去除即可
![shujushaixuan](./readme_image/shujujishaixuan.png)
- 删除误检图片后，使用 `./data_deal_support/vedio_trans_to_train_good.py`
  进行有效图片转移，无效图片清理

### 6.3 Dinomaly 训练
1. 可以只使用 train/good 进行训练
2. 为便于后续获取异常判定阈值和检测效果，最好利用 labelme 进行 test 标注
![yicahng](./readme_image/yichanghuizhi1.png)
   标出异常所在，并按 6.1 流程生成 ground_truth 掩码图像

**Dinomaly 数据集结构：**
```
-data1.0
  -MR_table
    -ground_truth
      -danger
        | --001.png
        | --002.png
    -test
      -danger
        | --001.png
        | --002.png
      -good
        | --001.png
    -train
      -good
        | --001.png
        | --002.png
      -pre
        -pre
          | --001.png
          | --001.png
```

注意：
- test 中除了 good 外的文件夹要和 ground_truth 下的文件夹一一对应
- 图片名称也要一一对应
- 本项目一般只设置 danger 一个异常种类

训练与测试：
1. 配置路径，使用相对路径记得 cd 到对应目录
2. 运行 `./Dinomaly/dinomaly_mvtec_sep_train.py` 得到 .pth 权重文件
3. 使用 `./Dinomaly/inference7visualize_location_shape_rect.py` 进行 pytorch 预测
4. 使用 `./Dinomaly/onxx_part/onxx_change_opset13.py` 转换为 onnx
5. 使用 `./Dinomaly/onxx_part/onxx_test_time.py` 检测 onnx 效果

测试时参照模型使用：
- 先 `--phase test` 获取判断阈值，保存为 json 文件
- 再 `--phase pre` 根据阈值文件进行预测并推理是否异常
- Jetson 部署时，可只带有小规模数据集和 json 阈值文件，无需完整数据集