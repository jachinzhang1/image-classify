# 图像分类训练框架

这是一个综合性的基于GUI的图像分类模型训练和测试框架。它支持多种数据集和模型架构，提供易于使用的配置和可视化界面。

## 如何运行

从终端运行应用程序：

```bash
python main_ui.py
```

这将打开包含三个选项卡的主GUI：
- **训练**：用于使用各种数据集和架构训练模型
- **数据集测试**：用于在完整数据集上测试模型
- **单图像测试**：用于使用训练好的模型对单个图像进行分类

## 模型训练

训练界面提供了一种用户友好的方式来训练图像分类模型：

### 主要训练功能

1. **数据集选择**：从支持的数据集中选择（CIFAR-10、CIFAR-100、MNIST）
2. **模型选择**：从各种CNN架构中选择
3. **训练参数**：
   - 训练轮数（epochs）
   - 批次大小（batch size）
   - 学习率（learning rate）
   - 早停耐心值（当性能不再提升时自动停止训练）
4. **实时进度**：通过进度条监控训练进度
5. **自动保存**：自动保存最佳模型

### 输出文件

训练完成后，将生成以下文件：
- `./ckpts/<dataset>/<model>/latest.pth`：训练过程中发现的最佳模型
- `./ckpts/<dataset>/<model>/<timestamp>/model.pth`：同样的最佳模型，使用时间戳存储以便版本控制

### 早停机制

框架实现了早停机制以防止过拟合：
- 如果性能在指定的耐心轮数内没有提高，训练将自动停止
- 保存的是训练过程中遇到的最佳模型，而不是最终模型

## 添加新模型

要向框架添加新的模型架构：

1. **添加模型代码**：
   - 将模型实现放在`model/`目录中（例如，`model/my_new_model.py`）
   - 确保模型接受`num_classes`参数以处理不同的数据集

2. **在配置中注册**：
   - 打开`options.yaml`
   - 将模型名称添加到`model_types`列表中：
     ```yaml
     model_types:
       - attention_cnn
       - resnet18
       - my_new_model  # 你的新模型
     ```

3. **更新训练逻辑**：
   - 修改`train.py`中的模型初始化代码：
     ```python
     if cfg.selected_model == "attention_cnn":
         model = AttentionCNN(num_classes=cfg.num_classes).to(device)
     elif cfg.selected_model == "my_new_model":
         # 在文件顶部导入你的模型
         from model.my_new_model import MyNewModel
         model = MyNewModel(num_classes=cfg.num_classes).to(device)
     # ... 其他模型 ...
     else:
         raise ValueError("不支持的模型类型。")
     ```

4. **更新测试逻辑**：
   - 类似地，更新`test.py`和`single_image_test.py`中的模型初始化

## 添加新数据集

要添加对新数据集的支持：

1. **添加数据集配置**：
   - 打开`options.yaml`
   - 在`datasets`部分添加你的数据集：
     ```yaml
     datasets:
       cifar10:
         path: ./data/cifar10
         num_classes: 10
         description: "CIFAR-10图像分类数据集，包含10个类别"
       # 添加你的新数据集
       my_new_dataset:
         path: ./data/my_new_dataset
         num_classes: 5  # 你的数据集中的类别数
         description: "你的数据集描述"
     ```

2. **更新数据集处理逻辑**：
   - 打开`dataset.py`
   - 为你的数据集添加适当的变换：
     ```python
     my_new_dataset_transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(mean=[...], std=[...])
     ])
     ```
   - 更新`get_dataset_and_transform`函数：
     ```python
     elif dataset_name == "my_new_dataset":
         # 对于标准PyTorch数据集
         return datasets.ImageFolder(
             root=data_root, transform=my_new_dataset_transform
         ), my_new_dataset_transform
         # 或者对于自定义数据集，实现你自己的加载逻辑
     ```

3. **添加类别标签**：
   - 对于单图像测试，更新`single_image_test.py`中的`dataset_classes`：
     ```python
     "my_new_dataset": {
         0: "类别1", 1: "类别2", # 等等
     }
     ```