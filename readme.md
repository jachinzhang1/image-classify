## 如何运行

终端运行命令：

```bash
python main_ui.py
```

## 如何添加模型

如果你想在该项目中添加新的模型结构，可按以下步骤进行（假设你要添加的模型代码为`model_A.py`，名称为`modelA`）：
1. 将结构相关的代码放在目录`model`中
2. 在`options.yaml`配置文件的`model_types`属性列表里添加模型的名称
   
   ```yaml
   model_types:
     - attention_cnn
     - modelA
   ```

   `Training`界面的模型种类选择列表是从上述数据加载的。当你完成在这里的添加后，就可以在GUI的选择栏里选择该模型
3. 在`train.py`的`main`函数中，对`selected_model`的条件分支选项进行扩充
   
   ```python
   if cfg.selected_model == "attention_cnn":
       model = AttentionCNN(num_classes=cfg.num_classes).to(device)
   elif cfg.selected_model == "modelA":   # 扩充的 modelA 分支
       model = ModelA(...).to(device)
   else:
       raise ValueError("Unsupported model type.")
   ```

   同理，在`test.py`的`main`函数中也要进行相关扩充