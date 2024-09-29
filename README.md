# RamanNet 拉曼光谱分析系统

这是一个用于拉曼光谱数据分析的机器学习系统。该系统使用深度学习模型RamanNet来训练和预测拉曼光谱数据,并提供了一个用户友好的图形界面。

# RamanNet
详情搜索https://github.com/nibtehaz/RamanNet

git clone https://github.com/nibtehaz/RamanNet.git

## 功能特点

- 数据加载和可视化
- 模型训练与评估
- 模型预测
- 结果可视化(包括损失曲线、混淆矩阵等)
- 用户友好的图形界面

## 安装

1. 克隆此仓库:
   ```
   git clone https://github.com/Family2002/RamanNet-Gui.git
   ```

2. 安装依赖:
   ```
   pip install -r requirements.txt
   ```
或者下载应用程序
## 使用方法

运行主程序:
python begin.py

这将启动图形用户界面,您可以通过界面进行以下操作:

1. 加载训练数据
2. 设置训练参数
3. 训练模型
4. 加载预测数据
5. 使用训练好的模型进行预测
6. 查看预测结果和评估指标

## 文件结构

- `begin.py`: 主程序,包含图形界面的实现
- `codes/`:
  - `data_processing.py`: 数据预处理函数
  - `RamanNet_model.py`: RamanNet模型定义
  - `train_model.py`: 模型训练函数
- `data/`: 存放训练和预测数据的目录
  - `train_data.CSV`: 训练数据
  - `predict_data.CSV`: 预测数据
- `requirements.txt`: 项目依赖列表


## 数据格式

训练和预测数据应为CSV格式,其中:
- 第一列为类别标签
- 其余列为光谱数据

## 依赖

主要的依赖包括:
- numpy
- matplotlib
- PyQt5
- scikit-learn
- seaborn
- keras
- tensorflow

详细的依赖列表请参见`requirements.txt`文件。

## 模型架构

RamanNet模型使用了多个输入窗口来处理光谱数据,每个窗口通过一系列的全连接层、批归一化和LeakyReLU激活函数进行特征提取。最后,模型输出一个嵌入向量和分类结果。

## 注意事项

- 确保您的Python环境中安装了所有必要的依赖
- 训练可能需要一定时间,请耐心等待
- 对于大型数据集,请确保有足够的计算资源
- 模型的性能可能会因数据集的不同而有所变化

## 贡献

欢迎提交问题和拉取请求。对于重大更改,请先开issue讨论您想要改变的内容。

## 许可证

MIT License

Copyright (c) 2024 Family2002

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.