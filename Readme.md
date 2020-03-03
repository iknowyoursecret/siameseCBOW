siamese-CBOW模型：

# 项目结构树：
## 项目名：siameseCBOW

    ├─data
    │  └─dataset.py #数据导入
    ├─layer
    │  ├─averageLayer.py #加和平均
    │  ├─cosineLayer.py #余弦相似度计算
    │  └─embeddingLayer.py #嵌入层
    ├─model
    │  └─siameseCBOW.py #模型
    ├─output #训练结果保存目录
    ├─preprocess
    │  ├─generateWordDic.py #生成字典
    │  ├─getPosData.py #获取正例集索引列表
    │  └─sentenceprocess.py #句子处理
    ├─1606.04640.pdf  #siameseCBOW模型论文
    ├─config.py  #训练或测试配置文件
    ├─main.py #训练
    └─prediction.py #句子相似预测
    

# 使用

## 环境配置：
### 根据environment.yml创建环境
    conda env create -f environment.yml

上面命令会安装在conda默认的环境路径
如果要指定其它安装路径，使用-p选项

    conda env create -f environment.yml -p /home/user/anaconda3/envs/env_name
    

## 数据集：
    建议放在data目录下
    #在config.py 配置数据集路径
    
## 配置config.py：
    根据需要配置
    
## 训练：
    直接执行main.py
    
## 判断句子相似程度:
    运行predition.py
    
