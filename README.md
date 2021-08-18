# S-DMM
This is a code sample for the paper "Deep Metric Learning-Based Feature Embedding for Hyperspectral Image Classification" [pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8887497)

# Requirements
Python 2.x/3.x 

Pytorch 0.4.1  

GPU

# Run the S-DMM
Please set your parameters in train.py or test.py before runing them. 

Training, run: python train.py 

Test, run: python test.py

# About datasets and checkpoints
The datasets are available in http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes  

The saving trainTestSplit and checkpoints of our experiments can be download at https://pan.baidu.com/s/1cXrVnLf662wtCW6xX3LWhg or at https://drive.google.com/drive/folders/1z2qDYLMdsVPk--Qhze-JKde9P4G4GOKw?usp=sharing

# Note
This is the code for the same-scene HSI classification; The code for cross-scene HSI classification is in https://github.com/szubing/ED-DMM-UDA

# Reference
This repository is contributed by [Bin Deng](https://bindeng.xyz/).
If you consider using this code or its derivatives, please consider citing:

```
@ARTICLE{Deng_HSI_2020,
  author={Deng, Bin and Jia, Sen and Shi, Daming},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Deep Metric Learning-Based Feature Embedding for Hyperspectral Image Classification}, 
  year={2020},
  volume={58},
  number={2},
  pages={1422-1435},
  doi={10.1109/TGRS.2019.2946318}}
```
