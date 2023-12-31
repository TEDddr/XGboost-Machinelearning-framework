# XGboost-Machinelearning-framework
一个集合了预测算法、优化算法的AI机器学习框架~
本项目设计了一个结合机器学习/深度学习与优化算法的AI预测模型框架，并开发了GUI让整体操作更加简易，更容易上手。如图所示：
(8/16更新)![image](https://github.com/TEDddr/XGboost-Machinelearning-framework/assets/130724106/ac108e6c-6f9f-49f9-93be-caa79cdedad4)

 ![image](https://github.com/TEDddr/XGboost-Machinelearning-framework/assets/130724106/96ed4bb3-4381-4e8e-8d72-0d5b665e7388)
 

如需参考，请自行下载或点击.py文件进行浏览。下载后，个别安装包需要自行安装：xgboost、pandas、numpy和pyqt5等库。进去哪里标红就安装对应的库就行了。这样方便下载


以下是一些基础知识的介绍，题主也是一名小菜，如有不足之处望多多谅解。
工业上一般拟合多用机器学习，因为可以一定程度解释模型，但在一些无需特征工程的场合，也可以使用深度学习等。解释性和精度的取舍可以根据自己的需求来，如下图所示：
 ![image](https://github.com/TEDddr/XGboost-Machinelearning-framework/assets/130724106/11eff20e-adf7-4f72-9bb8-5b5f8adbaccf)

拟合效果和模型验证结果都很好
如下表所示：
 ![image](https://github.com/TEDddr/XGboost-Machinelearning-framework/assets/130724106/90137789-77a0-4c60-a397-1b668b6fa54d)

近似公式：
线性回归模型简单拟合程度不高，多项式回归拟合程度尚可但模型稍显复杂，机器学习XGBoost拟合程度高但模型复杂（无法用公式解释，仅可解释不同特征的重要性）。
结论：去掉噪点后在做回归分析，最终目的是解释模型，结合相关性，整体符合。预测还是用XGBoost等预测算法。

现阶段：
1.分析并获取了关键参数的重要程度
2.初步建立预测模型（用于后期优化）框架：
	建立特征模型：相关性分析并降噪
	建立预测模型：机器学习预测并融合寻优算法
	确立特征工程：拟合近似公式
3.	建立了特征工程用于解释现有参数对输出特征的影响程度

后期预计可优化点：
1.	提高模型精度：新算法、数据集处理等手段
2.	增加模型泛化能力：添加不同型号但类型一样的模型，如高\低碳钢的不同版样组合。
3.	融入持续学习的思想：基于现有模型，添加新的数据集理论上可强化上述两点，同时可解决训练模型的“灾难性遗忘”问题。只需要花较小的代价即可优化模型。
