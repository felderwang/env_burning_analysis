By FelderWang, whydf@outlook.com

## 总体介绍

这是分析的1.0版本，基本功能都已经完成了。还有一些代码整理的工作，以及部分辅助功能待实现。

### version 3.0 (2021.05.10)

输入文件分为两部分， 一个是工厂排放量，另一个是工厂的年燃量、理论烟气量及燃料类型，非常简单

## 结构说明

### 处理脚本为 `dealer.py`。

所需的一些数据在`datas.json`中，其具体含义见下。

具体操作详见脚本中注释。目前的版本不进行筛选，对于`0`数值使用线性插值的办法填补。

### 处理所需的数据存放在 `datas.json` 中，其中各字段含义为：

* "company_name_list"：需要处理的公司列表。公司列表可使用 `company_list_creater.py` 生成，之后放到 `json` 文件中即可。
* "file_list": 需要处理的源文件路径列表。
* "table_column_dict"：源文件列名的映射。由于不同源文件列名可能会不一样，因此要改的话改这个映射即可，无需在 `dealer.py` 源码中修改
* "burn_type_dict"：最后计算时候需要再乘一个的系数。
* "flow_exception_name_list"：在给的 `9-12.xlsx` 中，流量该列，有的值是 `无`，需要先将其转换为 `0`。当然也可以直接改源 excel 文件中的值。

## 运行方法

先将所需数据在`datas.json`中修改好。然后执行：

`python dealer.py`

## Requirement

建议使用 anaconda 管理 Python

以下为所需包，具体视平台而定

* pandas
* numpy
* scipy
* matplotlib
