# ML Primer

- 分类问题, 一个线性函数"划分"
- 怎么衡量"划分"的好坏? naive的可以是和"分割线"的距离, 总距离越小越好
    * 所以引入loss函数, 如常见的sigmod: 输入越大越接近1, 输入越小(负)越接近0
    * 但在实际使用中会对sigmod取log取负: $-log(sigmod(x))$这样一来loss越接近0表示效果越好
- 引入非线性函数
- 如何更新参数?
    * 一种naive的方法就是随机枚举
    * 梯度怎么用? **对参数(权重)求导**, 可以知道参数对loss的影响, 从而调整参数: $w = w+\Delta w$
- 如何计算梯度: 自动微分
    * 一种naive的方法可以是用梯度的定义: 如$d = (y1-y2) / (x1-x2)$
    * 链式法则: 每个函数都是由一个个基础的函数组成
        + 对于任意函数$f = h(g(z), i(k))$都可以抽象为
            + $f'_x = d \times (g'_x(z) + i'_x(k))$
            + 其中d表示当前函数的导: $d = h'_x(x)$, 此时h的输入是纯粹而直接的x
            + g和i再用同样的方法展开
        + 利用拓扑排序从叶子节点开始计算
- 反向传播: 如何使用梯度
    * 什么叫梯度累积?
    * TODO

## 链式法则和反向传播

TODO: 

- 设计
- 梯度累积
    * TODO:
        + `accumulate_derivative`
- TODO: 入口
    * 一个算子(Function)和产生一个(或多个)结果(Scalar/Tensor)
        + 从最终输出的Scalar/Tensor出发反向传播: `Scalar:backword`
            + `Scalar:backword()` -> `backpropagate()` -> `chain_rule()`






