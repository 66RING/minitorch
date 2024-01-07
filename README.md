# minitorch

## a better roadmap

a better roadmap for engineering

- scalar


## ML Primer

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
                + NOTE: 在实际代码中, d用来表示其他模块的导, 计算当前算子的反向时会传入
            + g和i再用同样的方法展开
        + 利用拓扑排序从结果节点向叶子节点链式求导
- 反向传播: 如何使用梯度
    * 什么叫梯度累积?
    * TODO

## 链式法则和反向传播

TODO: 

- 设计
    * 每个Scalar/Tensor的产生都会记录产生的过程和使用的上下文
    * 每个算子产生新的Scalar/Tensor, 并保存对应记录
- 梯度累积
    * TODO:
        + `accumulate_derivative`
- 入口: 输出的结果Scalar/Tensor
    * 一个算子(Function)和产生一个(或多个)结果(Scalar/Tensor), 该结果记录了它的产生过程(history)
        + 从最终输出的Scalar/Tensor出发反向传播: `Scalar:backword`
            + `Scalar:backword()` -> `backpropagate()` -> `chain_rule()`
- 求导的链式法则
    * 流程
        1. 根据结果Scalar/Tensor执行拓扑排序, 从结果节点向叶子节点链式求导
            - h = f(z1, z2), z = g(x)
                * dh = (dfz1 + dfz2) * dg
                    + aka, dh = dout * dg
                * 又因为加法分配律, 所以实际的计算过程应该是这样:
                    + dh = dfz1 * dg + dfz2 * dg
                    + 即, 分解成两两条路径相加, 最后通过accumulate来合并最终结果
            - e.g. h = mul(add(x, y), add(x, z))
                * dh = (dmul1 + dmul2) * dadd
    * e.g. 手动根据计算图计算梯度

## scalar

> **Scalar a + b产生新的记录了历史信息的Scalar(该Scalar由哪个算子怎么产生)**

为了不破坏python底层的算子(加减乘除)，我们要记录网络的计算图那就需要自己封装一层算子: `ScalarFunction`。对于那么对于每个变量就是Scalar。

一种链式调用

```python
def __op__(self) -> Scalar
```

如`Scalar a`加`Scalar b`: `a + b`会返回一个新的`Scalar(res, history)`，并在这个新的Scalar中记录计算结果和操作历史(操作类型cls, ctx信息(反向传播需要的一些记录), 输入)。

```python

@dataclass
class ScalarHistory:
    """
    Attributes:
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()
```

形如

```
                 Scalar (add, (a, b))
                    ^
                    |
Scalar (None, (a,))    Scalar (None, (b,))
```














