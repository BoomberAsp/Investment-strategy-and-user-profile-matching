# 分层贝叶斯径向惩罚余弦相似度：理论推导

## 1. 问题背景

现有方案使用径向惩罚余弦相似度：

\[
sim(u,s)
=
\frac{u\cdot s}{\|u\|\|s\|}
\times
\exp\left(
-\lambda
\left|
\log\frac{\|u\|}{\|s\|}
\right|
\right)
\]

其中：

- \(u\)：用户画像向量；
- \(s\)：策略画像向量；
- \(\cos(u,s)=\frac{u\cdot s}{\|u\|\|s\|}\)：方向一致度；
- \(d_r(u,s)=\left|\log\frac{\|u\|}{\|s\|}\right|\)：径向距离；
- \(\lambda>0\)：径向惩罚强度。

该公式的直觉是：

\[
sim=\text{方向相似度}\times\text{模长匹配度}
\]

如果两个对象方向一致但模长差距很大，纯余弦会给出很高分，而径向惩罚会降低分数。

现有方案的问题是：所有用户共用同一个固定 \(\lambda\)。这隐含了一个强假设：

> 不同类型投资者对模长差异的敏感度完全相同。

这个假设并不稳。ETF 用户、长线用户、短线投机用户、高频用户对“行为强度差异”的容忍度可能不同。因此，\(\lambda\) 不应只是全局超参数，而应成为可由数据学习的行为敏感度参数。

---

## 2. 径向距离的含义

定义：

\[
d_r(u,s)
=
\left|
\log\frac{\|u\|}{\|s\|}
\right|
\]

它有三个重要性质。

第一，尺度对称：

\[
\left|\log\frac{10}{20}\right|
=
\left|\log\frac{20}{10}\right|
\]

也就是说，用户模长是策略的一半，和策略模长是用户的一半，惩罚强度相同。

第二，它衡量的是倍数差异，而不是绝对差异。模长从 10 到 20 和从 50 到 100，径向距离相同，因为它们都是 2 倍差异。

第三，当 \(\lambda=1\) 时：

\[
\exp\left(
-\left|\log\frac{\|u\|}{\|s\|}\right|
\right)
=
\frac{\min(\|u\|,\|s\|)}{\max(\|u\|,\|s\|)}
\]

更一般地：

\[
\exp(-\lambda d_r)
=
\left(
\frac{\min(\|u\|,\|s\|)}
{\max(\|u\|,\|s\|)}
\right)^\lambda
\]

所以 \(\lambda\) 控制的是“模长比例差异”被放大的程度。

---

## 3. 为什么可以对 \(\lambda\) 做贝叶斯建模

径向惩罚项为：

\[
\exp(-\lambda d_r)
\]

这和指数分布的核函数一致：

\[
p(d_r\mid \lambda)
=
\lambda \exp(-\lambda d_r),
\quad d_r\ge 0
\]

也就是说，可以把匹配对象之间的径向差异 \(d_r\) 理解为一个非负观测量，而 \(\lambda\) 是该差异的 rate parameter。

如果匹配对象本来应该相似，那么它们的 \(d_r\) 应该偏小；如果某一类投资者对模长差异非常敏感，那么该类的 \(\lambda\) 应该较大。

因此：

- \(\lambda\) 大：径向差异稍大就被强烈惩罚；
- \(\lambda\) 小：允许较大的径向差异；
- 不同用户群体可以拥有不同的 \(\lambda\) 分布。

这就是分层贝叶斯建模的入口。

---

## 4. Gamma 先验的合理性

因为 \(\lambda\) 是惩罚强度 / rate parameter，必须满足：

\[
\lambda>0
\]

Gamma 分布的支持域正好是：

\[
(0,\infty)
\]

采用 rate 参数化：

\[
\lambda \sim Gamma(a,b)
\]

密度为：

\[
p(\lambda)
=
\frac{b^a}{\Gamma(a)}
\lambda^{a-1}
\exp(-b\lambda)
\]

其中：

\[
E[\lambda]=\frac{a}{b}
\]

\[
Var(\lambda)=\frac{a}{b^2}
\]

因此：

- \(a/b\) 表示平均径向敏感度；
- \(a/b^2\) 表示该敏感度的不确定性；
- \(a,b\) 可以被解释为群体层面的行为风格参数。

Gamma 先验的另一个优势是与指数似然共轭，能得到闭式后验。

---

## 5. 单群体贝叶斯更新

先看最简单情况：所有用户共享一个未知 \(\lambda\)。

假设观测到 \(n\) 个正匹配样本的径向距离：

\[
d_1,d_2,\ldots,d_n
\]

并设：

\[
d_m\mid\lambda \sim Exponential(\lambda)
\]

似然为：

\[
p(d_1,\ldots,d_n\mid\lambda)
=
\prod_{m=1}^n
\lambda \exp(-\lambda d_m)
\]

\[
=
\lambda^n
\exp\left(-\lambda\sum_{m=1}^n d_m\right)
\]

先验为：

\[
\lambda\sim Gamma(a_0,b_0)
\]

则后验为：

\[
\lambda\mid data
\sim
Gamma
\left(
a_0+n,\,
b_0+\sum_{m=1}^n d_m
\right)
\]

这说明：

- 正匹配样本越多，后验越稳定；
- 正匹配样本的径向距离越小，后验 \(\lambda\) 越大；
- 正匹配样本的径向距离越大，后验 \(\lambda\) 越小。

这正符合业务直觉：如果一类匹配样本天然模长接近，那么模长差异就应被严惩；如果它们天然模长波动大，就不应过度惩罚。

---

## 6. 分层贝叶斯模型

现在引入用户行为群体：

\[
z_i\in\{1,\ldots,K\}
\]

其中 \(z_i\) 表示用户 \(i\) 的行为风格，例如：

- 长线分散；
- 长线集中；
- 短线高频；
- 短线投机；
- 中线均衡；
- 被动 ETF。

### 6.1 硬分类版本

如果用户群体已知，例如合成实验中的 T1-T6，则：

\[
z_i=k
\]

对每个群体 \(k\)，设：

\[
\lambda_k \sim Gamma(a_{0k}, b_{0k})
\]

若群体 \(k\) 中观测到的正匹配径向距离集合为：

\[
\mathcal{D}_k
=
\{d_{ij}: y_{ij}=1,z_i=k\}
\]

则：

\[
\lambda_k\mid \mathcal{D}_k
\sim
Gamma
\left(
a_{0k}+n_k,\,
b_{0k}+\sum_{d\in\mathcal{D}_k}d
\right)
\]

其中 \(n_k=|\mathcal{D}_k|\)。

### 6.2 软分类版本

现实用户往往不是纯粹的一类。可以让用户属于各群体的概率为：

\[
\pi_{ik}=P(z_i=k\mid x_i)
\]

例如：

\[
P(z_i=\text{长线})=0.5
\]

\[
P(z_i=\text{ETF})=0.3
\]

\[
P(z_i=\text{投机})=0.2
\]

此时第 \(k\) 类的有效样本数为：

\[
n_k^{eff}
=
\sum_{i,j}\pi_{ik}y_{ij}
\]

径向距离总和为：

\[
S_k^{eff}
=
\sum_{i,j}\pi_{ik}y_{ij}d_{ij}
\]

后验为：

\[
\lambda_k\mid data
\sim
Gamma
\left(
a_{0k}+n_k^{eff},\,
b_{0k}+S_k^{eff}
\right)
\]

这样可以避免“先硬分类再匹配”导致的错误传递。

---

## 7. 贝叶斯预测惩罚项

预测时，不能只把 \(\lambda_k\) 替换成后验均值。更贝叶斯的做法是对惩罚项本身取后验期望。

对某个群体 \(k\)，后验为：

\[
\lambda_k\mid data\sim Gamma(a_k,b_k)
\]

需要计算：

\[
E[\exp(-\lambda_k d)]
\]

由 Gamma 分布的拉普拉斯变换可得：

\[
E[\exp(-\lambda_k d)]
=
\left(
\frac{b_k}{b_k+d}
\right)^{a_k}
\]

推导如下：

\[
E[\exp(-\lambda d)]
=
\int_0^\infty
\exp(-\lambda d)
\frac{b^a}{\Gamma(a)}
\lambda^{a-1}
\exp(-b\lambda)
d\lambda
\]

\[
=
\frac{b^a}{\Gamma(a)}
\int_0^\infty
\lambda^{a-1}
\exp(-(b+d)\lambda)
d\lambda
\]

\[
=
\frac{b^a}{(b+d)^a}
=
\left(\frac{b}{b+d}\right)^a
\]

因此，群体 \(k\) 的贝叶斯径向惩罚为：

\[
Penalty_k(d)
=
\left(
\frac{b_k}{b_k+d}
\right)^{a_k}
\]

若用户有软群体概率 \(\pi_{ik}\)，则：

\[
Penalty_i(d)
=
\sum_{k=1}^K
\pi_{ik}
\left(
\frac{b_k}{b_k+d}
\right)^{a_k}
\]

最终贝叶斯相似度为：

\[
sim_B(u_i,s_j)
=
\cos(u_i,s_j)
\times
\sum_{k=1}^K
\pi_{ik}
\left(
\frac{b_k}{b_k+d_{ij}}
\right)^{a_k}
\]

其中：

\[
d_{ij}
=
\left|
\log\frac{\|u_i\|}{\|s_j\|}
\right|
\]

这就是分层贝叶斯版径向惩罚余弦相似度。

---

## 8. 与固定 \(\lambda\) 方案的关系

固定 \(\lambda\) 方案为：

\[
sim_{fixed}
=
\cos(u,s)\exp(-\lambda d_r)
\]

贝叶斯方案为：

\[
sim_B
=
\cos(u,s)E[\exp(-\lambda d_r)\mid data]
\]

如果后验非常集中在某个 \(\lambda^\*\) 附近，那么：

\[
E[\exp(-\lambda d_r)\mid data]
\approx
\exp(-\lambda^\* d_r)
\]

此时贝叶斯方案退化为固定 \(\lambda^\*\) 的径向惩罚余弦。

因此，固定 \(\lambda\) 是贝叶斯方案的特例。贝叶斯方案的新增价值是：

1. 不同群体可以有不同 \(\lambda_k\)；
2. 样本少的群体会保留更大的不确定性；
3. 可以输出 \(\lambda_k\) 的后验区间；
4. 可以判断“模长敏感度差异”本身是否有统计证据。

---

## 9. 与项目代码的对应关系

当前项目中，固定径向惩罚余弦在 `pipeline.py` 中实现：

```python
def compute_radial_penalty_cosine(user_pca, strategy_pca, lam=LAMBDA):
    cos_sim = cosine_similarity(user_pca, strategy_pca)
    user_norms = np.linalg.norm(user_pca, axis=1, keepdims=True)
    strategy_norms = np.linalg.norm(strategy_pca, axis=1, keepdims=True).T
    log_ratio = np.abs(np.log(user_norms / strategy_norms))
    radial_penalty = np.exp(-lam * log_ratio)
    return cos_sim * radial_penalty
```

贝叶斯版本可以保持前半部分不变，只替换惩罚项：

```python
cos_sim = cosine_similarity(user_pca, strategy_pca)
d_r = abs(log(norm_user / norm_strategy))

penalty = sum_k pi_ik * (b_k / (b_k + d_r)) ** a_k
sim = cos_sim * penalty
```

工程上建议新增后端：

```text
app/services/backends/bayesian_radial.py
```

而不是直接改现有 `StatisticalBackend`。这样可以保留以下并行比较：

- `statistical`：固定 \(\lambda\) 径向惩罚；
- `bayesian_radial`：分层贝叶斯自适应 \(\lambda\)；
- `lstm`：序列风格嵌入；
- `fusion`：统计 + LSTM 融合。

---

## 10. 第一版实现建议

### 10.1 不建议一开始上完整 MCMC

当前项目依赖中没有 PyMC / NumPyro。并且真实用户样本目前只有 3 个，直接做完整 MCMC 意义有限。

第一版建议做闭式 Empirical Bayes：

1. 用现有特征空间给用户分群；
2. 用正匹配样本的径向距离更新每个群体的 Gamma 后验；
3. 用后验预测惩罚项计算相似度；
4. 与纯余弦、欧氏距离、固定 \(\lambda\) 径向惩罚、LSTM 进行比较。

### 10.2 后续升级版本

如果后续有真实用户反馈，例如点击、订阅、持有、满意度、专家标注，可以升级为监督式贝叶斯排序模型：

\[
y_{ij}\sim Bernoulli(p_{ij})
\]

\[
logit(p_{ij})
=
\eta_0
+\eta_1\cos(u_i,s_j)
-\lambda_i d_{ij}
+\eta_2 R_{ij}
\]

其中：

- \(y_{ij}=1\)：用户接受或专家认为匹配；
- \(R_{ij}\)：收益、回撤、风险适当性等控制变量；
- \(\lambda_i\mid z_i=k\sim Gamma(\alpha_k,\beta_k)\)。

这个版本更严格，但需要真实标签支撑。

---

## 11. 可解释输出

贝叶斯方案可以输出比固定 \(\lambda\) 更多的信息：

| 输出 | 含义 |
|---|---|
| \(\pi_{ik}\) | 用户属于各行为群体的概率 |
| \(E[\lambda_k]\) | 群体平均径向敏感度 |
| 95% credible interval | 群体敏感度的不确定性 |
| \(Penalty_i(d)\) | 用户级贝叶斯径向惩罚 |
| posterior shrinkage | 小样本群体向总体均值收缩的程度 |

示例解释：

> 该用户有 62% 概率属于短线高频风格，28% 概率属于短线投机风格。模型估计短线群体对模长差异的容忍度较高，因此本次推荐中径向惩罚较轻；最终得分主要由方向相似度决定。

---

## 12. 结论

分层贝叶斯方案不是简单地把固定 \(\lambda\) 写成随机变量，而是把“对模长差异的敏感度”从全局超参数提升为用户行为风格的一部分。

固定径向惩罚回答的是：

> 模长差异是否有用？

分层贝叶斯径向惩罚进一步回答：

> 不同类型投资者对模长差异的敏感度是否不同？

如果实验表明不同群体的 \(\lambda\) 后验存在稳定差异，并且贝叶斯相似度在分类、排序或推荐任务上优于固定 \(\lambda\)，则该方案可以作为“行为自适应相似度框架”的理论核心。
