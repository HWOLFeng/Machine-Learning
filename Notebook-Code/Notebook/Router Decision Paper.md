# Router Decision Paper

# 基于P4和机器学习的路由选择方案探讨

出处：邮电设计技术 0.707

年份：2018 

作者：李倩 

团队：北邮 

关键词：强化学习、路由选择

概述：基于P4和强化学习的路由选择方案，路由决策，最大链路利用率的比值，基于PPO强化学习算法与ECMP算法比较。

研究的问题：现有方法及优缺点：监督学习、非监督学习无法做到最优化主体的表现。

论文提出的思路和方法，及优缺点

论文使用的数据集：自己采集

工具：P4数据层采集工具；强化学习算法：PPO

论文使用的实验方法：

![](./Router Decision/P4.png)

# Learning To Route

出处：Proceeding HotNets-XVI Proceedings of the 16th ACM Workshop on Hot Topics in Networks Pages 185-191

年份：2017

作者：Asaf Valadarsky

团队：Hebrew University of Jerusalem

关键词：

Reinforcement Learning, Intradomain Route Configuration, Demand Matrix, Routing Strategy, Softmin

概述：

域内流量工程的经典设置。 有关数据驱动路由，应用ML（特别是深度强化学习）可产生高性能，并且是进一步研究的有希望的方向。 

研究的问题：

ML-guided routing the optimization of routing within a single, self-administered network.

(1) How should routing be formulated as an ML problem? 

(2) What are suitable representations for the inputs and outputs of learning in this domain? 

现有方法及优缺点：

- supervised learning ，supervised learning might be ineffective if the traffic conditions do not exhibit very high regularity

论文提出的思路和方法：

- a model for data-driven (intradomain) routing that builds on the rich body of literature on intradomain TE and (multicommodity) flow optimization. 
- For Question (2)，devise methods for constraining the size of the output without losing “too much” in terms of routing expressiveness，leverage ideas from the literature on **hop-by-hop**

论文使用的数据集：

工具：P4数据层采集工具；强化学习算法：PPO

论文使用的实验方法：

（1）**Network**：a capacitated directed graph $G=(V, E, c)$，where V and E are the vertex and edge sets, respectively, and $c: E \rightarrow \mathbb{R}^{+}$ assigns a capacity to each edge. 

（2）**Routing**：$\Gamma(v)$denote vertex v’s neighboring vertices in G.$\mathcal{R}_{v,(s, t)}: \Gamma(v) \rightarrow[0,1]$，such that $\mathcal{R}_{v,(s, t)}(u)$ is the **fraction** of traffic from $s$ to $t$ **traversing** v that v forwards to its neighbor u. $\sum_{u \in \Gamma(v)} \mathcal{R}_{v,(s, t)}(u)=1$（no traffic is blackholed at a non-destination），and，$\sum_{u \in \Gamma(v)} \mathcal{R}_{t,(s, t)}(u)=0$（all traffic to a destination is absorbed at that destination ）

（3）**Induced flows of traffic**：***demand matrix*** (DM) （the realistic scenario in which the DM is not known beforehand.）D is a $n×n$ ，$D_{i, j}$specifies the **traffic demand** between source i and destination j。（好像是每个vertex的流量，可以层层递推）

（4）**evaluate traffic flow**：**minimizing link (over) utilization** 。The link utilization under a specific multicommodity flow $f$ is $\max _{e \in E} \frac{f_{e}}{c(e)}$， $f_e$ is the total amount of flow traversing edge $e$ under flow $f$ 。

（5）**Routing future traffic demands **：the routing strategy $\mathcal{R}^{(t)}$ for that epoch is decided ，can depend only on the history of ***past*** traffic patterns and routing strategies (from epochs 1, . . . , t − 1). 

未来可能的应用方向：

(1) extending our approach to other routing contexts

(2) examining other performance metrics

(3) identifying better supervised learning approaches to traffic-demand estimation

(4) scaling reinforcement learning in this context, and beyond.