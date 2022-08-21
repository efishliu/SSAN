# SSAN
Self-Supervised Signed Graph Attention Network for Social Recommendation 论文源码

### 主要工作
* 我们提出了一种用于社交推荐 SSAN 的自监督签名图注意网络框架，通过引入用户的态度来构建高阶社交签名网络和自监督任务来聚合社交信息。
* 我们研究了签名网络在社交推荐中的作用，并设计了一种基于 LightGCN 和平衡理论的图卷积方法来聚合混合社交关系的图信息。
* 受社会理论的启发，我们设计了两个自监督信号来学习具有语义信息的社交图结构，并将自监督任务作为辅助任务集成到统一的推荐框架中。
* 实验结果证明了 SSAN 在几个真实数据集上的有效性。

### 源码介绍
SSAN 基于开源推荐代码QRec二次开发，核心的代码模块为：[SSAN](./SSAN/model/ranking/SSAN.py)  
配置文件：[SSAN.conf](./SSAN/config/SSAN.conf)

项目运行： `python main.py`

更多使用信息参考 [SSAN](https://www.showdoc.com.cn/QRecHelp/7341995904538619)
<h3>Reference</h3>
<p>[1]. Tang, J., Gao, H., Liu, H.: mtrust:discerning multi-faceted trust in a connected world. In: International Conference on Web Search and Web Data Mining, WSDM 2012, Seattle, Wa, Usa, February. pp. 93–102 (2012)</p>
<p>[2]. Massa, P., Avesani, P.: Trust-aware recommender systems. In: Proceedings of the 2007 ACM conference on Recommender systems. pp. 17–24. ACM (2007) </p>
<p>[3]. G. Zhao, X. Qian, and X. Xie, “User-service rating prediction by exploring social users’ rating behaviors,” IEEE Transactions on Multimedia, vol. 18, no. 3, pp. 496–506, 2016.</p>
<p>[4]. Iván Cantador, Peter Brusilovsky, and Tsvi Kuflik. 2011. 2nd Workshop on Information Heterogeneity and Fusion in Recom- mender Systems (HetRec 2011). In Proceedings of the 5th ACM conference on Recommender systems (RecSys 2011). ACM, New York, NY, USA</p>
<p>[5]. Yu et al. Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation, WWW'21.</p>
<p>[6]. He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, SIGIR'20.</p>
<h2>Acknowledgment</h2>
<p>This project is supported by the Responsible Big Data Intelligence Lab (RBDI) at the school of ITEE, University of Queensland, and Chongqing University.</p>
