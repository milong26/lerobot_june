这个文件夹的功能是检查数据集每一集经过embedding之后还有没有差别
see_diff的结果是
===========================================================
方案1: 逐帧余弦相似度 & 欧氏距离分析
============================================================
  总帧数: 61
  初期(0-33%)平均余弦相似度: 1.0000
  中期(33-66%)平均余弦相似度: 1.0000
  后期(66-100%)平均余弦相似度: 1.0000
  初期→后期变化: -0.0000
  可区分度(初期距离/后期距离): 0.33
  ❌ 模型对位置变化不敏感
  图表已保存: /data/zhonglinye/jun/lerobot/personal/work2/see_embedding/results/01_cosine_similarity.png

============================================================
方案2: PCA降维可视化
============================================================
  PCA解释方差比: [0.5881626  0.13492095]
  总解释方差: 72.31%
  图表已保存: /data/zhonglinye/jun/lerobot/personal/work2/see_embedding/results/02_pca_visualization.png
  PCA空间中起点距离: 0.0166
  PCA空间中终点距离: 0.0365
  起点/终点距离比: 0.45

============================================================
方案3: 交叉时间步相似度矩阵
============================================================
  降采样: 61→60, 64→60
  图表已保存: /data/zhonglinye/jun/lerobot/personal/work2/see_embedding/results/03_similarity_matrix.png
  Diagonal avg similarity: 1.0000
  Offset-1 avg similarity: 1.0000
  Offset-5 avg similarity: 1.0000

============================================================
方案4: 分类可区分度分析
============================================================
  Early: Accuracy = 70.00% (+-18.71%)
    Model has some distinguishing ability
  Mid: Accuracy = 87.50% (+-15.81%)
    Model can well distinguish the two episodes
  Late: Accuracy = 70.28% (+-22.53%)
    Model has some distinguishing ability
  图表已保存: /data/zhonglinye/jun/lerobot/personal/work2/see_embedding/results/04_classification_accuracy.png

============================================================
方案5: t-SNE降维可视化
============================================================
  图表已保存: /data/zhonglinye/jun/lerobot/personal/work2/see_embedding/results/05_tsne_visualization.png

============================================================
分析完成!
所有结果已保存到: /data/zhonglinye/jun/lerobot/personal/work2/see_embedding/results
============================================================


所以写一个新的代码
see_embedding_deep.py
