# Implicit Neural Representations

- 主要来自[Awesome Implicit Neural Representations](https://github.com/YinGuoX/awesome-implicit-representations)翻译和**拓展**

> - 这个列表并不详尽，因为隐式神经表征是一个快速发展的研究领域，迄今已有数百篇论文。
> - 此列表旨在列出相隐式神经表示的关键概念和基础应用的论文。 如果您想在该领域开始，这是一个很棒的阅读列表！
> - 因为对于大多数论文来说，最重要的是论文概要和贡献。
> - 待改格式
>   - 论文、链接、源码
>     - 贡献
>     - 大概思想

## 0. 大牛们的Talks

- [Vincent Sitzmann: Implicit Neural Scene Representations (Scene Representation Networks, MetaSDF, Semantic Segmentation with Implicit Neural Representations, SIREN)](https://www.youtube.com/watch?v=__F9CCqbWQk&t=1s)
- [Andreas Geiger: Neural Implicit Representations for 3D Vision (Occupancy Networks, Texture Fields, Occupancy Flow, Differentiable Volumetric Rendering, GRAF)](https://www.youtube.com/watch?v=F9mRv4v80w0)
- [Gerard Pons-Moll: Shape Representations: Parametric Meshes vs Implicit Functions](https://www.youtube.com/watch?v=_4E2iEmJXW8)
- [Yaron Lipman: Implicit Neural Representations](https://www.youtube.com/watch?v=rUd6qiSNwHs&list=PLat4GgaVK09e7aBNVlZelWWZIUzdq0RQ2&index=11)

## 1. 什么是隐含神经表示？

- 隐式神经表示（有时也称为基于坐标的表示）是一种新的参数化信号的新方法。 传统的信号表示通常是离散的 - 例如，图像是像素的离散网格，音频信号是幅度的离散样本，并且3D形状通常被参数化为体素，点云或网格的网格。 相反，隐式的神经表示参数化信号作为连续函数，该函数映射信号的域（即，坐标，例如图像的像素坐标）到该坐标处的任何坐标（对于图像，r，g ，b颜色）。 当然，这些函数通常没有分析易行 - 无法“记下”作为数学公式参数化自然图像的功能。 因此，隐式神经表示通过神经网络近似该功能。

## 2. 它们为什么有趣？

- 隐式神经表示具有若干好处：首先，它们不再耦合到空间分辨率，例如图像耦合到像素的数量。 这是因为它们是连续的功能！ 因此，参数化信号所需的存储器独立于空间分辨率，并且仅具有升压信号的复杂性的尺度。 另一个推论是隐式表示具有“无限分辨率” - 它们可以在任意空间分辨率下进行采样。这可以立即有用，例如超分辨率，或者在3D和更高尺寸中的参数化信号中有用，因为传统的3D表示的随着空间分辨率的快速增长存储空间也会急速增长。
- 然而，在未来，隐式神经表示的关键在于直接在这些表示的空间中直接运行。 换句话说：什么是“卷积神经网络”相当于在由隐式表示表示的图像上运行的神经网络？ 像这样的问题为一类独立于空间分辨率的算法提供了一条道路！

## 3.5 About Me Insight

### 1. 起源

- [the Autonomous Vision Group at MPI-IS Tübingen and University of Tübingen](https://github.com/autonomousvision)：在论证隐式神经表示的优越性后还做了一系列的相关工作！
  - 该实验室最早论证了隐式神经表示在参数化几何中优于基于网格、点和网格的表示，并且无缝地允许学习形状的先验知识
    - （同年还有另外几篇类似的工作，请看第3节）
    - [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://arxiv.org/abs/1812.03828) (CVPR2019)[有源码](https://github.com/autonomousvision/occupancy_networks)
      - 提出了一种新的三维表示：occupancy function、展示了这种表示形式：在各种输入类型(体素、点云、网格、图片)下如何创建三维物体、论证了这种表现形式：能够生成高质量的三维网格并且优于现有技术
  - [Convolutional Occupancy Networks](https://link.springer.com/content/pdf/10.1007/978-3-030-58580-8_31.pdf)(ECCV2020)[有源码](https://github.com/autonomousvision/convolutional_occupancy_networks)
    - 针对Occupancy Network的缺点：observation与3d point之间无相关性，做出的改进
  - [Differentiable Volumetric Rendering: Learning Implicit 3D Representations without 3D Supervision](http://www.cvlibs.net/publications/Niemeyer2020CVPR.pdf)(CVPR2020)[有源码](https://github.com/autonomousvision/differentiable_volumetric_rendering)
    - 贡献：2D图像的纹理和形状的三维重建
    - 针对现有隐式表示的缺点：都是使用真实(合成)的3D数据进行学习=>提出利用2D图像和相机参数进行重建的DVR模型

### 2. 各个领域百花齐放

-  [DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction ](https://arxiv.org/abs/1905.10711)(NeurIPS 2019)[有源码](https://github.com/Xharlie/DISN)
  - 贡献：单视图的三维重建
  - 针对现有的缺点、根据隐式表示的思想(SDF而不是Occupancy)、提出了DISN模型：相机参数预测+单张图片=>预测SDF=>使用Marching Cubes来进行可视化论证重建有效性



---

- 待看&待修正

## 3.相关论文

### 3.1 几何形状的隐含神经表示

- 以下三篇论文首先（同时）证明了隐式神经表示在参数化几何中优于基于网格、点和网格的表示，并且无缝地允许学习形状的先验知识。

> - [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103) (Park et al. 2019)
> - [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://arxiv.org/abs/1812.03828) (Mescheder et al. 2019)
> - [IM-Net: Learning Implicit Fields for Generative Shape Modeling](https://arxiv.org/abs/1812.02822) (Chen et al. 2018)

- 从那时起，隐式神经表示在三维计算机视觉中取得了最先进的成果

> - 论证如何从没有地面真值符号距离值的原始数据中学习 SDFs 
>   - [Sal: Sign agnostic learning of shapes from raw data](https://github.com/matanatz/SAL) 
>   - [Implicit Geometric Regularization for Learning Shapes](https://github.com/amosgropp/IGR) (Gropp et al. 2020)
> - 同时提出了适合大规模三维场景的混合体素网格/隐式表示方法。
>   - [Local Implicit Grid Representations for 3D Scenes](https://geometry.stanford.edu/papers/jsmhnf-lligrf3s-20/jsmhnf-lligrf3s-20.pdf)
>   - [Convolutional Occupancy Networks](https://arxiv.org/abs/2003.04618)
>   - [Deep Local Shapes: Learning Local SDF Priors for Detailed 3D Reconstruction](https://arxiv.org/abs/2003.10983) 
> - 演示如何利用正弦激活函数，通过单个隐式神经表示来参数化房间规模的三维场景。
>   - [Implicit Neural Representations with Periodic Activation Functions](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020)
> - 论证从原始点云中学习无符号距离场，从而消除水密表面的要求。
>   - [Neural Unsigned Distance Fields for Implicit Function Learning](https://arxiv.org/pdf/2010.13938.pdf) (Chibane et al. 2020) 

### 3.2 几何形状和外观纹理的隐式神经表示

- 仅从2D图像中监督学习（“逆图形学”）

> - 提出通过a differentiable ray-marcher从2D图像中学习3D形状和几何形状的隐式表示，并通过超网络从单个图像中重建的3D场景概括。这是针对单一对象场景展示的，也是简单的房间规模场景
>   - [Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations](https://vsitzmann.github.io/srns/) 
> - 使用完全连接的神经网络和分析梯度将基于LSTM的Ray-Marcher替换为SRNS，使最终的3D几何形状轻松提取。
>   - [Differentiable volumetric rendering: Learning implicit 3d representations without 3d supervision](https://github.com/autonomousvision/differentiable_volumetric_rendering)
> - 提出了基于位置编码、体绘制和光线方向调节的单场景高质量重建方法，并对三维隐式表示的体绘制进行了大量的研究。一份专门针对NeRF后续工作的策划清单（ see [awesome-NeRF](https://github.com/yenchenlin/awesome-NeRF)）
>   - [Neural Radiance Fields (NeRF)](https://www.matthewtancik.com/nerf) (Mildenhall et al. 2020)
> - 演示如何从单个观测对象来训练场景表示网络
>   - [SDF-SRN: Learning Signed Distance 3D Object Reconstruction from Static Images](https://github.com/chenhsuanlin/signed-distance-SRN) (Lin et al. 2020)
> - 提出一种基于局部特征的NeRF方法，该方法是根据PiFU中提出的从接触图像中提取的摄像机光线的局部特征
>   - [Pixel-NERF](https://alexyu.net/pixelnerf/) (Yu et al. 2020) 
> - 针对复杂三维场景的重建问题，提出了基于位置编码的球体追踪算法，并提出了一种基于表面法线和视线方向的球体追踪算法。
>   - [Multiview neural surface reconstruction by disentangling geometry and appearance](https://lioryariv.github.io/idr/) (Yariv et al. 2020) 

- 从3D表示中监督学习

> - Pifu首先提出了将隐式表示条件化为从上下文图像中提取的局部特征的概念。后续工作实现了照片真实感、实时重渲染。
>   - [Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization(Saito et al. 2019)](https://shunsukesaito.github.io/PIFu/)
> - (待改)[Texture Fields: Learning Texture Representations in Function Space](https://autonomousvision.github.io/texture-fields/)

- 对于动态场景

> - 首先提出了通过用隐式神经表示来表示4D扭曲场来学习时空神经隐式表示。
>   - [Occupancy flow: 4d reconstruction by learning particle dynamics](https://avg.is.tuebingen.mpg.de/publications/niemeyer2019iccv) (Niemeyer et al. 2019) 

- 同时提出的以下论文仅利用类似的方法来通过神经辐射场地从2D观察中重建动态场景。
  - [D-NeRF: Neural Radiance Fields for Dynamic Scenes](https://arxiv.org/abs/2011.13961)
  - [Deformable Neural Radiance Fields](https://nerfies.github.io/)
  - [Neural Radiance Flow for 4D View Synthesis and Video Processing](https://yilundu.github.io/nerflow/)
  - [Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes](http://www.cs.cornell.edu/~zl548/NSFF/)
  - [Space-time Neural Irradiance Fields for Free-Viewpoint Video](https://video-nerf.github.io/)
  - [Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Deforming Scene from Monocular Video](https://gvv.mpi-inf.mpg.de/projects/nonrigid_nerf/)

### 3.3 隐式/显式混合(局部特征上的隐式条件)

- 以下四篇论文同时提出了对存储在体素网格中的局部特征进行隐式神经表示：
  - [Implicit Functions in Feature Space for 3D ShapeReconstruction and Completion](https://virtualhumans.mpi-inf.mpg.de/papers/chibane20ifnet/chibane20ifnet.pdf)
  - [Local Implicit Grid Representations for 3D Scenes](https://geometry.stanford.edu/papers/jsmhnf-lligrf3s-20/jsmhnf-lligrf3s-20.pdf)
  - [Convolutional Occupancy Networks](https://arxiv.org/abs/2003.04618)
  - [Deep Local Shapes: Learning Local SDF Priors for Detailed 3D Reconstruction](https://arxiv.org/abs/2003.10983)
  - [Neural Sparse Voxel Fields](https://github.com/facebookresearch/NSVF)
- 以下论文对局部面片上的深符号距离函数进行了条件化处理：
  - [Local Deep Implicit Functions for 3D Shape](https://ldif.cs.princeton.edu/)
  - [PatchNets: Patch-Based Generalizable Deep Implicit 3D Shape Representations](http://gvv.mpi-inf.mpg.de/projects/PatchNets/)

### 3.4 下游任务的隐式神经表示学习

- 利用场景表示网络学习的特征对三维对象进行弱监督语义分割
  - [Inferring Semantic Information with 3D Neural Scene Representations](https://www.computationalimaging.org/publications/semantic-srn/) 

### 3.5 具有神经隐式表示的泛化与元学习

- 提出了从上下文图像中提取光线特征的局部条件隐式表示方法。
  - [Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization](https://shunsukesaito.github.io/PIFu/) (Saito et al. 2019)
- 通过超级网络进行元学习
  - [Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations](https://vsitzmann.github.io/srns/) (Sitzmann et al. 2019) 
- 提出了一种基于梯度的隐式神经表征元学习方法
  - [MetaSDF: MetaSDF: Meta-Learning Signed Distance Functions](https://vsitzmann.github.io/metasdf/) (Sitzmann et al. 2020) 
- 展示如何仅从单图像监督中学习3D隐式表示
  - [SDF-SRN: Learning Signed Distance 3D Object Reconstruction from Static Images](https://github.com/chenhsuanlin/signed-distance-SRN)
- 为NeRF探索了基于梯度的元学习
  - [Learned Initializations for Optimizing Coordinate-Based Neural Representations](https://www.matthewtancik.com/learnit)

### 3.6 用位置编码和周期性非线性拟合高频细节

- 提出了位置编码
  - [Neural Radiance Fields (NeRF)](https://www.matthewtancik.com/nerf) (Mildenhall et al. 2020)
- 提出了具有周期性非线性的隐含表示
  - [Implicit Neural Representations with Periodic Activation Functions](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020)
- 探索NTK框架中的位置编码
  - [Fourier features let networks learn high frequency functions in low dimensional domains](https://people.eecs.berkeley.edu/~bmild/fourfeat/) (Tancik et al. 2020) 

### 3.7 图像的隐式神经表征

- 首先提出通过神经网络隐式地参数化图像。
  - [Compositional Pattern-Producing Networks: Compositional pattern producing networks: A novel abstraction of development](https://link.springer.com/content/pdf/10.1007/s10710-007-9028-8.pdf) (Stanley et al. 2007)
- 提出了一种基于超网络的图像隐式表示方法。
  - [Implicit Neural Representations with Periodic Activation Functions](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020) 

- 将像素位置的Jacobian参数化为视图，时间，照明等到自然内插图像。
  - [X-Fields: Implicit Neural View-, Light- and Time-Image Interpolation](https://xfields.mpi-inf.mpg.de/)
- 提出了一种基于超网络的 GAN 图像处理方法。
  - [Learning Continuous Image Representation with Local Implicit Image Function](https://github.com/yinboc/liif) (Chen et al. 2020)

### 3.8 合成隐式神经表示

- 下面的论文提出用每一个物体的三维隐式神经表示来组装场景。
  - [GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields](https://arxiv.org/abs/2011.12100)
  - [Object-centric Neural Rendering](https://arxiv.org/pdf/2012.08503.pdf)

### 3.9 偏微分方程边值问题的隐式表示

- 通过损失强制执行Eikonal方程的约束来学习SDF。
  - [Implicit Geometric Regularization for Learning Shapes](https://github.com/amosgropp/IGR) (Gropp et al. 2020)
- 建议将周期性正弦作为激活函数利用，从而实现具有非普通高阶导数和复杂PDE的解决方案的函数的参数化。
  - [Implicit Neural Representations with Periodic Activation Functions](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020)
- 使用本地隐式代表使用辅助PDE损耗来执行超分辨率的时空流量函数
  - [MeshfreeFlowNet: Physics-Constrained Deep Continuous Space-Time Super-Resolution Framework](http://www.maxjiang.ml/proj/meshfreeflownet) (Jiang et al. 2020)

### 3.10 具有隐式表示的生成对抗网络

- 从3D数据
  - [Generative Radiance Fields for 3D-Aware Image Synthesis](https://autonomousvision.github.io/graf/) (Schwarz et al. 2020)
  - [pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis](https://arxiv.org/abs/2012.00926) (Chan et al. 2020)
  - [Unconstrained Scene Generation with Locally Conditioned Radiance Fields](https://arxiv.org/pdf/2104.00670.pdf) (DeVries et al. 2021) Leverage a hybrid implicit-explicit representation, by generating a 2D feature grid floorplan with a classic convolutional GAN, and then conditioning a 3D neural implicit representation on these features. This enables generation of room-scale 3D scenes.
- 从2D数据
  - [Adversarial Generation of Continuous Images](https://arxiv.org/abs/2011.12026) (Skorokhodov et al. 2020)
  - [Learning Continuous Image Representation with Local Implicit Image Function](https://github.com/yinboc/liif) (Chen et al. 2020)
  - [Image Generators with Conditionally-Independent Pixel Synthesis](https://arxiv.org/abs/2011.13775) (Anokhin et al. 2020)

### 3.11 图像到图像翻译

- [Spatially-Adaptive Pixelwise Networks for Fast Image Translation](https://arxiv.org/pdf/2012.02992.pdf) (Shaham et al. 2020) leverages a hybrid implicit-explicit representation for fast high-resolution image2image translation.

### 3.12 Articulated representations

- [NASA: Neural Articulated Shape Approximation](https://virtualhumans.mpi-inf.mpg.de/papers/NASA20/NASA.pdf) (Deng et al. 2020) represents an articulated object as a composition of local, deformable implicit elements.




