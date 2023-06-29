本实验基于和AlexNet相同参数量的ViT在CIFAR-100进行训练，并与作业二中AlexNet结果进行对比，使用相同的数据增强方法。

模型参数保存至百度云盘，下载链接：https://pan.baidu.com/s/1OqUDy-b58L0o9c-JHIFxlg 提取码：dama

文件架构：

util：内含train会用到的函数

mixup_log, cutmix_log, cutout_log, baseline_log: 包含用于tensorboard可视化的参数，输入tensorboard --logdir ./baseline_log即可查看

data：CIFAR-100数据集

log：train过程中日志信息，包括每轮的train和test loss以及acc

nohup.out: 包含训练过程中命令行输出的信息

train：训练代码

### 一、CIFAR-100数据集介绍

	CIFAR数据集包含100个类别，每个类别包含600个图像，每个类各有500个训练图像和100个测试图像。CIFAR-100中的100个类被分成20个超类。每个图像都带有一个“精细”标签（它所属的类）和一个“粗糙”标签（它所属的超类）。
	官方提供python、matlab以及二进制版本（适用于C语言），[官方下载地址](http://www.cs.toronto.edu/~kriz/cifar.html)或者通过torchvision中datasets类进行导入。
训练和测试的数据结构（考虑到数据大小问题，均为二进制编码）是一个字典，内含5个键值对，以训练数据为例，分别为：

- filenames：每张图片的图片名字
- batch_label: training batch 1 of 1，意为这里面只有1个batch，即50000个图片训练数据未作切分。
- fine_labels:  细粒度label，取值0-99。一个长度为50000的列表。
- coarse_labels: 粗粒度label，取值0-19。一个长度为50000的列表
- data: 一个numpy array 训练集为50000 ![](https://www.yuque.com/api/services/graph/generate_redirect/latex?%5Ctimes#card=math&code=%5Ctimes&id=bJAKl) 3074 uint8。一共RGB三通道，每个通道32![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=ClOWP)32个像素点。

此外还有个meta文件，内含一个元数据字典，介绍有哪些细粒度和粗粒度类别。测试集为 10000![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=IK2Ns)3074 uint8
### 二、网络架构
采用Vision Transformer架构。采用和第二次作业1中的AlexNet相同参数量的网络模型。

1. AlexNet有5个CNN层和3个全连接层，参数量约为2800万。
| 层名称 | 说明 | 参数数量（w和b） |
| --- | --- | --- |
| Input | W*H*C=32*32*3 | N/A |
| CNN1 | 64个3![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=Z8xbm)3![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=gYUX6)3的卷积核 | 3*3*3*64+64=1792 |
| CNN2 | 192个3![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=F8GPr)3![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=tk9pX)64的卷积核 | 3*3*64*192+192=110784 |
| CNN3 | 384个3![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=XssS2)3![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=qnMnA)192的卷积核 | 3*3*192*384+384=663936 |
| CNN4 | 256个3![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=fRHlU)3![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=C3Qzj)384的卷积核 | 3*3*384*256+256=442496 |
| CNN5 | 256个3![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=AZADJ)3![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=PsjyC)256的卷积核 | 3*3*256*256+256=295040 |
| FC1 | 全连接层 | 3*3*256*4096+4096=9441280 |
| FC2 | 全连接层 | 4096*4096+4096=16781312 |
| FC3 | 全连接层 | 4096*100+100=409700 |
| 合计 | N/A | 28146340 |

2. Transformer的参数量

transformer模型由 ![](https://cdn.nlark.com/yuque/__latex/6945e109777fe3fd777e8254f0ec0f0c.svg#card=math&code=l&id=lZq5w) 个相同的层组成，每个层分为两部分：self-attention块和MLP块。
self-attention块的模型参数有 ![](https://cdn.nlark.com/yuque/__latex/9ed7e05535662fb8ce85ffeae3849546.svg#card=math&code=Q%E3%80%81K%E3%80%81V&id=kv5XU) 的权重矩阵 ![](https://cdn.nlark.com/yuque/__latex/88f0896682bf1e752edf406c127d8dd8.svg#card=math&code=W_Q%E3%80%81W_K%E3%80%81W_V&id=okgfK) 和偏置，输出权重矩阵 ![](https://cdn.nlark.com/yuque/__latex/a583a4d789e5dbe148d913294939793c.svg#card=math&code=W_O&id=NHWhw)和偏置，4个权重矩阵的形状为 [ℎ,ℎ] ，4个偏置的形状为 [ℎ] 。self- attention块的参数量为 ![](https://cdn.nlark.com/yuque/__latex/68a0bb9b04aba964befec1460efbab95.svg#card=math&code=4%E2%84%8E%5E2%2B4%E2%84%8E&id=SvhU2)。
MLP块由2个线性层组成，一般地，第一个线性层是先将维度从 ℎ 映射到 4ℎ ，第二个线性层再将维度从4ℎ映射到ℎ。第一个线性层的权重矩阵 ![](https://cdn.nlark.com/yuque/__latex/67d6dc1a5ff0c821d637dfb812a1b45c.svg#card=math&code=W_1&id=aK8Uc) 的形状为 [ℎ,4ℎ] ，偏置的形状为 [4ℎ] 。第二个线性层权重矩阵 ![](https://cdn.nlark.com/yuque/__latex/e7dce22f135611eb17988572021f74cc.svg#card=math&code=W_2&id=Qspkt) 的形状为 [4ℎ,ℎ] ，偏置形状为 [ℎ] 。MLP块的参数量为![](https://cdn.nlark.com/yuque/__latex/bd48d080a7da2e02cf5f627934a9010e.svg#card=math&code=8%E2%84%8E%5E2%2B5%E2%84%8E&id=qYcYS) 。
self-attention块和MLP块各有一个layer normalization，包含了2个可训练模型参数：缩放参数 ![](https://cdn.nlark.com/yuque/__latex/4aa418d6f0b6fbada90489b4374752e5.svg#card=math&code=%5Cgamma&id=N5dq4) 和平移参数 ![](https://cdn.nlark.com/yuque/__latex/6100158802e722a88c15efc101fc275b.svg#card=math&code=%5Cbeta&id=O78pq) ，形状都是 [ℎ] 。2个layer normalization的参数量为 4ℎ 。
所以每个transformer层的参数量为![](https://cdn.nlark.com/yuque/__latex/91108a05174a37b34e555b5fa3c6007f.svg#card=math&code=12h%5E2%2B13h&id=EvM0m)。除此之外，词嵌入矩阵的参数量也较多，词向量维度通常等于隐藏层维度 ℎ ，词嵌入矩阵的参数量为 ![](https://cdn.nlark.com/yuque/__latex/fba7f357f4d2bdd62c20ca90b7148358.svg#card=math&code=Vh&id=heDry) 。最后的输出层的权重矩阵通常与词嵌入矩阵是参数共享的。
综上，![](https://cdn.nlark.com/yuque/__latex/6945e109777fe3fd777e8254f0ec0f0c.svg#card=math&code=l&id=H9wKs)层transformer模型的可训练模型参数量为![](https://cdn.nlark.com/yuque/__latex/79a052107e6f2eb24ed89f32ad61f67a.svg#card=math&code=l%2812h%5E2%2B13h%29%2BVh&id=APEz2)。当隐藏维度h较大时，可以忽略一次项，模型的参数量近似为![](https://cdn.nlark.com/yuque/__latex/f2e51a7a751cbdebf6dec824a3c47065.svg#card=math&code=12lh%5E2&id=nlxZN)。
假设每个patch经过embedding转化为1000维向量，即![](https://cdn.nlark.com/yuque/__latex/2c26a1b95d69e841cccb26095e2a5e9d.svg#card=math&code=h%3D768&id=iIO5N)，则![](https://cdn.nlark.com/yuque/__latex/df90afe1ef4fbf9a2cb904f33e36ee8b.svg#card=math&code=l%5Capprox%204&id=yI4pZ)。
![](https://cdn.nlark.com/yuque/0/2023/webp/29336337/1687163094333-7cd2e11f-97db-4fad-936f-7844806b6a7c.webp#averageHue=%23f2f0ee&clientId=ua9720ee9-c85b-4&from=paste&id=u496371e5&originHeight=453&originWidth=834&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u0a1b7f61-178a-438f-987c-fc64163da1b&title=)
上图为ViT的架构图，实验采用的其他参数值均为默认值，每个patch大小为16![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=mJ3V5)16![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=TrcRN)3，patch_embedding后的维度为1024，经过5个transformer encoder，多头自注意力中num_heads设置为12，dropout_rate设置为0.5。最后经过一个MLP层，获得一个100维向量。
### 三、实验
#### 实验设置

在目录下输入：python train.py --parameter value 即可训练

设置参数的默认值，除了模型架构与AlexNet不同，其余设置相同，如下所示：
```
batch_size: 64
lr: 0.001
epochs: 200
image_size: 1024
patch_size: 16
num_heads: 12 #多头注意力的头数
num_layers: 12 # transformer的block层数
num_classes: 100
dropout_rate: 0.5
hidden_dim: 768
phase: 'train'
model_path: ./checkpoints/CIFAR100_baseline_epoch_201.pth
augment: baseline #数据增强方法，可取值：baseline（不使用任何数据增强）、mixup、cutout、cutmix
alpha：1.0 # mixup和cutmix方法的参数
p：0.5 # cutout和cutmix裁剪图片的概率
maskout_size: 8 # cutout被裁剪的大小
loss_function: CrossEntropyLoss #交叉熵
optimizer: Adam优化器
评测指标: Top1(acc)、Top5(acc). Topk是指只要结果前k个最大的分量含有正确答案即可
```
#### 实验结果

输入tensorboard --logdir baseline_log 即可

主要对比的是使用不同数据增强方法训练时的loss，测试的loss和acc。
##### baseline
|  | AlexNet | ViT |
| --- | --- | --- |
| 训练loss | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687940864034-89d95898-347d-4fc6-8576-925feb8bd0c2.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=510&id=u7348b553&originHeight=510&originWidth=759&originalType=binary&ratio=1&rotation=0&showTitle=false&size=20472&status=done&style=none&taskId=u7f53fc16-9d49-4f44-97fe-4dc792655af&title=&width=759) | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687941233060-59f7e35c-3987-475f-98d7-762a1d36be9c.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=512&id=u7a951f9d&originHeight=512&originWidth=757&originalType=binary&ratio=1&rotation=0&showTitle=false&size=20926&status=done&style=none&taskId=u44874ff5-acef-4854-8a27-31cdd62f1f1&title=&width=757) |
| 测试loss | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687941099953-63bbda62-b4ef-4a07-8b05-5cf041ffb6f3.png#averageHue=%23fbfbfb&clientId=u52dee214-909a-4&from=paste&height=511&id=uf02c8e1c&originHeight=511&originWidth=764&originalType=binary&ratio=1&rotation=0&showTitle=false&size=26960&status=done&style=none&taskId=uebd51b10-f2a1-4615-a567-a2f8efb32be&title=&width=764) | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687941245329-ce6b64c2-722d-4244-ac17-0e7b7aee0599.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=527&id=u5d73016c&originHeight=527&originWidth=766&originalType=binary&ratio=1&rotation=0&showTitle=false&size=27614&status=done&style=none&taskId=u6eaa64a8-a53d-480f-9286-11863c87190&title=&width=766) |
| 测试acc | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687941138510-70c80f88-7df3-43b3-b81f-3c4ee51ab0e7.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=535&id=uec101f16&originHeight=535&originWidth=771&originalType=binary&ratio=1&rotation=0&showTitle=false&size=25455&status=done&style=none&taskId=u49fee215-a19f-49e6-a000-e4bb63f1a23&title=&width=771) | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687941259215-f3e13d04-6db6-4d01-baaf-837c5cc8be38.png#averageHue=%23e6e8eb&clientId=u52dee214-909a-4&from=paste&height=523&id=u562cc98a&originHeight=523&originWidth=762&originalType=binary&ratio=1&rotation=0&showTitle=false&size=39940&status=done&style=none&taskId=u6c93c482-5ecd-4831-990b-88794df8248&title=&width=762) |
| 训练acc | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687941120069-7f7d3e5d-1ae5-4c9f-b694-e879db136104.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=537&id=u0ff9adc6&originHeight=537&originWidth=751&originalType=binary&ratio=1&rotation=0&showTitle=false&size=22054&status=done&style=none&taskId=u5845ab69-68ae-4222-921e-0e7b5100cfa&title=&width=751) | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687941269327-7d970570-e426-4a5f-b169-dde61d1adbf8.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=523&id=u050c259a&originHeight=523&originWidth=752&originalType=binary&ratio=1&rotation=0&showTitle=false&size=24078&status=done&style=none&taskId=u0878fd13-36e7-4891-93e4-a032973cffb&title=&width=752) |

##### mixup
|  | AlexNet | ViT |
| --- | --- | --- |
| 训练loss | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687947504912-84364960-7b66-47dc-9b6d-dacd29804e46.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=505&id=u5026b5d0&originHeight=505&originWidth=751&originalType=binary&ratio=1&rotation=0&showTitle=false&size=28156&status=done&style=none&taskId=u2c02b149-7e27-41af-a7d0-4d33ebc3a7a&title=&width=751) | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687947355054-f0feee0a-7689-417b-a287-7d77332c9ca1.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=526&id=uc8be253a&originHeight=526&originWidth=770&originalType=binary&ratio=1&rotation=0&showTitle=false&size=24051&status=done&style=none&taskId=u6f9f6b7c-9c6e-4d64-b7c8-b0a34f49f5e&title=&width=770) |
| 测试loss | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687947520054-4fdd01a6-7c1b-4075-b167-dca56a04865e.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=529&id=uc25f87fe&originHeight=529&originWidth=772&originalType=binary&ratio=1&rotation=0&showTitle=false&size=28472&status=done&style=none&taskId=udd885431-3af5-409f-b1f1-7e90b87235c&title=&width=772) | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687947387880-4f2fcb35-04dd-495e-b7eb-48f70990b6d1.png#averageHue=%23fbfbfb&clientId=u52dee214-909a-4&from=paste&height=541&id=u4ef1a22a&originHeight=541&originWidth=792&originalType=binary&ratio=1&rotation=0&showTitle=false&size=38628&status=done&style=none&taskId=u99a50ad5-cf31-4f97-8a87-d9cff34ca11&title=&width=792) |
| 测试acc | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687956083168-5906b6ee-2b17-450e-b8b4-97471c2d8bb2.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=539&id=u99a7bbf6&originHeight=539&originWidth=771&originalType=binary&ratio=1&rotation=0&showTitle=false&size=26808&status=done&style=none&taskId=u52ff49f4-b0ed-4313-bbe9-7af4eca0dad&title=&width=771) | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687947437626-58e8c8fa-7dda-4dcd-b73f-d27441250e0b.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=529&id=uc32c004e&originHeight=529&originWidth=766&originalType=binary&ratio=1&rotation=0&showTitle=false&size=33622&status=done&style=none&taskId=uda4621f0-6b01-4f90-8683-01d4645cb47&title=&width=766) |
| 训练acc | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687941120069-7f7d3e5d-1ae5-4c9f-b694-e879db136104.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=537&id=R7zhf&originHeight=537&originWidth=751&originalType=binary&ratio=1&rotation=0&showTitle=false&size=22054&status=done&style=none&taskId=u5845ab69-68ae-4222-921e-0e7b5100cfa&title=&width=751) | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687947421929-ac7dcb7e-b25a-47a7-b94d-ca14e9bcd609.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=529&id=u8fba28f5&originHeight=529&originWidth=761&originalType=binary&ratio=1&rotation=0&showTitle=false&size=16707&status=done&style=none&taskId=u7ed66b7d-a706-470f-905f-a765d88b08d&title=&width=761) |

##### cutout
|  | AlexNet | ViT |
| --- | --- | --- |
| 训练loss | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687955788965-14ce1f41-ceb7-4682-ba71-0f8907bf1f33.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=540&id=u3bc96ac5&originHeight=540&originWidth=771&originalType=binary&ratio=1&rotation=0&showTitle=false&size=20715&status=done&style=none&taskId=u133aea91-71c4-4553-9800-6323c2209f5&title=&width=771) | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687955707288-1bdc9358-4330-412b-89ac-062ab813cd82.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=530&id=u35bf88f9&originHeight=530&originWidth=765&originalType=binary&ratio=1&rotation=0&showTitle=false&size=17111&status=done&style=none&taskId=udfcb092a-b785-48a2-a844-37b91fa6ca7&title=&width=765) |
| 测试loss | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687955799247-541cdc18-b01a-454a-a255-2f12b0cdcbf7.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=528&id=u7949b619&originHeight=528&originWidth=762&originalType=binary&ratio=1&rotation=0&showTitle=false&size=29354&status=done&style=none&taskId=ua0b2a16a-989d-4a58-8b67-de85326803f&title=&width=762) | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687955718890-ae86d28b-8cc7-4941-8780-fda1ace05187.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=525&id=ubba6507b&originHeight=525&originWidth=763&originalType=binary&ratio=1&rotation=0&showTitle=false&size=28892&status=done&style=none&taskId=ua05b897d-352c-4f43-b911-1600fd1bd4a&title=&width=763) |
| 测试acc | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687955955986-1c50c956-51ec-4c4b-aaa7-f8440cfe0280.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=533&id=ucb297308&originHeight=533&originWidth=759&originalType=binary&ratio=1&rotation=0&showTitle=false&size=26212&status=done&style=none&taskId=u64ba5531-42ea-4b13-9dfc-3cc89aed7f7&title=&width=759) | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687955743023-9d360dce-f2e7-4cab-b615-bd1511775034.png#averageHue=%23edeef0&clientId=u52dee214-909a-4&from=paste&height=543&id=ud6a905ea&originHeight=543&originWidth=781&originalType=binary&ratio=1&rotation=0&showTitle=false&size=38292&status=done&style=none&taskId=u85d5e1f5-7766-48e7-b23d-8043f4ac9d9&title=&width=781) |
| 训练acc | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687955943374-a4f190e0-7871-47fb-a5ce-59f7b39b5ba8.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=531&id=u37b80555&originHeight=531&originWidth=784&originalType=binary&ratio=1&rotation=0&showTitle=false&size=22091&status=done&style=none&taskId=u194f38da-5db9-423e-a94e-c3b4df33235&title=&width=784) | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1687955752828-9bc0800a-7cb9-4ffb-b2e2-40094d62010e.png#averageHue=%23fcfcfc&clientId=u52dee214-909a-4&from=paste&height=539&id=u30139a04&originHeight=539&originWidth=766&originalType=binary&ratio=1&rotation=0&showTitle=false&size=20437&status=done&style=none&taskId=u1f769367-bea0-4816-a0e5-ce6c8974df9&title=&width=766) |

##### cutmix
|  | AlexNet | ViT |
| --- | --- | --- |
| 训练loss | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1688046334839-0c8c09a7-86a0-4e36-9d97-73fb739a8a8e.png#averageHue=%23fcfcfc&clientId=u0146afa4-3ab7-4&from=paste&height=541&id=ue1acac04&originHeight=541&originWidth=773&originalType=binary&ratio=1&rotation=0&showTitle=false&size=30312&status=done&style=none&taskId=uc50a9289-1d58-49c3-b212-7fa074d3aef&title=&width=773) | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1688046224682-3e299759-c0c5-40d5-8c9a-da1630a72488.png#averageHue=%23fcfcfc&clientId=u0146afa4-3ab7-4&from=paste&height=525&id=uf23887e2&originHeight=525&originWidth=764&originalType=binary&ratio=1&rotation=0&showTitle=false&size=25768&status=done&style=none&taskId=u99f01b17-f770-4a7e-af42-d610ce2b543&title=&width=764) |
| 测试loss | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1688046312792-23883a3f-5c13-4291-ae67-a7a6a5544780.png#averageHue=%23fcfcfc&clientId=u0146afa4-3ab7-4&from=paste&height=529&id=u0d3829be&originHeight=529&originWidth=765&originalType=binary&ratio=1&rotation=0&showTitle=false&size=32442&status=done&style=none&taskId=u712c9745-055b-46c0-85a7-0cfb8767222&title=&width=765) | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1688046247017-c8438fb7-1647-4424-994b-b802c1b67f19.png#averageHue=%23fcfcfc&clientId=u0146afa4-3ab7-4&from=paste&height=526&id=ub51f8459&originHeight=526&originWidth=765&originalType=binary&ratio=1&rotation=0&showTitle=false&size=35902&status=done&style=none&taskId=u4727b6b7-53c9-4b83-b0e3-371994663be&title=&width=765) |
| 测试acc | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1688046304760-2519c370-a8b0-49af-9211-957763fc1b95.png#averageHue=%23fcfcfc&clientId=u0146afa4-3ab7-4&from=paste&height=498&id=u1a963597&originHeight=498&originWidth=742&originalType=binary&ratio=1&rotation=0&showTitle=false&size=25804&status=done&style=none&taskId=u5ad4848b-5589-4cae-98e7-5fb3423615c&title=&width=742) | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1688046234062-e29a4378-8260-447f-8c71-21f80e5547d3.png#averageHue=%23fcfcfc&clientId=u0146afa4-3ab7-4&from=paste&height=530&id=ua5a1c3ff&originHeight=530&originWidth=747&originalType=binary&ratio=1&rotation=0&showTitle=false&size=31096&status=done&style=none&taskId=u902d21b4-8c9a-4c3c-acb5-cbdc44add97&title=&width=747) |
| 训练acc | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1688046324434-bea5e419-c339-47fc-a692-1737483b1f65.png#averageHue=%23fcfcfc&clientId=u0146afa4-3ab7-4&from=paste&height=534&id=ud9ec60aa&originHeight=534&originWidth=761&originalType=binary&ratio=1&rotation=0&showTitle=false&size=22130&status=done&style=none&taskId=u791c90d6-5ce1-4aa7-8a28-c52b6480344&title=&width=761) | ![image.png](https://cdn.nlark.com/yuque/0/2023/png/29336337/1688046256630-d81e7c86-58ab-438e-999e-d650813e3e54.png#averageHue=%23fcfcfc&clientId=u0146afa4-3ab7-4&from=paste&height=533&id=uac40ca2c&originHeight=533&originWidth=769&originalType=binary&ratio=1&rotation=0&showTitle=false&size=19157&status=done&style=none&taskId=ufe539123-6171-47bc-b9cc-64400801b72&title=&width=769) |

#### 结果分析
从baseline结果上看，AlexNet最后的最佳准确率大约在37%左右，而ViT为31.45%左右，且呈现严重的震荡现象。AlexNet的测试loss在第21个epoch开始呈上升趋势，而ViT从第5个epoch开始呈上升趋势。但训练loss和acc都没有大问题，ViT的loss下降速度和acc的上升速度更快，且两者的训练acc都逼近100%。因而出现严重的过拟合现象，ViT更为严重。
这可能是正常现象，对于小规模数据集（CIFAR-100），ViT可能并不会显著优于传统的卷积网络。主要是因为Transformer模型的数据需求经常较大，由于ViT自注意力机制，模型可能会在训练数据上过拟合，导致测试数据上的泛化能力下降。
其次数据增强的三个方法对于AlexNet与ViT提升精度类似，均在5%-10%左右，说明数据增强对于底层模型的依赖性不强。
