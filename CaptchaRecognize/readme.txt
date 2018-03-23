第一次训练：

	模型 ： logistic regression
	学习率： 0.01
	迭代次数： 2000
	训练样本： 130
	测试样本： 20
	图片尺寸： 46*46*1
	正规化项： 无

	结果： train_loss = 1 过拟和

第二次训练：

	模型 ： logistic regression
	学习率： 0.01
	迭代次数： 2000
	训练样本： 130
	测试样本： 20
	图片尺寸： 46*46*1
	正规化项： 无
	数据归一化：255->0, 0->1 

	结果：训练时间加快，train_loss = 0.7 欠拟和


第三次训练：

	模型 ： logistic regression
	学习率： 0.01
	迭代次数： 2000 -> 3000
	训练样本： 130
	测试样本： 20
	图片尺寸： 46*46*1
	正规化项： 无
	数据归一化：255->0, 0->1 

	结果：训练时间加快，train_loss = 0.64 欠拟和

第四次训练：

	模型 ： logistic regression
	学习率： 0.015
	迭代次数： 2000 -> 3000
	训练样本： 130
	测试样本： 20
	图片尺寸： 46*46*1
	正规化项： 无
	数据归一化：255->0, 0->1 

	结果：训练时间加快，train_loss = 0.64 欠拟和

第五次训练：

	模型 ： logistic regression
	学习率： 0.01
	迭代次数： 2000
	训练样本： 130->216
	测试样本： 20
	图片尺寸： 46*46*1
	正规化项： 无
	数据归一化：无

	结果：训练时间加快，train_loss = 0.000003 过拟和

第六次训练：

	模型 ： logistic regression
	学习率： 0.01
	迭代次数： 2000
	训练样本： 130->216
	测试样本： 20
	图片尺寸： 46*46*1
	正规化项： L2正则项: lamda = 1.
	数据归一化：无 

	结果：训练时间加快，train_loss = 0.6， test_accuracy = 0.723


第七次训练：

	模型 ： logistic regression
	学习率： 0.01
	迭代次数： 2000
	训练样本： 130->216
	测试样本： 20
	图片尺寸： 46*46*1
	正规化项： L2正则项: lamda = 0.6.
	数据归一化：无 

	结果：训练时间加快，train_loss = 0.01， test_accuracy = 0.703

sample_count = 232
alpha = 0.0188 , lambda = 0.5318
epoch =  0  loss=  20320.3
epoch =  240  loss=  160.863
epoch =  480  loss=  2.5586
epoch =  720  loss=  5.50947
epoch =  960  loss=  4.28473
Accuracy = 0.660

alpha = 0.0153 , lambda = 0.6803
epoch =  0  loss=  25851.2
epoch =  240  loss=  164.876
epoch =  480  loss=  4.70061
epoch =  720  loss=  3.93133
epoch =  960  loss=  0.131117
Accuracy = 0.596

alpha = 0.0100 , lambda = 0.6453
epoch =  0  loss=  25108.8
epoch =  240  loss=  1084.55
epoch =  480  loss=  48.7658
epoch =  720  loss=  2.19539
epoch =  960  loss=  0.166029
Accuracy = 0.681

alpha = 0.0115 , lambda = 0.6131
epoch =  0  loss=  23630.3
epoch =  240  loss=  751.852
epoch =  480  loss=  24.8947
epoch =  720  loss=  0.831378
epoch =  960  loss=  1.3794
Accuracy = 0.681

alpha = 0.0092 , lambda = 0.5890
epoch =  0  loss=  22778.2
epoch =  240  loss=  1627.52
epoch =  480  loss=  120.297
epoch =  720  loss=  8.89971
epoch =  960  loss=  0.659463
Accuracy = 0.660

alpha = 0.0185 , lambda = 0.5598
epoch =  0  loss=  21309.9
epoch =  240  loss=  140.809
epoch =  480  loss=  3.34119
epoch =  720  loss=  3.07447
epoch =  960  loss=  2.77257
Accuracy = 0.638

alpha = 0.0176 , lambda = 0.5626
epoch =  0  loss=  21129.0
epoch =  240  loss=  171.488
epoch =  480  loss=  1.48737
epoch =  720  loss=  6.65983
epoch =  960  loss=  0.547397
Accuracy = 0.702

alpha = 0.0125 , lambda = 0.5905
epoch =  0  loss=  22764.7
epoch =  240  loss=  622.49
epoch =  480  loss=  17.6432
epoch =  720  loss=  2.74942
epoch =  960  loss=  2.3474
Accuracy = 0.702

alpha = 0.0096 , lambda = 0.5351
epoch =  0  loss=  21270.9
epoch =  240  loss=  1683.07
epoch =  480  loss=  140.582
epoch =  720  loss=  11.7586
epoch =  960  loss=  0.997707
Accuracy = 0.638

alpha = 0.0140 , lambda = 0.5786
epoch =  0  loss=  22255.2
epoch =  240  loss=  432.758
epoch =  480  loss=  8.6865
epoch =  720  loss=  0.269655
epoch =  960  loss=  0.0487019
Accuracy = 0.638