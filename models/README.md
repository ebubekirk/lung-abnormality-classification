Following resources used to implement the base models.

#EfficientNetB0
TAN, Mingxing and LE, Quoc V., 2020. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.. Online. 11 September 2020. arXiv. arXiv:1905.11946.

##Train first 10.000 patients, image dimensions 200x200, 10 folds, without augmentation
~3 hrs for 10 fold, 10 epochs
precision=0.7513, recall=0.9412, f1=0.7947, accuracy=0.7260, auc=0.5231

#DenseNet121
HUANG, Gao, LIU, Zhuang, MAATEN, Laurens van der and WEINBERGER, Kilian Q., 2018. Densely Connected Convolutional Networks.. Online. 28 January 2018. arXiv. arXiv:1608.06993.

Code and pre-trained models are available at https://github.com/liuzhuang13/DenseNet

##Train first 10.000 patients, image dimensions 200x200, 10 folds, without augmentation
~10 hrs for 10 fold, 10 epochs
precision=0.8260, recall=0.8962, f1=0.8547, accuracy=0.8260, auc=0.5543

#ResNet34

HE, Kaiming, ZHANG, Xiangyu, REN, Shaoqing and SUN, Jian, 2015. Deep Residual Learning for Image Recognition.. Online. 10 December 2015. arXiv. arXiv:1512.03385.

##Train first 10.000 patients, image dimensions 200x200, 10 folds, without augmentation
~6 hrs for 10 fold, 10 epochs
precision=0.8350, recall=0.9249, f1=0.8777, accuracy=0.7870, auc=0.6140
