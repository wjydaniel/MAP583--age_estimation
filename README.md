# MAP583--age_estimation

In this section, based on [Projet age estimation](https://github.com/dataflowr/Project-age-estimation-pytorch), we implement the some modifications to make a comparation of age estimation performance using different methods: <br>
* train as Classification problem<br>
* train as Regression problem <br>
* use softmax expected age based on [DEX](http://people.ee.ethz.ch/~timofter/publications/Rothe-IJCV-2016.pdf)<br>
* implement [residual DEX](http://people.ee.ethz.ch/~timofter/publications/Agustsson-FG-2017.pdf) <br>

### Age Estimation PyTorch

##### Requirements:
```
pip install -r requirements.txt
```
##### Train:
Download and extract the [APPA-REAL dataset](chalearnlap.cvc.uab.es/dataset/26/description/)
```
wget http://158.109.8.102/AppaRealAge/appa-real-release.zip
unzip appa-real-release.zip
```
Train a model using the APPA-REAL dataset.
```
python train.py --data_dir [PATH/TO/appa-real-release] --tensorboard tf_log
```
Check training progress:
```
tensorboard --logdir=tf_log
```
Test Trained Model:
```
python test.py --data_dir [PATH/TO/appa-real-release] --resume [PATH/TO/BEST_MODEL.pth]
```

### Notice
An ignore list is given because some of the facial images in the dataset is not chosen properly from their original photo.
