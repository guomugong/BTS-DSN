## BTS-DSN: Deeply Supervised Neural Network with Short Connections for Retinal Vessel Segmentation
Please read our [paper] (https://doi.org/10.1016/j.ijmedinf.2019.03.015) for more details!
### Introduction:
Background and Objective: The condition of vessel of the human eye is an important factor for the diagnosis of ophthalmological diseases. Vessel segmentation in fundus images is a challenging task due to complex vessel structure, the presence of similar structures such as microaneurysms and hemorrhages, micro-vessel with only one to several pixels wide, and requirements for finer results.
Methods:In this paper, we present a multi-scale deeply supervised network with short connections (BTS-DSN) for vessel segmentation. We used short connections to transfer semantic information between side-output layers. Bottom-top short connections pass low level semantic information to high level for refining results in high-level side-outputs, and top-bottom short connection passes much structural information to low level for reducing noises in low-level side-outputs. In addition, we employ cross-training to show that our model is suitable for real world fundus images.
Results: The proposed BTS-DSN has been verified on DRIVE, STARE and CHASE_DB1 datasets, and showed competitive performance over other state-of-the-art methods. Specially, with patch level input, the network achieved 0.7891/0.8212 sensitivity, 0.9804/0.9843 specificity, 0.9806/0.9859 AUC, and 0.8249/0.8421 F1-score on DRIVE and STARE, respectively. Moreover, our model behaves better than other methods in cross-training experiments.
Conclusions: BTS-DSN achieves competitive performance in vessel segmentation task on three public datasets.

# Training BTSDSN
1. Download the DRIVE dataset from (https://www.isi.uu.nl/Research/Databases/DRIVE/download.php).
2. Prepare the training set.
3. Download fully convolutional VGG model (248MB) from (http://vcl.ucsd.edu/hed/5stage-vgg.caffemodel) and put it in $CAFFE_ROOT/btsdsn/.	
4. Build Caffe
5. Modify solver.prototxt, train.py and list files (data/drive/*.lst)
6. Run the python scripts in $CAFFE_ROOT/btsdsn
	```bash
	python train.py
	```
# Testing BTSDSN
1. Clone the respository
	```bash
	git clone https://github.com/guomugong/BTS-DSN.git
	```
2. Build Caffe
	```bash
	cp Makefile.config.example Makefile.config
	make all -j8
	make pycaffe
	```
3. Prepare your retinal images and modify test.py
4. Run
	```bash
	python test.py
	```
## License
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![Badge](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu/#/zh_CN)
