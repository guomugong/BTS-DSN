## BTS-DSN: Deeply Supervised Neural Network with Short Connections for Retinal Vessel Segmentation
Please read our [paper] (https://arxiv.org/abs/1803.03963) for more details!
### Introduction:
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
3. Prepare your retinal images and set its path in test.py
4. Run
	```bash
	python test.py
	```

# Acknowledgment

## License
This code can not be used for commercial applications
