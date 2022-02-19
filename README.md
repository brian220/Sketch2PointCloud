## Introduction
Code for my master thesis: A Sketch based 3D Point Cloud Modeling System based on Deep Generation Network and Detail Editing

## Sketch-based 3D Point Cloud Modeling
### System overview:
![image](https://user-images.githubusercontent.com/27956674/153366179-b50d9409-d666-427e-8d0b-e8827d8ca950.png)

### Main functions:
1. Generate 3D point clouds from sketches by reconstruction module.
2. Add erasing hints to improve point clouds' details by erasing module.
3. Add thin structures.
4. Apply deformation.

After editing, user can save the result point clouds.

### Demo Video
Demo video on youtube: https://www.youtube.com/watch?v=Vch4nURRrAE&t=50s

## Code
### Set up environment
Create [Python Virtual Env](https://docs.python.org/zh-tw/3/tutorial/venv.html)
```
virtualenv -p virtualenv -p /usr/bin/python3.6 sketch2pc_venv
source sketch2pc_venv/bin/activate
```

Install Pytorch
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Compile the EMD loss function, we use the code from the [SpareNet](https://github.com/microsoft/SpareNet)
```
cd cuda/emd
python setup.py install
```

For other packages, install the packages in requirements.txt
```
pip install -r requirements.txt
```

This code is tested under CUDA 11.0 on Ubuntu 20.04 and a RTX2080 super GPU.

### For model training and testing
Before training and testing, please check the config files in `configs/` to make sure the paths are correct

<br />

**Training**

For reconstruction module:
```
python runner.py --train_gan
```

For erasing module:
```
python runner.py --train_refine
```

<br />

**Testing**

Compute CD, EMD:<br />
For reconstruction module:
```
python runner.py --test_gan
```

For erasing module:
```
python runner.py --test_refine
```

<br />

**Evaluating**

Visualize the point clouds created from models:<br />
For reconstruction module:
```
python runner.py --evaluate_gan
```

For erasing module:
```
python runner.py --evaluate_refine
```

### Running the User Interface
Please run:
```
python sketch_3d_app.py
```

## Downloads
### Dataset
We render the 3D models in [ShapeNet](https://shapenet.org/) to different views and convert the rendering images to
sketch images by [Photo-Sketching] (https://github.com/mtli/PhotoSketch)

Links for point clouds: 
Links for sketch images: https://drive.google.com/file/d/1kibjgEdlkqeUIEumexFKe2SWif0LX2xN/view?usp=sharing
Links for sketch images and erasing annotations: 

### Pretrained Weight
Links for reconstruction module:
Links for erasing module:

## Reference
https://github.com/hzxie/Pix2Vox.git <br />
https://github.com/justanhduc/graphx-conv <br />
https://github.com/kangxue/P2P-NET <br />
https://github.com/microsoft/SpareNet <br />
https://github.com/mathLab/PyGeM
