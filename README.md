## Introduction
Code for my master thesis: A Sketch based 3D Point Cloud Modeling System based on Deep Generation Network and Detail Editing

## Sketch-based 3D Point Cloud Modeling
### Functions
System overview:
![image](https://user-images.githubusercontent.com/27956674/153366179-b50d9409-d666-427e-8d0b-e8827d8ca950.png)

Main functions:
1. Generate 3D point clouds from sketches.
2. 2. Add erasing hints to improve point clouds' details.
3. Add thin structures.
4. Apply deformation.

After editing, user can save the result point clouds.

### Demo Video
Demo video on youtube.

## Code
### Set up environment
Please install the packages in requirements.txt
```
pip install -r requirements.txt
```

We recommand using python virtual env to set up the environment.

### For model training and testing
To train the reconstruction model:
```
python runner.py --train_gan
```

To train the erasing model:
```
python runner.py --train_refine
```

*Please check the config files in `configs/` folder to make sure the paths are correct

<br />

To test the models (Compute CD, EMD):
For reconstruction model:
```
python runner.py --test_gan
```

For refinement model:
```
python runner.py --test_refine
```
<br />

To evaluate the models (Visualize the point clouds created from models):
For reconstrution model:
```
python runner.py --evaluate_gan
```

For refinement model:
```
python runner.py --evaluate_refine
```

### Running the User Interface
Please run:
```
python sketch_3d_app.py
```

## Dataset
Links:

## Pretrained Weight
Links

## Reference
https://github.com/hzxie/Pix2Vox.git <br />
https://github.com/microsoft/SpareNet
