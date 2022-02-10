## Introduction
Code for my master thesis: A Sketch based 3D Point Cloud Modeling System based on Deep Generation Network and Detail Editing

    
## Sketch-based 3D Point Cloud Modeling

### Functions

### Demo Video

## Code
### Set up environment
Please install the packages in requirements.txt
```
pip install -r requirements.txt
```

We recommand using python virtual env to set up the environment.

### Training
To train the reconstruction model:
```
python runner.py --train_gan
```

To train the erasing model:
```
python runner.py --train_refine
```

*Please check the config files in `configs` folder to make sure the paths are correct

### Testing
To test the models (Compute CD, EMD):

For reconstruction model:
```
python runner.py --test_gan
```

For refinement model:
```
python runner.py --test_refine
```

### Evaluating
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
https://github.com/hzxie/Pix2Vox.git
https://github.com/microsoft/SpareNet
