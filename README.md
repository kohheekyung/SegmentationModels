# Pretrained segmentation models
Customization of pretrained segmentation models based on https://github.com/qubvel/segmentation_models (See details in this repository)

#### Available models in this project 
- ResNet (it's not pretrained model)
- UNET (pretrained available)
- Linknet (pretrained available)
- FPN (pretrained available)
- PSPNet(pretrained available)
 
#### Customize loss
add loss functions under './base/loss_utils.py' (There is an example for segmentation_boundary_loss)

#### Additional functions for data
- data augmentation
- patch extraction

### How to train 
```python
python main.py 
```
### Configuration using Anaconda
```python
conda create -n tf python=3.7

conda install tensorflow-gpu=2.1.0 keras-gpu=2.3.1 cudnn cudatoolkit=10.1

pip install matplotlib

pip install sklearn

pip install git+https://www.github.com/keras-team/keras-contrib.git

pip install segmentation-models
```


