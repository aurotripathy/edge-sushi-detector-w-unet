## Detecting sushi at the edge with Percept Device Kit (DK)

This repo captures all of my steps to segment sushi in images. 

At a very high-level in includes the following steps:
* image collection and annotation
* model training and saving in the ONNX format
* post-training model optimization w/OpenVino's model optimizer
* conversion to intermediate repesentations (IR and blobs)
* getting the model into the Azure model store
* moving the model from the model store to Percept DK

Much of the recipe is from [Azure Percept training notebook](https://github.com/microsoft/azure-percept-advanced-development/blob/main/machine-learning-notebooks/train-from-scratch/SemanticSegmentationUNet.ipynb). 

I did the training step locally on my machine (with a Nvidia 2080 card) instead of Azure ML.

To get a feel for the demo, [watch a 30-second clip](https://www.youtube.com/watch?v=2mIZ-Qxhjr8).

#### Setup below (mimicking a sushi production line). 
![setup](/assets/IMG_1090-cropped.JPG)

### Training Dataset
My initial training dataset size was 25 images and they are all in the  [training dataset](/resized_images/) folder.
For increasingly better segmenation, you can certainly add more images and their corresponding masks. 

Below is a sample image.

<img src="/resized_images/IMG_1052-size_818_616.jpg" alt="sample image" width="400"/>

You'll find all the masks in the [masks for training dataset](/resized_masks/) folder.

Below is a sample mask.

<img src="/resized_masks/IMG_1052-size_818_616.png" alt="sample mask" width="400"/>


More documentation coming...
