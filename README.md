# Disaster damage assessment

## Overview
Damage assessments during and after natural disasters are core activities for humanitarian organizations. They are used to increase situational awareness during the disaster or to provide actionable information for rescue teams. In post-disasters, they are essential to evaluate the financial impact and the recovery cost.

However, effectively implementing damage assessment, using surveys, represents a challenge for several organizations as it takes weeks and months and is logistically difficult for rapid assessment. In recent years, a lot of research has been done on using AI and machine learning to automate classification of the damages. Those studies and systems use social media imagery data such twitter text and images, aerial and satellite images to assess and map damages during and after a disaster. For the AWS disaster response hackathon one of the challenge is to answer the question :How might we accurately and efficiently determine the extent of damage to individual homes in a given disaster-impacted area?  

To answer that question our team proposed to **train and deploy a proof of concept image classifier for damage assessment**. We built a simple app that would allow users to load an image of a building, road or bridge  and classify the extend of damage cause by an earthquake. The severity of damage in an image is the extent of physical destruction shown in it. For this experiment we only consider three level of damages :

* severe damage
* mild damage
* no damage (none)

## What's next

We believe this proof of concept could evolve in several types of application for rapid assessment during disaster or post disaster. It could be used by rescue teams, assessors or victims of natural disasters.

Also, extended with some ground truth financial data from other resources we think this type of appllication could help estimate the USD equivalent of the infrastructural damage. Other features can be added in the future such as: build an edge device app for smartphone or drones, visual question answering to extract actionable information from a central database, a no-code deep learning platform  to assist humanitarian organization to build thier own model for specific natural disaster event.

## Our proof of concept

Our demo does not consider aerial and satellite images and does not include location and mapping capabilities. The model deploy was mainly trained on earthquake images.

You can access and test our app here:[disaster-damage-classifier](https://huggingface.co/spaces/mayerantoine/disaster-damage-classifier) on Hugging face spaces deployed as a Gradio app.

This project is inspired and built on top of existing work from the manuscripts: “*Damage Assessment from Social Media Imagery Data During Disasters*”  and “*Automatic Image Filtering on Social Networks Using Deep Learning and Perceptual Hashing During Crises*” from ( D. T. Nguyen et al) .

### Training and Data

We trained the model using social media imagery data using robust and recent pre-trained computer vision neural networks such as EfficientNet. The public data used come from the manuscript which comprises images collected from Twitter during four natural disasters, namely Typhoon Ruby (2014), Nepal Earthquake (2015), Ecuador Earthquake (2016), and Hurricane Matthew (2016). In addition to Twitter images, it contains images collected from Google using queries such as "damage building".

To improve our model we include data from the Haiti Earhtquakes 2010 and 2021, and created a cross-event earthquakes dataset using Nepal, Haiti, Google , 30% of Equator data for training  and 70% of Equator Earthquake data for validation and testing.

We use Amazon SageMaker Studio and Studio Lab  with Tensorflow to build and train the model.

## Setup

### 1. Check out the repo

```
git clone https://github.com/mayerantoine/disaster-damage-assessment-ml.git
cd disaster-damage-assessment-ml
```

### 2. Set up the Python environment

We use `conda` and `pip` for managing Python and installed dependencies in the file `environment.yml`.
Run the command to create our conda env.

```
conda env create --file environment.yml
```

Note: conda is an open-source package management system and environment management system that runs on Windows, macOS, and Linux.  To install `conda`, follow instructions at https://conda.io/projects/conda/en/latest/user-guide/install/linux.html

Next, activate the conda environment.

```sh
conda activate disaster-damage-assessment
```

### 3. Download dataset

We need to download the dataset and also plan 4GB for the tar file and 4GB for the extracted images(a total 8GB)
```
python download_dataset.py
```

**Note**: on SageMaker Studio lab with only 15GB of free storage, please check space avaialble and remove trash files

``` 
df -h
du -h /home/studio-lab-user/
rm -r /home/studio-lab-user/.local/share/Trash/files
```

#### 3.1 Haiti dataset

If you want to use the cross-event dataset which include haiti data you need to manually download the data on kaggle [here](https://www.kaggle.com/mayerantoine/haiti-damage-assessment). You can skip this if not needed.

Run this command to move the images and create the cross-event haiti dataset.

```
python cross_event_haiti.py
```

### 4. Run Experiment

Run the following command to train the model on a specific natural disaster events or cross disaster events:

1. Equator  
2. Nepal
3. Haiti
4. Google
5. Matthew
6. Ruby
7. Cross events: Nepal, Google and Equator (Earthquakes)  data available in `data/damage_csv/cross_event_ecuador`
8. Cross events: Nepal, Google, Haiti and Equator (Earthquakes) data available in `data/damage_csv/cross_event_ecuador_haiti`
9. Cross events: Ruby, Google and Matthew (Typhoons) data available in `data/damage_csv/cross_event_matthew`

```
python run_experiment.py --event cross_event_ecuador --model efficientnet --epochs 5 --lr 1e-3 --frac 0.1 --finetune
```
The training parameters are :
* `event` which natural disaster event. The events supported are the folders in `data/damage_csv`
* `model` which model such as : efficientnet or mobilenet
* `epochs` number epochs
* `lr` learning rate
* `batch`  batch size ,default is 32
* `frac` proportion of data to use between 0 and 1.
* `finentune`  to include finetuning after the training

The model will be saved in the folder `outputs/model`.


### 5.  Check the notebook

You can also open and run the following notebook: `/notebooks/02-Damage_assessment_using_EfficientNet_v3.ipynb`.

## Important links and references
1. [Using Social Media Imagery for Disaster Response](https://crisiscomputing.qcri.org/2018/03/04/using-social-media-imagery-for-disaster-response/)

2. [Damage Assessment from Social Media Imagery Data During Disasters](https://ieeexplore.ieee.org/document/9069136)

3. [Automatic Image Filtering on Social Networks Using Deep Learning and Perceptual Hashing During Crises](https://arxiv.org/abs/1704.02602)
