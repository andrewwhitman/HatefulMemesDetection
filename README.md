# Hateful Memes Detection
![Mean memes from https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set](https://scontent-ort2-1.xx.fbcdn.net/v/t39.2365-6/96215346_2581440138810047_7336682629883756544_n.png?_nc_cat=107&ccb=1-5&_nc_sid=ad8a9d&_nc_ohc=p5xSTm2IWxMAX-q_C84&_nc_ht=scontent-ort2-1.xx&oh=e5c1e3dc76873b31dc7303cb1899e3cb&oe=61AE36B4)
Photo from [Facebook AI](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set)

## Overview

This repo


## Business and Data Understanding

The data comes from [Facebook AI's Hateful Memes Challenge](https://hatefulmemeschallenge.com/).

See the [Reproducibility](##Reproducibility "Go to Reproducibility") section of this README to access the data.


## Modeling and Evaluation

Text here


## Conclusions

Text here


## Information

Check out this [notebook](https://github.com/andrewwhitman/HatefulMemesDetection/blob/main/HatefulMemesDetection.ipynb) for a more thorough discussion of the project, as well as this [presentation](https://github.com/andrewwhitman/HatefulMemesDetection/blob/main/presentation.pdf).

## Reproducibility

To download the data, visit the [challenge website](https://hatefulmemeschallenge.com/#download) and agree to the dataset license by filling out the form. The data (3.93 GB) will be downloaded as a compressed `.zip` folder. Unzip and store the contents (includes both an `img` folder and five `.jsonl` files) in a folder named `raw`. Move the `raw` folder into the `data/` directory of this repository.

In addition to the dataset, GloVe embeddings need downloaded. Follow this [link](https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip) to download the compressed `glove.6B.zip` folder. Unzip and store the contents in a folder named `glove.6B`. Move the `glove.6B` folder into the `data/` directory of this repository.

To reproduce an environment with the necessary dependencies locally on your computer, use the `environment.yml` in this repo with `conda`. For `pip` compatibility, use the `requirements.txt` to install the dependencies.

## Repository Structure

```

├── data                           <- placeholder folder for local storage of data
│   └── ...
├── notebooks                      <- contains additional notebooks for data exploration and modeling
│   └── ...
├── .gitignore                     <- specifies files/directories to ignore
├── environment.yml                <- reproduces the environment
├── HatefulMemesDetection.ipynb    <- notebook detailing the data science process containing code and narrative
├── presentation.pdf               <- presentation slides for a business audience
├── README.md                      <- Top-level README
├── requirements.txt               <- reproduces the environment
└── utils.py                       <- contains helper functions and classes

``` 