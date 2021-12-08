# Hateful Memes Detection
![Mean memes from https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set](https://github.com/andrewwhitman/HatefulMemesDetection/blob/main/images/facebook_mean_memes.png)
Photo from [Facebook AI](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set). These are examples of "mean" memes, not actual hateful memes, which would be distasteful to display.

## Overview

Moderating online content, particularly the *multimodal* combination of text and images in the form of memes, is an extensive and difficult problem for social media and society. This project uses a curated dataset of memes from Facebook AI to detect whether a meme is hateful based on its textual content. The model accurately classifies 61% of memes, with an ROC-AUC of 0.578.


## Business and Data Understanding

Hate speech in the form of hateful memes is a challenging problem for automatic detection due to its multimodal combination of text and images. The semantic value of a meme is determined by both the text and image and their interaction. Facebook AI has curated a dataset of memes using human annotators to determine a meme's hateful classification. Benign confounders were created and added to the dataset by the researchers. These confounders take a meme that is hateful and either changes the text or image to switch the classification from hateful to not-hateful. See this [paper](https://arxiv.org/pdf/2005.04790.pdf) for further details on the curation of the dataset.

Distinguishing between the two classes is a priority for automatic detection as the consequences of false positives and false negatives are both troublesome. If content removal is based on such detection, removing user content that isn't hateful and keeping up truly hateful content will both be an issue. Therefore, the metrics to prioritize are accuracy, f1 score, and ROC-AUC.

The data can be retrieved from [Facebook AI's Hateful Memes Challenge](https://hatefulmemeschallenge.com/). See the [Reproducibility](#Reproducibility "Go to Reproducibility") section of this README to access the data.


## Modeling and Evaluation

Only the textual component of a meme is modeled using Naive Bayes classifiers based on frequency counts and TF-IDF, as well as logistic regression and random forest models based on GloVe word embeddings. The best-performing model of these is a Multinomial Naive Bayes classifier used on text that has been vectorized with frequency counts. This model achieves an accuracy score of 61% on holdout data, while a base rate classification based on the target distribution in the training set achieves an accuracy score of 53% on holdout data.


## Conclusions

The final model can be used in an automatic detection system for hateful memes. One recommendation is to tag memes predicted as hateful by the final model with a content warning message and blur. In addition, a team could manually review these predicated-hateful memes and either remove content that is truly hateful or restore content that was a false positive. Such manual review would be less of a workload compared to reviewing all memes without a first-pass automated tag. 

Continued work to incorporate the visual content of a meme is a next step for the project, along with using advanced language models like BERT for pre-training.


## Information

Check out this [notebook](https://github.com/andrewwhitman/HatefulMemesDetection/blob/main/HatefulMemesDetection.ipynb) for a more thorough discussion of the project, as well as this [presentation](https://github.com/andrewwhitman/HatefulMemesDetection/blob/main/presentation.pdf).

## Reproducibility

To download the data, visit the [challenge website](https://hatefulmemeschallenge.com/#download) and agree to the dataset license by filling out the form. The data (3.93 GB) will be downloaded as a compressed `.zip` folder. Unzip and store the contents (includes both an `img` directory and five `.jsonl` files) in a directory named `raw`. Move `raw` into the `data/` directory of this repository.

In addition to the dataset, GloVe embeddings need downloaded. Follow this [link](https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip) to download the compressed `glove.6B.zip` folder. Unzip and store the contents in a directory named `glove.6B`. Move `glove.6B` into the `data/` directory of this repository.

To reproduce an environment with the necessary dependencies locally on your computer, use the `environment.yml` in this repo with `conda`. For `pip` compatibility, use the `requirements.txt` to install the dependencies.

## Repository Structure

```

├── data                           <- placeholder folder for local storage of data
│   └── ...
├── images                         <- contains images for README and notebooks
│   └── ...
├── notebooks                      <- contains additional notebooks for data exploration and modeling
│   └── ...
├── .gitignore                     <- specifies files/directories to ignore
├── HatefulMemesDetection.ipynb    <- details the data science process with code and narrative
├── README.md                      <- Top-level README
├── environment.yml                <- reproduces the environment
├── presentation.pdf               <- presentation slides for a business audience
├── requirements.txt               <- reproduces the environment
└── utils.py                       <- contains helper functions and classes

``` 