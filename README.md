# Google Cloud Translation API Analysis
## Introduction

## Architecture Overview

## Code Walkthrough
### Working Environment Set Up
#### Import relevant libraries
```python
import pandas as pd
import os
import sklearn
import re
from google.cloud import translate_v2
from google.cloud import translate
from google.colab import files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```
#### Upload api credentials as a json file and store it as an environmental variable called "GOOGLE_APPLICATION_CREDENTIALS"
```python
files.upload()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "api_credentials.json"
```
#### Define a function that converts two texts into all lowercase, remove punctuations, and computes the cosine similarity score of two texts
```python
def text_similarity(text1, text2):
  # remove punctuations and make text all lowercase
  text1 = re.sub('[^\w\s]', '', text1).lower()
  text2 = re.sub('[^\w\s]', '', text2).lower()

  vectorizer = TfidfVectorizer()
  vectors = vectorizer.fit_transform([text1, text2])
  similarity = cosine_similarity(vectors)
  return similarity[1,0]
```
  

### Data Collection

### Data Analysis

## Analysis
### Explanatory Variables

### Response Variable
