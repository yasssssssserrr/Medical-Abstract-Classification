# Medical-Abstract-Classification
> Goal: build and compare **LLM-based** pipelines to classify medical abstracts into **5 disease categories** using representation models, embeddings, generative models, and clustering.

## Overview
This repository implements all four approaches requested in the brief:
1) **Representation models** (Hugging Face `transformers`)  
   - **Without fine-tuning**: direct `pipeline` evaluation  
   - **With fine-tuning**: `Trainer` on the training split
2) **Embeddings** (Sentence-Transformers)  
   - **Supervised**: sentence embeddings → **Logistic Regression**  
   - **Zero-shot**: cosine similarity between abstract embeddings and **label embeddings**
3) **Generative models** (encoder-decoder)  
   - Prompted inference; parse outputs → final labels
4) **Clustering & topic modelling**  
   - **UMAP** ↓ dimension, **K-Means** (k=5), KNN labelling of test samples

##  Dataset

- **Source**: Hugging Face dataset “TimSchopf/medical_abstracts” (NLPIR 2022).  
- **Size / splits**: **14 438** total → **11 550** train, **2 888** test.  
- **Labels (5 classes)**: Neoplasms; Digestive System Diseases; Nervous System Diseases; Cardiovascular Diseases; General Pathological Conditions.  
- **Fields**: `medical_abstract` (text), `condition_label` (1–5).  
- **Local cache**: pre-materialized to Parquet: `04_AbstractClassification/HuggingFaceData/train.parquet`, `test.parquet`, `labels.parquet`.

Details and preprocessing instructions (rename `condition_label`→`label`, shift to 0–4, mapping to textual labels) are in the PDF.

## How to run

1. Open 04_AbstractClassification/AbstractClassification_LLMs.ipynb (local or Colab).
2. Data: Notebook loads Parquet from 04_AbstractClassification/HuggingFaceData/. To fetch directly from HF, enable the datasets.load_dataset cell.
3. Execute sections in order:
- Preprocessing & EDA (renaming, label shift, distributions)
- Representation models: pipeline → Trainer
- Embeddings: supervised + zero-shot
- Generative: prompt design, inference, mapping
- Clustering: UMAP → K-Means, 2D plots, KNN labelling
4. Evaluation: report Accuracy / F1 / Precision / Recall on test, compare trade-offs (quality vs. cost/latency)
