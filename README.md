# BDS_2025_M3_Exercise2

# Fine-Tuning a Transformer Model for Multi-Class Text Classification

This project demonstrates how to fine-tune a transformer model (BERT-based) on the AG News dataset for a multi-class text classification task. The model is trained to predict the category of news articles (4 possible classes) based on the text content.

## Project Overview

The goal of this project is to fine-tune a pre-trained transformer model for a text classification task, using the AG News dataset. We will:
- Load and preprocess the dataset.
- Tokenize the text data for transformer input.
- Fine-tune a pre-trained BERT model.
- Evaluate the model's performance using accuracy, precision, and recall metrics.
- Upload the final model to Hugging Face.

### Steps Involved:
1. **Dataset**: The AG News dataset is loaded from Hugging Face and contains 4 categories of news articles.
2. **Data Preprocessing**: Text data is tokenized using the `AutoTokenizer` from Hugging Face.
3. **Model Fine-Tuning**: A pre-trained BERT model (`bert-base-cased`) is fine-tuned for the classification task.
4. **Evaluation**: We evaluate the model's performance using standard classification metrics (accuracy, precision, recall).
5. **Model Upload**: Once fine-tuned, the model is saved and uploaded to Hugging Face for easy sharing and deployment.

## Dataset

The AG's news topic classification dataset is constructed by choosing 4 largest classes from the original corpus. Each class contains 30,000 training samples and 1,900 testing samples. The total number of training samples is 120,000 and testing 7,600.

The file classes.txt contains a list of classes corresponding to each label.

The files train.csv and test.csv contain all the training samples as comma-sparated values. There are 3 columns in them, corresponding to class index (1 to 4), title and description. The title and description are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). New lines are escaped by a backslash followed with an "n" character, that is "\n".

The labels are:
1: World 2: Sports 3: Business 4: Sci/Tech

The dataset is available from [Hugging Face Datasets](https://huggingface.co/datasets/ag_news).

## Installation

To set up the environment and run this project, you need to install the following dependencies:

```bash
pip install transformers datasets evaluate huggingface_hub



