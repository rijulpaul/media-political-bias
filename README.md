# Media Political Bias Classification

This project trains and evaluates NLP models to classify political bias in news articles. It uses a sample dataset (20%) for initial experimentation and DistilBERT for full-scale training.

<!-- <div align="center"> -->
  Try it on:  
  <a href="https://huggingface.co/spaces/rijulpaul/news-political-bias" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-rijulpaul-ffc107?color=ffc107&logoColor=white" />
  </a>
<!-- </div> -->

## Project Structure:
```
media-political-bias/
│
├── README.md
├── requirements.txt
|
├── notebooks/
│   ├── mediapoliticalbias_20.ipynb          # initial model comparison (20% data)
│   ├── mediapoliticalbias_linearsvc.ipynb   # final LinearSVC training
│   └── mediapoliticalbias_distilbert.ipynb  # final DistilBERT training
│
├── models/
│   ├── linearsvc/
│   └── distilbert/
|
└── main.py                      # CLI entry point (inference only)
```

## Usage
```bash
git clone https://github.com/rijulpaul/media-political-bias.git
cd media-political-bias
```
### Notebooks (Training & Experiments)
1. Open the desired notebook in Kaggle.
2. Add the dataset: [cymerunaslam/allthenews](https://www.kaggle.com/datasets/cymerunaslam/allthenews) from Kaggle.
3. Go to Kaggle → Settings → Secrets and create a secret named WANDB_API_KEY with your W&B key.
4. Run all cells.

### CLI Interface
#### Create and activate a virtual environment (recommended)
**Linux/MacOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```
#### Install dependencies
```python
pip install --upgrade pip
pip install -r requirements.txt
```
#### Run the script
```python
# Help
python main.py --help
# Run both models (default)
python main.py --file article.txt
# DistilBERT only
python main.py --model distilbert --file article.txt
# LinearSVC only
python main.py --model linearsvc < article.txt
```

# Methodology
- The project uses the [cymerunaslam/allthenews](https://www.kaggle.com/datasets/cymerunaslam/allthenews) dataset as the base corpus. Each article is labeled with political bias categories derived from the [AllSides Media Bias Chart](https://www.allsides.com/media-bias/media-bias-chart): Left, Lean Left, Center, Lean Right, and Right.
- Labels were assigned by mapping each article’s source to its corresponding bias rating from the [AllSides chart](https://www.allsides.com/media-bias/media-bias-chart). After labeling, the dataset was cleaned, balanced where needed, and prepared for both classical ML models and DistilBERT fine-tuning.
- A 20% split was used for initial experiments; full-scale training was done on the complete augmented dataset.

# Results & Observations
Two modeling approaches were evaluated: a fine-tuned DistilBERT transformer and a TF-IDF + LinearSVC classifier.
### DistilBERT
DistilBERT achieved strong performance on the validation and test splits derived from the training dataset. However, during qualitative evaluation on real-world news articles, the model exhibited limited generalization. This behavior is consistent with overfitting to dataset-specific language patterns and source biases, a common challenge when fine-tuning large transformer models on relatively homogeneous datasets.
### LinearSVC
The LinearSVC model showed slightly weaker performance on the held-out test set compared to DistilBERT. Despite this, it demonstrated more consistent behavior on real-world articles. The use of TF-IDF features combined with lemmatization likely contributed to improved robustness by reducing vocabulary sparsity and emphasizing core lexical signals over surface-level phrasing.
### Summary
While DistilBERT excels at capturing in-distribution patterns, the LinearSVC model provides more stable predictions in out-of-distribution settings. This highlights the trade-off between model capacity and generalization, and suggests that simpler models with stronger inductive biases can remain competitive for real-world deployment.
