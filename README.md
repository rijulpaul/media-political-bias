# Media Political Bias Classification

This project trains and evaluates NLP models to classify political bias in media text. It uses a sample dataset (20%) for initial experimentation and DistilBERT for full-scale training.

<div align="center">
  <a href="https://huggingface.co/rijulpaul/media-political-bias" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-rijulpaul-ffc107?color=ffc107&logoColor=white" />
  </a>
</div>

## Project Structure:
- [mediapoliticalbias_20.ipynb](https://github.com/rijulpaul/media-political-bias/blob/main/mediapoliticalbias_20.ipynb): Inital training and testing of multiple ML models on 20% dataset.
- [mediapoliticalbias_distilbert_final.ipynb](https://github.com/rijulpaul/media-political-bias/blob/main/mediapoliticalbias_distilbert_final.ipynb): Final training of DistilBERT on the full dataset.

## Usage
1. Open the notebook in Kaggle.
2. Add the dataset: [cymerunaslam/allthenews](https://www.kaggle.com/datasets/cymerunaslam/allthenews) from Kaggle.
3. Go to Kaggle → Settings → Secrets and create a secret named WANDB_API_KEY with your W&B key.
4. Run all cells.

# Methodology
- The project uses the [cymerunaslam/allthenews](https://www.kaggle.com/datasets/cymerunaslam/allthenews) dataset as the base corpus. Each article is labeled with political bias categories derived from the [AllSides Media Bias Chart](https://www.allsides.com/media-bias/media-bias-chart): Left, Lean Left, Center, Lean Right, and Right.
- Labels were assigned by mapping each article’s source to its corresponding bias rating from the [AllSides chart](https://www.allsides.com/media-bias/media-bias-chart). After labeling, the dataset was cleaned, balanced where needed, and prepared for both classical ML models and DistilBERT fine-tuning.
- A 20% split was used for initial experiments; full-scale training was done on the complete augmented dataset.
