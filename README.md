# CSCK507 End of Module Assignment - Generative ChatBot

## Project structure
```
├── bleu.py                    # BLEU score utilities for model evaluation
├── decoder.py                 # Decoder module for the seq2seq model
├── encoder.py                 # Encoder module for the seq2seq model
├── inference.py               # Inference / chatbot response generation
├── seq2Seq.py                 # Main seq2seq model wiring encoder + decoder
├── train.py                   # Training loop and model training utilities
├── utils.py                   # Shared helper functions
├── data-analysis.ipynb        # Exploratory analysis of the dataset
├── data-preprocessing.ipynb   # Data cleaning, preprocessing, and export
├── embedding_comparison.ipynb # Notebook for comparing embeddings
├── evaluation.ipynb           # Model evaluation and metrics
├── model-a.ipynb              # Notebook for Model A training
├── model-b.ipynb              # Notebook for Model B training
├── data/                      # Raw and processed dataset files
├── models/                    # Saved model artifacts
├── visualisations/            # Plots and charts generated during analysis
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview and setup instructions
```

## Project Setup
Initialise a Python virtual environment (for example, using version 3.14) for local development. Project dependencies are in `requirement.txt`.

1. Create a virtual environment: `python3.14 -m venv .venv`
2. Activate it: `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Boot up Jupyter Lab: `jupyter lab` or open in your IDE: `code .`.

---

## Git LFS

This project manages large files (CSVs, models and state dict checkpoint PT files, PNGs, JSON files). [Git LFS](https://git-lfs.com/) is a convenient way to keep these large BLOBs in version control. After cloning the repository, setup Git LFS:
```sh
# macOS
brew install git-lfs

# Windows
# Download installer from https://git-lfs.github.com/

# Linux
sudo apt-get install git-lfs
```

Pull the latest LFS files from the remote repository:
```sh
# Install git hooks and initialise LFS locally
git lfs install
git lfs pull
```

## Kaggle

Kaggle is included as a project dependency. If you configure your API key, you can delete the `/data/Ubuntu-dialogue-corpus` folder and run the first cell in `data-preprocessing.ipynb` to redownload [the dataset](https://www.kaggle.com/datasets/rtatman/ubuntu-dialogue-corpus). Note this will include 3 CSV files totalling around 3GB. This project uses `dialogueText.csv`, and this is the only file tracked by LFS.