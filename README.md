# CSCK507 End of Module Assignment - Generative ChatBot

Quick overview of project structure:
- Notebooks stored in project root (`data-preprocessing.ipynb` is the only one that I've worked on so far).
- Raw kaggle datasets AND outputs from the preprocessing notebook stored under `data` folder.
- Graphs, plots, etc saved to `visualisations` folder (please feel free to create visuals as you go, it may be good content for the final report).
- Project dependencies in `requirements.txt`.

## Project Setup
I initialised a Python venv using version `3.14` for my local development. Here are the steps if you haven't done this before. Feel free to use uv or conda or whatever else you like (project dependencies are in requirement.txt).

1. Create you virtual environment using `python3.14 -m venv .venv`
2. Activate it with `source .venv/bin/activate`
3. Install dependencies with `pip install -r requirements.txt`
4. Boot up Jupyter lab with `jupyter lab`

---

## Git LFS

This project will likely accumulate large files (csvs, json files, model checkpoints, state dicts, etc) that we want to share with each other. [Git LFS](https://git-lfs.com/) is a convenient way to keep these large BLOBs in version control. Here's how to set it up:
```sh
# macOS
brew install git-lfs

# Windows
# Download installer from https://git-lfs.github.com/

# Linux
sudo apt-get install git-lfs
```

After cloning or pulling, LFS files are not pulled by default. Make sure to:
```sh
# Install git hooks and initialise LFS locally
git lfs install
git lfs pull
```

## Kaggle

Kaggle is included as a project dependency. If you configure your API key, you can delete the `/data/Ubuntu-dialogue-corpus` folder and run the first cell in `data-preprocessing.ipynb` to redownload [the dataset](https://www.kaggle.com/datasets/rtatman/ubuntu-dialogue-corpus). Note this will include 3 CSV files totalling around 3GB. I have only tracked `dialogueText.csv` in LFS as this dataset is more than sufficient for the time being, but if you'd like to look at the rest of the conversation turns, feel free to download them, just dont push to LFS please :)