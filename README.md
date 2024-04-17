
# Retrieval-Augmented Generation: Impact on Information Retrieval Systems for SMEs

This repository contains the source code for the case study exploring the effects of Retrieval-Augmented Generation (RAG) on Information Retrieval Systems tailored for Small and Medium-Sized Enterprises (SMEs) as part of my bachelor thesis.

## Getting Started

To begin examining the results yourself, follow these steps to set up the environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/nneubacher/Bachelorarbeit.git
   ```
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Repository Contents

- `data/`: Directory containing the Stanford Question Answering Dataset.
- `chromaDB/`: Directory containing the vector store with the embeddings of SQuAD.
- `toChroma.py`: Script for embedding and storing data from the `data` directory into the `chromaDB` vector store.
- `RAG.py`: Evaluation script for the RAG-based information retrieval system.
- `noRAG.py`: Evaluation script for the base model information retrieval system.
- `compare.ipynb`: Jupyter notebook for interactive analysis and comparison.
- `compare.py`: Python script for comparing total correct predictions at different thresholds.
- `predictions.json`: Output from the RAG-based Information Retrieval system.
- `predictions_gpt.json`: Output from the Information Retrieval system using just the base model.

## Usage

After setting up, you can run `compare.py` or explore `compare.ipynb` to see how different configurations of the Information Retrieval system perform with respect to your datasets and queries.
