<div align="center">
  <img src="doc/Gemini_Generated_Image_brvyepbrvyepbrvy.png" alt="Banner" width="100%">
</div>

---

<div align="center">

# Learning from Earnings Calls: Graph-Based Conversational Modeling for Financial Prediction

</div>

## Documentation

- **[Quick Start Guide](doc/quick_start.md)** - Get started with sample data in minutes
- **[Complete Pipeline](doc/pipeline.md)** - Detailed step-by-step guide for the entire workflow
- **[Dataset README](dataset/README.md)** - Information about the dataset structure

## Data

* [Source Data](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ytangch_connect_ust_hk/EVxg1Z5AEAxDs4df3uag9FwBEowKLZbAM4qplw5KdMa1XA?e=ucjTrQ) - Please download the data from this link.

## Project Structure

```
GraphConversationalAnalysis/
├── main.py                 # Main training and testing script
├── run_sent_lda.py        # Sentence-level LDA topic modeling script
├── model/                 # Model implementations
│   ├── GCA.py            # Graph Conversational Analysis model
│   └── MRQASelector.py   # Multi-round Q&A selector
├── trainer/              # Training scripts
│   └── CGATrainer.py     # Main trainer for CGA model
├── utils/                # Utility functions
│   ├── attention.py      # Attention mechanisms
│   ├── graph.py          # Graph neural network layers
│   ├── optim.py          # Optimizer utilities
│   ├── sequential.py     # Sequential model utilities
│   ├── utils.py          # General utilities
│   └── print_rawtext_by_path.py  # Text processing utilities
└── doc/                  # Documentation directory
```

## Requirements

The code requires Python 3.8+ and the following packages:

- PyTorch (with CUDA support recommended)
- NumPy, Pandas
- Transformers (Hugging Face)
- Sentence Transformers
- NLTK, Spacy
- Gensim (for LDA)
- scikit-learn, SciPy
- NetworkX
- Matplotlib, Seaborn
- TensorBoardX
- PyYAML, tqdm

## Quick Installation

```bash
# Create conda environment
conda create -n gca python=3.10
conda activate gca

# Install dependencies
pip install torch torchvision
pip install numpy pandas scikit-learn scipy
pip install transformers sentence-transformers
pip install gensim nltk spacy
pip install networkx matplotlib seaborn
pip install tensorboardX pyyaml tqdm
```

For detailed setup instructions, see [Complete Pipeline](doc/pipeline.md#prerequisites).

## Quick Start

```bash
# 1. Preprocess data
python preprocess.py \
    --dir dataset/earningscall \
    --out_dir dataset/processed \
    --out_filename processed_all.csv

# 2. Train LDA model and generate training data
python run_sent_lda.py \
    --in_path dataset/processed/processed_all.csv \
    --out_dir dataset/processed \
    --out_filename nlp_processed.csv \
    --need_train True \
    --need_topic_rep True \
    --doc LDA_0707

# 3. Train model
python main.py \
    --config topic_profet_gnn.yml \
    --trainer GCATrainer

# 4. Test model
python main.py \
    --config topic_profet_gnn.yml \
    --trainer GCATrainer \
    --test True
```

See [Quick Start Guide](doc/quick_start.md) for more details.

## Data Format

The model expects earnings call transcripts in the following structure:

**Transcript CSV files:**
- `role`: Speaker role (analyst/executive)
- `name`: Speaker name  
- `sentence`: Text content
- `section`: Section type (intro/qa)

**Label CSV file (updated_index_with_DVs.csv):**
- `path`: Relative path to transcript
- `ticker`, `quarter`, `year`: Identifying information
- `firm_std_X_pre/post`: Pre/post-event volatility metrics

See [Complete Pipeline](doc/pipeline.md#data-structure) for detailed format specifications.

## Configuration

Model configuration is specified in YAML files. Example `conf/topic_profet_gnn.yml`:

```yaml
train:
  batch_size: 64
  n_epochs: 12
  loss: "mse_loss"

model:
  embedding_size: 384
  hidden_size: [300, 200, 100]
  dropout: 0.1

data:
  train_path:
    - 'dataset/processed/topic_sent_bert_based_data_2015.pkl'
    - 'dataset/processed/topic_sent_bert_based_data_2016.pkl'
    - 'dataset/processed/topic_sent_bert_based_data_2017.pkl'
  test_path:
    - 'dataset/processed/topic_sent_bert_based_data_2018.pkl'
  label: 'firm_std_60_post'
  batch_size: 16

optim:
  optimizer: 'Adam'
  lr: 0.0015
  weight_decay: 0.00001

checkpoint:
  choice: 'best'
```

## Citation

If you use this code, please cite the corresponding paper.

```
@article{doi:10.1287/isre.2023.0519,
    author = {Yang, Yi and Tang, Yixuan and Fan, Yangyang and Zhang, Kunpeng},
    title = {Learning from Earnings Calls: Graph-Based Conversational Modeling for Financial Prediction},
    journal = {Information Systems Research},
    volume = {0},
    number = {0},
    pages = {null},
    year = {0},
    doi = {10.1287/isre.2023.0519},
    URL = { https://doi.org/10.1287/isre.2023.0519},
    eprint = { https://doi.org/10.1287/isre.2023.0519}
}
```

## License

See LICENSE file for details.
