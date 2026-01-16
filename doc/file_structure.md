# File Structure and Documentation

This document provides a comprehensive overview of all files and directories in the GraphConversationalAnalysis project.

## Root Directory

```
GraphConversationalAnalysis/
├── main.py                 # Main entry point for training and testing
├── preprocess.py           # Data preprocessing script
├── run_lda.py             # Document-level LDA topic modeling
├── run_sent_lda.py        # Sentence-level LDA topic modeling
├── run_pipeline.sh        # Complete pipeline automation script
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── LICENSE               # License information
└── README.md             # Project overview
```

## Core Scripts

### main.py
Main training and testing script that orchestrates the entire model workflow.

**Key Functions:**
- `dict2namespace()`: Converts config dictionaries to namespace objects
- `parse_args_and_config()`: Parses command-line arguments and YAML config
- Training loop initialization
- Model checkpoint management

**Usage:**
```bash
python main.py --config <config_file> --trainer GCATrainer [--test True]
```

### preprocess.py
Filters and validates earnings call transcripts.

**Key Functions:**
- `read_origin_rawtext_data()`: Reads transcript CSV files
- `preprocess_data()`: Main preprocessing pipeline
- Validates presence of presentation and Q&A sections
- Creates processed dataset

**Usage:**
```bash
python preprocess.py --dir dataset/earningscall --out_dir dataset/processed
```

### run_sent_lda.py
Sentence-level LDA topic modeling and representation generation.

**Key Functions:**
- `read_dataset()`: Loads and processes transcripts
- `multi_thread_process_list()`: Parallel text preprocessing
- LDA model training using Mallet
- Topic representation generation with sentence-BERT
- Training data generation with sentence-BERT embeddings

**Usage:**
```bash
# Train LDA model and generate training data (recommended)
python run_sent_lda.py \
    --in_path dataset/processed/processed_all.csv \
    --out_dir dataset/processed \
    --out_filename nlp_processed.csv \
    --need_train True \
    --need_topic_rep True \
    --doc LDA_0707
```

### run_lda.py
Document-level LDA topic modeling (alternative to sentence-level).

**Key Functions:**
- Similar to `run_sent_lda.py` but operates at document level
- Used for different granularity of topic analysis

## Model Directory

```
model/
├── __init__.py           # Model package initialization
├── GCA.py               # Graph Conversational Analysis model
└── MRQASelector.py      # Multi-round Q&A selector
```

### model/GCA.py
Main GCA model implementation.

**Classes:**
- `TopicProfetGNN`: Graph neural network with topic-based representations
  - Processes presentation and Q&A separately
  - Applies GNN layers (GCN, GAT, or Gated GNN)
  - Generates volatility predictions

**Key Methods:**
- `forward()`: Main forward pass
- `gnn_based_forward()`: GNN-based prediction
- `create_pre_a_pair_graph()`: Creates graph structure

### model/MRQASelector.py
Multi-round Q&A selection and attention mechanisms.

**Classes:**
- `SentenceSelector`: Selects relevant sentences from Q&A
- `QA_Attention`: Applies attention between questions and answers
- `MRQA_simple`: Complete Q&A processing module

## Trainer Directory

```
trainer/
├── __init__.py          # Trainer package initialization
└── CGATrainer.py        # Main GCA trainer
```

### trainer/CGATrainer.py
Complete training pipeline for the GCA model.

**Classes:**
- `GCATrainer`: Main trainer class

**Key Methods:**
- `train()`: Training loop
- `test()`: Testing and evaluation
- `test_one_firm_with_interpretability()`: Interpretability analysis
- `multi_thread_load_dataset()`: Parallel data loading

**Features:**
- Early stopping
- Checkpoint saving
- TensorBoard logging
- Multi-metric evaluation (MSE, MAE, Spearman, Kendall)

## Utils Directory

```
utils/
├── __init__.py                    # Utils package initialization
├── attention.py                   # Attention mechanisms
├── graph.py                       # Graph neural network layers
├── optim.py                       # Optimizer utilities
├── sequential.py                  # Sequential model components
├── utils.py                       # General utilities
└── print_rawtext_by_path.py      # Text processing utilities
```

### utils/attention.py
Various attention mechanisms.

**Classes:**
- `Response_score`: Additive attention for responses
- `CrossRef_score`: Cross-reference attention
- `SentenceSelector`: Sentence selection with attention
- `QA_Attention`: Q&A attention mechanism
- `MRQA_simple`: Multi-round Q&A processing

### utils/graph.py
Graph neural network layer implementations.

**Classes:**
- `GraphConvolution`: Standard GCN layer
- `GCN`: Multi-layer graph convolution network
- `GraphAttentionLayer`: Single GAT layer
- `GAT`: Multi-layer graph attention network
- `GatedGNN`: Gated graph neural network

### utils/optim.py
Optimizer configurations and learning rate scheduling.

**Functions:**
- `get_optimizer()`: Returns configured optimizer
- Learning rate scheduling utilities

### utils/sequential.py
Sequential model utilities for building layer stacks.

**Functions:**
- `make_seq()`: Creates sequential layers
- `make_ff()`: Creates feedforward networks

### utils/utils.py
General utility functions.

**Functions:**
- `prepare_device()`: Sets up GPU/CPU device

## Configuration Directory

```
conf/
└── topic_profet_gnn.yml    # Model configuration file
```

### conf/topic_profet_gnn.yml
YAML configuration file for model hyperparameters.

**Sections:**
- `train`: Training settings (batch_size, n_epochs, loss)
- `model`: Architecture settings (embedding_size, hidden_size, dropout)
- `data`: Data paths (train_path, test_path) and label selection
- `optim`: Optimizer parameters (optimizer, lr, weight_decay)
- `scheduler`: Learning rate scheduling settings
- `checkpoint`: Checkpoint saving strategy (best or latest)

## Documentation Directory

```
doc/
├── pipeline.md          # Complete step-by-step pipeline guide
├── quick_start.md      # Quick start guide
└── file_structure.md   # This file
```

### doc/pipeline.md
Comprehensive guide covering:
- Prerequisites and environment setup
- Data structure explanation
- Step-by-step execution instructions
- Troubleshooting guide

### doc/quick_start.md
Quick reference for:
- Fast setup commands
- Expected execution times
- Output verification
- Common issues

## Dataset Directory

```
dataset/
├── earningscall/
│   ├── 0/ to 7/                  # Transcript subdirectories by year
│   └── update_index_Volatility.csv # Labels and metadata
├── processed/                     # Generated during preprocessing
│   ├── processed_all.csv         # Preprocessed data index
│   ├── nlp_processed.csv # Sentence-level corpus for LDA
│   ├── sent_lda_rep_v2_0707.pkl  # LDA topic representations
│   └── topic_sent_bert_based_data_*.pkl # Training/validation/test data
├── stopwords.txt                 # Custom stopwords list
└── README.md
```

See [dataset/README.md](../dataset/README.md) for detailed data documentation.

## Result Directory (Generated)

```
result/
├── checkpoint/
│   ├── GCATrainer/              # Model checkpoints
│   │   ├── best_checkpoint.pth
│   │   └── latest_checkpoint.pth
│   └── LDA_0707/                # LDA models
│       ├── sent_lda.model
│       └── sent_lda_id2word.pkl
├── log/
│   └── GCATrainer/
│       └── stdout.txt           # Training logs
├── tensorboard/
│   └── GCATrainer/              # TensorBoard logs
└── output/
    └── inter_topic_neg.pkl      # Interpretability results
```

## Package Directory (User-Created)

```
packages/
└── mallet-2.0.8/                # Mallet installation for LDA
```

## File Naming Conventions

### Python Files
- Use snake_case for filenames: `preprocess.py`, `run_sent_lda.py`
- Class names use PascalCase: `TopicProfetGNN`, `GCATrainer`
- Function names use snake_case: `preprocess_data()`, `multi_thread_load_dataset()`

### Data Files
- CSV files: descriptive names with underscores
  - `processed_all.csv`
  - `updated_index_with_DVs.csv`
  - `nlp_processed_sent_lda.csv`
- Pickle files: include version/date for tracking
  - `sent_lda_rep_v2_0707.pkl` (LDA topic representations)
  - `topic_sent_bert_based_data_2015.pkl` (training data)
  - `topic_sent_bert_based_data_2016.pkl` (training data)
  - `topic_sent_bert_based_data_2017.pkl` (validation data)
  - `topic_sent_bert_based_data_2018.pkl` (test data)

### Configuration Files
- YAML files: lowercase with underscores
  - `topic_profet_gnn.yml`

### Documentation Files
- Markdown files: lowercase with underscores
  - `pipeline.md`, `quick_start.md`
- No all-caps filenames (e.g., avoid `README.MD`)

## Import Structure

### Main Imports
```python
from trainer import *              # Imports all trainers
from model import *                # Imports all models
from utils.utils import *          # Imports utilities
```

### Model Imports
```python
from model.GCA import TopicProfetGNN
from model.MRQASelector import MRQA_simple, SentenceSelector, QA_Attention
```

### Trainer Imports
```python
from trainer.CGATrainer import GCATrainer
```

## Dependencies

### Core Dependencies
- PyTorch (torch, torch.nn, torch.optim)
- NumPy, Pandas
- transformers, sentence-transformers
- gensim (for LDA)

### Graph Libraries
- NetworkX
- Custom GNN implementations (utils/graph.py)

### NLP Libraries
- NLTK (tokenization, stopwords)
- Spacy (advanced NLP)
- contractions, demoji

### Training Utilities
- tensorboardX (logging)
- tqdm (progress bars)
- PyYAML (config parsing)

## Code Organization Principles

1. **Separation of Concerns**: Models, trainers, and utilities are in separate directories
2. **Modularity**: Each file has a specific purpose
3. **Reusability**: Common functions are in utils/
4. **Configuration**: Hyperparameters externalized to YAML files
5. **Documentation**: Comprehensive docstrings and markdown docs

## Adding New Components

### Adding a New Model
1. Create file in `model/` directory
2. Import in `model/__init__.py`
3. Follow naming conventions (PascalCase for classes)

### Adding a New Trainer
1. Create file in `trainer/` directory
2. Import in `trainer/__init__.py`
3. Update `main.py` if needed

### Adding New Utilities
1. Add to appropriate file in `utils/`
2. Or create new file for new functionality
3. Import in `utils/__init__.py` if needed

## Version Control

Files tracked in git:
- All source code (`.py` files)
- Configuration files (`.yml` files)
- Documentation (`.md` files)
- Requirements (`requirements.txt`)
- Scripts (`.sh` files)

Files ignored (see `.gitignore`):
- `__pycache__/`, `*.pyc`
- `result/` directory
- `dataset/processed/`
- Model checkpoints (`.pth`, `.pkl`)
- Virtual environments
- IDE-specific files

## Related Documentation

- [Complete Pipeline](pipeline.md) - Detailed execution guide
- [Quick Start](quick_start.md) - Fast setup instructions
- [Dataset README](../dataset/README.md) - Data documentation
- [Main README](../README.md) - Project overview

