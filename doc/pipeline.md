# Complete Pipeline Guide

This document provides a step-by-step guide to run the Graph-based Conversational Analysis (GCA) model for financial prediction using earnings call transcripts.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Data Structure](#data-structure)
3. [Step 1: Data Preprocessing](#step-1-data-preprocessing)
4. [Step 2: LDA Topic Modeling](#step-2-lda-topic-modeling)
5. [Step 3: Model Training](#step-3-model-training)
6. [Step 4: Model Testing](#step-4-model-testing)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Environment Setup
```bash
# Create conda environment
conda create -n gca python=3.10
conda activate gca

# Install required packages
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install transformers sentence-transformers
pip install gensim nltk spacy
pip install networkx matplotlib seaborn
pip install tensorboardX pyyaml tqdm
```

### Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('sentiwordnet')
```

### Install Mallet for LDA
```bash
# Download and install Mallet
wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz
tar -xzf mallet-2.0.8.tar.gz
mkdir -p packages
mv mallet-2.0.8 packages/
```

## Data Structure

Your dataset should follow this structure:

```
dataset/
├── earningscall/
│   ├── 0/          # Year 2018
│   ├── 1/          # Year 2017
│   ├── 2/          # Year 2016
│   ├── 3/          # Year 2016
│   ├── 4/          # Year 2015/2016
│   ├── 5/          # Year 2015
│   ├── 6/          # Year 2014/2015
│   ├── 7/          # Year 2014
│   └── updated_index_with_DVs.csv
└── processed/      # Will be created during preprocessing
```

### Transcript File Format
Each earnings call transcript CSV should have columns:
- `role`: Speaker role (analyst/executive)
- `name`: Speaker name
- `sentence`: Text content
- `section`: Section type (intro/qa)

### Label File Format
`updated_index_with_DVs.csv` should contain:
- `path`: Relative path to transcript (e.g., "4/3974165-albemarle-alb-q1-2016-results-earnings-call-transcript")
- `ticker`: Stock ticker
- `quarter`: Quarter (q1/q2/q3/q4)
- `year`: Year
- `firm_std_10_pre`: Pre-event volatility (10 days)
- `firm_std_20_pre`: Pre-event volatility (20 days)
- `firm_std_60_pre`: Pre-event volatility (60 days)
- `firm_std_10_post`: Post-event volatility (10 days)
- `firm_std_20_post`: Post-event volatility (20 days)
- `firm_std_60_post`: Post-event volatility (60 days)

## Step 1: Data Preprocessing

This step filters valid transcripts and prepares the processed dataset.

```bash
python preprocess.py \
    --dir dataset/earningscall \
    --out_dir dataset/processed \
    --out_filename processed_all.csv \
    --verbose info
```

**What this does:**
- Reads `dataset/earningscall/update_index_Volatility.csv`
- Validates each transcript has presentation (intro) and Q&A sections
- Filters out incomplete transcripts
- Creates `dataset/processed/processed_all.csv` with valid transcripts and labels

**Expected Output:**
```
Total valid transcripts found: X
Total transcripts after sorting: X
Saved processed data to dataset/processed/processed_all.csv
```

## Step 2: LDA Topic Modeling and Training Data Generation

### Step 2.1: Train LDA Model and Generate Training Data

**Note:** This step automatically creates the corpus file (`nlp_processed.csv`) if it doesn't exist, trains the LDA model, and generates training data files. On first run, all operations happen together.

Train the LDA topic model and generate training data:

```bash
python run_sent_lda.py \
    --in_path dataset/processed/processed_all.csv \
    --out_dir dataset/processed \
    --out_filename nlp_processed.csv \
    --need_train True \
    --need_topic_rep True \
    --doc LDA_0707
```

**What this does:**
- Generates training data for each year:
  - `dataset/processed/topic_sent_bert_based_data_2015.pkl`
  - `dataset/processed/topic_sent_bert_based_data_2016.pkl`
  - `dataset/processed/topic_sent_bert_based_data_2017.pkl`
  - `dataset/processed/topic_sent_bert_based_data_2018.pkl`

Each data file contains:
- `path`: Transcript path
- `label`: Volatility labels
- `pre_reps`: Presentation sentence representations
- `pre_topic_pro`: Presentation topic probabilities
- `q_reps`: Question representations
- `a_reps`: Answer representations
- `qa_topic_pro`: Q&A topic probabilities

## Step 3: Model Training

### Step 3.1: Prepare Configuration

Check and modify the configuration file `conf/topic_profet_gnn.yml`:

```yaml
train:
  batch_size: 64
  n_epochs: 12
  loss: "mse_loss"
  grad_clip: 0

model:
  embedding_size: 384
  hidden_size: [300, 200, 100]
  hidden_layers: 3
  dropout: 0.1
  in_bn: False
  hid_bn: False
  out_bn: True

optim:
  optimizer: 'Adam'
  lr: 0.0015
  weight_decay: 0.00001

scheduler:
  is_scheduler: false
  lr_dc: 0.1
  lr_dc_step: 10

data:
  train_path:
    - 'dataset/processed/topic_sent_bert_based_data_2015.pkl'
    - 'dataset/processed/topic_sent_bert_based_data_2016.pkl'
    - 'dataset/processed/topic_sent_bert_based_data_2017.pkl'
  test_path:
    - 'dataset/processed/topic_sent_bert_based_data_2018.pkl'
  label: 'firm_std_60_post'  # Options: firm_std_10_post, firm_std_20_post, firm_std_60_post
  batch_size: 16

checkpoint:
  choice: 'best'  # latest or best
```

### Step 3.2: Train the Model

```bash
python main.py \
    --config topic_profet_gnn.yml \
    --trainer GCATrainer \
    --verbose info \
    --comment "Training GCA model"
```

**What this does:**
- Loads preprocessed data
- Trains the Graph Conversational Analysis model
- Saves checkpoints to `result/checkpoint/GCATrainer/`
  - `best_checkpoint.pth`: Best model based on validation loss
  - `latest_checkpoint.pth`: Most recent model
- Logs training progress to `result/log/GCATrainer/stdout.txt`
- Saves TensorBoard logs to `result/tensorboard/GCATrainer/`

**Monitor Training:**
```bash
tensorboard --logdir result/tensorboard/GCATrainer/
```

**Expected Output:**
```
Epoch [1/12], Train Loss: X.XXXX, Val Loss: X.XXXX
Epoch [2/12], Train Loss: X.XXXX, Val Loss: X.XXXX
...
Best score is: X.XXXX, current score is: X.XXXX, save best_checkpoint.pth
Training completed. Total epochs: 12
```

## Step 4: Model Testing

### Step 4.1: Test on Hold-out Set

```bash
python main.py \
    --config topic_profet_gnn.yml \
    --trainer GCATrainer \
    --test True \
    --verbose info
```

**What this does:**
- Loads best checkpoint
- Evaluates on test set (2018 data)
- Computes metrics:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Spearman's correlation
  - Kendall's tau

**Expected Output:**
```
Loading model from: result/checkpoint/GCATrainer/best_checkpoint.pth
Test Results:
- MSE: X.XXXX
- MAE: X.XXXX
- Spearman's correlation: X.XXXX (p-value: X.XXXX)
- Kendall's tau: X.XXXX (p-value: X.XXXX)
```



## Directory Structure After Pipeline

```
GraphConversationalAnalysis/
├── dataset/
│   ├── earningscall/          # Original transcripts
│   └── processed/             # Preprocessed data
│       ├── processed_all.csv
│       ├── sent_lda_rep_v2_0707.pkl
│       └── topic_sent_bert_based_data_*.pkl
├── result/
│   ├── checkpoint/
│   │   ├── GCATrainer/        # Model checkpoints
│   │   └── LDA_0707/          # LDA models
│   ├── log/
│   │   └── GCATrainer/        # Training logs
│   ├── tensorboard/
│   │   └── GCATrainer/        # TensorBoard logs
│   └── output/                # Test results
├── packages/
│   └── mallet-2.0.8/          # Mallet installation
└── conf/
    └── topic_profet_gnn.yml   # Configuration file
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={Learning from Earnings Calls: Graph-Based Conversational Modeling for Financial Prediction},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

