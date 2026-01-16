# Quick Start Guide

This guide helps you quickly run the GCA model with the sample dataset.

## Prerequisites

```bash
# Activate conda environment
conda activate gca

# Ensure you're in the project root
cd /path/to/GraphConversationalAnalysis
```

## Quick Run (All Steps)

### Step 1: Preprocess Data (5 minutes)
```bash
python preprocess.py \
    --dir dataset/earningscall \
    --out_dir dataset/processed \
    --out_filename processed_all.csv
```

### Step 2: Train LDA Model and Generate Training Data (30-60 minutes)
```bash
# This step automatically creates the corpus if needed, trains LDA model, and generates training data
python run_sent_lda.py \
    --in_path dataset/processed/processed_all.csv \
    --out_dir dataset/processed \
    --out_filename nlp_processed.csv \
    --need_train True \
    --need_topic_rep True \
    --doc LDA_0707
```

### Step 3: Train Model (2-4 hours depending on GPU)
```bash
python main.py \
    --config topic_profet_gnn.yml \
    --trainer GCATrainer \
    --comment "Training on sample data"
```

### Step 4: Test Model (5 minutes)
```bash
python main.py \
    --config topic_profet_gnn.yml \
    --trainer GCATrainer \
    --test True
```

## Expected Timeline

| Step | Time | Output |
|------|------|--------|
| Preprocessing | ~5 min | `dataset/processed/processed_all.csv` |
| LDA Training & Data Generation | ~30-60 min | `result/checkpoint/LDA_0707/sent_lda.model`<br>`dataset/processed/topic_sent_bert_based_data_*.pkl` |
| Model Training | ~2-4 hours | `result/checkpoint/GCATrainer/best_checkpoint.pth` |
| Model Testing | ~5 min | Test metrics |

## Verify Your Results

After each step, verify the outputs:

```bash
# After preprocessing
ls dataset/processed/processed_all.csv

# After LDA training
ls result/checkpoint/LDA_0707/sent_lda.model
ls result/checkpoint/LDA_0707/sent_lda_id2word.pkl
ls dataset/processed/sent_lda_rep_v2_0707.pkl

# After topic representation
ls dataset/processed/topic_sent_bert_based_data_*.pkl

# After model training
ls result/checkpoint/GCATrainer/best_checkpoint.pth

# Check training logs
cat result/log/GCATrainer/stdout.txt
```

## Monitoring Training

```bash
# In a separate terminal
tensorboard --logdir result/tensorboard/GCATrainer/

# Open browser to: http://localhost:6006
```

## Common Issues

1. **Memory Error:** Reduce batch size in `conf/topic_profet_gnn.yml`
2. **CUDA Error:** Check GPU availability with `nvidia-smi`
3. **Import Error:** Run `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`
4. **Mallet Error:** Check `packages/mallet-2.0.8/bin/mallet` exists

## Next Steps

After successful training:
- Review test metrics in logs
- Run interpretability analysis: `python main.py --test_interpretability True`
- Try different labels: modify `data.label` in config file
- Experiment with hyperparameters

For detailed documentation, see [pipeline.md](pipeline.md).

