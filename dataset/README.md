# Dataset Directory

This directory contains sample earnings call transcripts for training and testing the GCA model.

## Structure

```
dataset/
├── earningscall/
│   ├── 0/          # Year 2018 transcripts
│   ├── 1/          # Year 2017 transcripts
│   ├── 2/          # Year 2016 transcripts
│   ├── 3/          # Year 2016 transcripts
│   ├── 4/          # Year 2015/2016 transcripts
│   ├── 5/          # Year 2015 transcripts
│   ├── 6/          # Year 2014/2015 transcripts
│   ├── 7/          # Year 2014 transcripts
│   └── updated_index_with_DVs.csv  # Labels and metadata
├── processed/      # Generated during preprocessing
│   ├── processed_all.csv
│   ├── nlp_processed.csv
│   ├── sent_lda_rep_v2_0707.pkl
│   └── topic_sent_bert_based_data_*.pkl
└── README.md
```

## Data Description

### Earnings Call Transcripts

Each subdirectory (0-7) contains earnings call transcript CSV files with the following format:

- **Filename**: `{id}-{company}-{ticker}-{quarter}-{year}-results-earnings-call-transcript.csv`
- **Columns**:
  - `role`: Speaker role (empty for operator, analyst name, or executive name)
  - `name`: Speaker name
  - `sentence`: Text content of the speech
  - `section`: Section type (`intro` for presentation, `qa` for Q&A)

### Label File: updated_index_with_DVs.csv

Contains financial labels and metadata for each transcript:

**Key Columns:**
- `path`: Relative path to transcript file (e.g., "4/3974165-albemarle-alb-q1-2016...")
- `ticker`: Stock ticker symbol (e.g., ALB, MELI, BSAC)
- `quarter`: Fiscal quarter (q1, q2, q3, q4)
- `year`: Year
- `gvkey`: Compustat identifier
- `rdq`: Report date
- `conm`: Company name

**Pre-event Volatility Metrics:**
- `firm_std_10_pre`: Standard deviation of returns 10 days before event
- `firm_std_20_pre`: Standard deviation of returns 20 days before event
- `firm_std_60_pre`: Standard deviation of returns 60 days before event

**Post-event Volatility Metrics (Prediction Targets):**
- `firm_std_10_post`: Standard deviation of returns 10 days after event
- `firm_std_20_post`: Standard deviation of returns 20 days after event
- `firm_std_60_post`: Standard deviation of returns 60 days after event

**Other Financial Metrics:**
- `CAR_XX_YY`: Cumulative Abnormal Returns
- `PEratio_*`: Price-to-Earnings ratios
- `EarningYield_*`: Earnings yield metrics
- `BMratio`: Book-to-Market ratio


## Preprocessing

Before using the data, run the preprocessing script:

```bash
python preprocess.py --dir dataset/earningscall --out_dir dataset/processed
```

This will:
1. Validate each transcript has both presentation and Q&A sections
2. Filter out incomplete transcripts
3. Create `processed/processed_all.csv` with valid data

## Data Privacy and Usage

This is sample data for demonstration purposes. The earnings call transcripts are publicly available information from company investor relations. 

When using this dataset:
- Ensure compliance with data usage terms
- Cite the original data sources appropriately
- Use for research and educational purposes

## References

For more information on the complete data processing pipeline, see:
- [Quick Start Guide](../doc/quick_start.md)
- [Complete Pipeline](../doc/pipeline.md)