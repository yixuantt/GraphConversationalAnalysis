"""
Data preprocessing script for earnings call transcripts.
1. Scans data directory for valid transcripts
2. Filters transcripts with complete presentation and Q&A sections
3. Saves processed data with volatility labels
"""
import argparse
import logging
import os

import pandas as pd
from tqdm import tqdm


def preprocess_data(args):
    logging.info(">" * 100)
    if not os.path.exists(os.path.join(args.out_dir, args.out_filename)):

        df = pd.read_csv(os.path.join(args.dir, 'update_index_Volatility.csv'),
                         usecols=['path',
                                  'year',
                                  'firm_std_3_pre',
                                  'firm_std_3_post',
                                  'firm_std_7_pre',
                                  'firm_std_7_post',
                                  'firm_std_10_pre',
                                  'firm_std_10_post',
                                  'firm_std_15_pre',
                                  'firm_std_15_post',
                                  'firm_std_20_pre',
                                  'firm_std_20_post',
                                  'firm_std_60_pre',
                                  'firm_std_60_post']).dropna()
        paths = df['path']
        years = df['year']
        firm_std_3_pre = df['firm_std_3_pre']
        firm_std_3_post = df['firm_std_3_post']
        firm_std_7_pre = df['firm_std_7_pre']
        firm_std_7_post = df['firm_std_7_post']
        firm_std_10_pre = df['firm_std_10_pre']
        firm_std_10_post = df['firm_std_10_post']
        firm_std_15_pre = df['firm_std_15_pre']
        firm_std_15_post = df['firm_std_15_post']
        firm_std_20_pre = df['firm_std_20_pre']
        firm_std_20_post = df['firm_std_20_post']
        firm_std_60_pre = df['firm_std_60_pre']
        firm_std_60_post = df['firm_std_60_post']

        new_paths = []
        new_years = []
        new_3_pre = []
        new_3_post = []
        new_7_pre = []
        new_7_post = []
        new_10_pre = []
        new_10_post = []
        new_15_pre = []
        new_15_post = []
        new_20_pre = []
        new_20_post = []
        new_60_pre = []
        new_60_post = []

        for path, year, pre_3, post_3, pre_7, post_7, pre_10, post_10, pre_15, post_15, pre_20, post_20, pre_60, post_60 in tqdm(zip(paths, years, firm_std_3_pre, firm_std_3_post, firm_std_7_pre, firm_std_7_post, firm_std_10_pre, firm_std_10_post, firm_std_15_pre, firm_std_15_post, firm_std_20_pre, firm_std_20_post, firm_std_60_pre, firm_std_60_post), total=len(paths)):
            cur_path = os.path.join(args.dir, path) + '.csv'
            if os.path.exists(cur_path):
                df = pd.read_csv(cur_path).dropna()
                present = df.loc[df['section'] == 'intro'].dropna(axis=0, subset=['sentence'])
                q = df.loc[(df['section'] == 'qa') & (df['role'] == 'analyst')].dropna(axis=0, subset=['sentence'])['sentence']
                a = df.loc[(df['section'] == 'qa') & (df['role'] != 'analyst')].dropna(axis=0, subset=['sentence'])['sentence']
                if len(present) != 0 and len(q) != 0 and len(a) != 0:
                    new_years.append(year)
                    new_paths.append(cur_path)
                    new_3_pre.append(pre_3)
                    new_3_post.append(post_3)
                    new_7_pre.append(pre_7)
                    new_7_post.append(post_7)
                    new_10_pre.append(pre_10)
                    new_10_post.append(post_10)
                    new_15_pre.append(pre_15)
                    new_15_post.append(post_15)
                    new_20_pre.append(pre_20)
                    new_20_post.append(post_20)
                    new_60_pre.append(pre_60)
                    new_60_post.append(post_60)
        zip_data = zip(new_years, new_paths, new_3_pre, new_3_post, new_7_pre, new_7_post, new_10_pre,
                       new_10_post, new_15_pre, new_15_post, new_20_pre, new_20_post, new_60_pre, new_60_post)
        logging.info(f"Total valid transcripts found: {len(new_years)}")
        sorted_data = sorted(zip_data)
        logging.info(f"Total transcripts after sorting: {len(sorted_data)}")
        processed_years = [data[0] for data in sorted_data]
        processed_paths = [data[1] for data in sorted_data]
        processed_firm_std_3_pre = [data[2] for data in sorted_data]
        processed_firm_std_3_post = [data[3] for data in sorted_data]
        processed_firm_std_7_pre = [data[4] for data in sorted_data]
        processed_firm_std_7_post = [data[5] for data in sorted_data]
        processed_firm_std_10_pre = [data[6] for data in sorted_data]
        processed_firm_std_10_post = [data[7] for data in sorted_data]
        processed_firm_std_15_pre = [data[8] for data in sorted_data]
        processed_firm_std_15_post = [data[9] for data in sorted_data]
        processed_firm_std_20_pre = [data[10] for data in sorted_data]
        processed_firm_std_20_post = [data[11] for data in sorted_data]
        processed_firm_std_60_pre = [data[12] for data in sorted_data]
        processed_firm_std_60_post = [data[13] for data in sorted_data]

        dataframe = pd.DataFrame(
            {
                'year': processed_years,
                'path': processed_paths,
                'firm_std_3_pre': processed_firm_std_3_pre,
                'firm_std_3_post': processed_firm_std_3_post,
                'firm_std_7_pre': processed_firm_std_7_pre,
                'firm_std_7_post': processed_firm_std_7_post,
                'firm_std_10_pre': processed_firm_std_10_pre,
                'firm_std_10_post': processed_firm_std_10_post,
                'firm_std_15_pre': processed_firm_std_15_pre,
                'firm_std_15_post': processed_firm_std_15_post,
                'firm_std_20_pre': processed_firm_std_20_pre,
                'firm_std_20_post': processed_firm_std_20_post,
                'firm_std_60_pre': processed_firm_std_60_pre,
                'firm_std_60_post': processed_firm_std_60_post,
            }
        )
        os.makedirs(args.out_dir, exist_ok=True)
        dataframe.to_csv(os.path.join(args.out_dir, args.out_filename), index=False, sep=',')
        logging.info(f"Saved processed data to {os.path.join(args.out_dir, args.out_filename)}")
    else:
        logging.info(f"Processed file already exists: {os.path.join(args.out_dir, args.out_filename)}")
    
    logging.info("<" * 100)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--dir', type=str, default='dataset/earningscall')
    parser.add_argument('--out_dir', type=str, default='dataset/processed')
    parser.add_argument('--out_filename', type=str, default='processed_all.csv')
    parser.add_argument('--comment', type=str, default='Pre-process', help='A string for experiment comment')
    parser.add_argument('--scan', default=True, help='Whether to scan data')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--result', type=str, default='result', help='Path for saving running related data.')

    args = parser.parse_args()
    args.log = os.path.join(args.result, 'log', 'pre_process')

    if not os.path.exists(args.log):
        os.makedirs(args.log)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    return args


def main():
    args = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Args = {}".format(args))

    preprocess_data(args)


if __name__ == '__main__':
    main()
