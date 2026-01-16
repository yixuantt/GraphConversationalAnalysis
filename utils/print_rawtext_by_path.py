import pandas as pd
from nltk import sent_tokenize


def print_raw_text(path):
    cur_df = pd.read_csv(path)
    cur_df.dropna(inplace=True)
    cur_df.reset_index(drop=True, inplace=True)

    # presentation
    pre = cur_df.loc[(cur_df['section'] == 'intro')]['sentence']
    merged_pre = " ".join(pre)
    pre_sentences = sent_tokenize(merged_pre)

    round_qr_sent = []  # save all round q
    round_ar_sent = []  # save all round a
    # question-answer drop the final question, always thanks. if index i+1 - index i == 1, drop it
    q_index = cur_df.loc[(cur_df['section'] == 'qa') & (cur_df['role'] == 'analyst')].index
    for i in range(len(q_index) - 1):
        if q_index[i + 1] - q_index[i] > 1:
            merged_q = " ".join(cur_df.loc[[q_index[i]]]['sentence'])
            round_q_sentences = sent_tokenize(merged_q)
            round_qr_sent.append(round_q_sentences)

            a_index = list(range(q_index[i] + 1, q_index[i + 1]))
            merged_a = " ".join(cur_df.loc[a_index]['sentence'])
            round_a_sentences = sent_tokenize(merged_a)
            round_ar_sent.append(round_a_sentences)
    return pre_sentences, round_qr_sent, round_ar_sent
