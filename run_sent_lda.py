import argparse
import logging
import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
from pandarallel import pandarallel
from joblib import Parallel, delayed
import time
import pickle
import gc
from copy import deepcopy
import threading
import queue

# logging.getLogger('gensim').setLevel(logging.WARNING)

# NLP stuff
import contractions
import demoji
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sentence_transformers import SentenceTransformer

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS

# plot
import matplotlib.pyplot as plt

with open('dataset/stopwords.txt', 'r', encoding='utf-8') as file:
    custom_stopwords = set([line.strip() for line in file])

from nltk.corpus import sentiwordnet as swn
# Do this first, that'll do something eval()
# to "materialize" the LazyCorpusLoader
next(swn.all_senti_synsets())
wnl = WordNetLemmatizer()


def preprocess_text(text):
    """This function preprocesses a single text string."""
    # Initialize pandarallel (optional if you don't use it here)
    # pandarallel.initialize(nb_workers=os.cpu_count())

    # convert to lowercase
    text = ' '.join([w.lower() for w in text.split()])

    # remove emojis
    text = demoji.replace(text, "")

    # expand contractions
    text = ' '.join([contractions.fix(word) for word in text.split()])

    # remove punctuation
    text = ''.join([i for i in text if i not in string.punctuation])

    # remove numbers
    text = ' '.join(re.sub("[^a-zA-Z]+", " ", text).split())

    # remove stopwords
    stopwords_set = set(stopwords.words('english')) - {'not', 'no'}
    text = ' '.join([w for w in text.split() if w not in stopwords_set])

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])

    # remove short words
    text = ' '.join([w.strip() for w in text.split() if len(w.strip()) >= 3])

    return text

def preprocess(text_col):
    """This function will apply NLP preprocessing lambda functions over a pandas series such as df['text'].
       These functions include converting text to lowercase, removing emojis, expanding contractions, removing punctuation,
       removing numbers, removing stopwords, lemmatization, etc."""

    # convert to lowercase
    text_col = text_col.apply(lambda x: ' '.join([w.lower() for w in x.split()]))

    # remove emojis
    text_col = text_col.apply(lambda x: demoji.replace(x, ""))

    # expand contractions
    text_col = text_col.apply(lambda x: ' '.join([contractions.fix(word) for word in x.split()]))

    # remove punctuation
    text_col = text_col.apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))

    # remove numbers
    text_col = text_col.apply(lambda x: ' '.join(re.sub("[^a-zA-Z]+", " ", x).split()))

    # remove stopwords
    # stopwords = [sw for sw in nltk.corpus.stopwords.words('english') if sw not in ['not', 'no']]
    text_col = text_col.apply(lambda x: ' '.join([w for w in x.split() if w not in custom_stopwords]))

    # lemmatization
    text_col = text_col.apply(lambda x: ' '.join([wnl.lemmatize(w, pos='n') for w in x.split()]))
    text_col = text_col.apply(lambda x: ' '.join([wnl.lemmatize(w, pos='v') for w in x.split()]))
    text_col = text_col.apply(lambda x: ' '.join([wnl.lemmatize(w, pos='r') for w in x.split()]))
    text_col = text_col.apply(lambda x: ' '.join([wnl.lemmatize(w, pos='s') for w in x.split()]))
    text_col = text_col.apply(lambda x: ' '.join([wnl.lemmatize(w, pos='a') for w in x.split()]))

    # remove short words
    text_col = text_col.apply(lambda x: ' '.join([w.strip() for w in x.split() if len(w.strip()) >= 3]))

    return text_col


def apply_parallel(df_grouped, func):
    results = Parallel(n_jobs=-1)(delayed(func)(group) for name, group in df_grouped)
    return pd.concat(results)


def parallel_preprocess(text_col):
    """This function apply parallel for preprocess."""
    df_grouped = text_col.groupby(text_col.index)
    text_col = apply_parallel(df_grouped, preprocess)
    return text_col


def parallel_preprocess_pandarallel(text_col):
    """This function use pandarallel to accelerate computing"""
    pandarallel.initialize(nb_workers=os.cpu_count())

    # convert to lowercase
    text_col = text_col.parallel_apply(lambda x: ' '.join([w.lower() for w in x.split()]))

    # remove emojis
    text_col = text_col.parallel_apply(lambda x: demoji.replace(x, ""))

    # expand contractions
    text_col = text_col.parallel_apply(lambda x: ' '.join([contractions.fix(word) for word in x.split()]))

    # remove punctuation
    text_col = text_col.parallel_apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))

    # remove numbers
    text_col = text_col.parallel_apply(lambda x: ' '.join(re.sub("[^a-zA-Z]+", " ", x).split()))

    # remove stopwords
    # stopwords = [sw for sw in nltk.corpus.stopwords.words('english') if sw not in ['not', 'no']]
    text_col = text_col.apply(lambda x: ' '.join([w for w in x.split() if w not in stopwords]))

    # lemmatization
    text_col = text_col.apply(lambda x: ' '.join([WordNetLemmatizer().lemmatize(w) for w in x.split()]))

    # remove short words
    text_col = text_col.apply(lambda x: ' '.join([w.strip() for w in x.split() if len(w.strip()) >= 3]))

    return text_col


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3, out_model_path='.'):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------  
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    # Create output directory if it doesn't exist
    os.makedirs(out_model_path, exist_ok=True)
    mallet_path = 'packages/mallet-2.0.8/bin/mallet'
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path,
                                                 corpus=corpus,
                                                 num_topics=num_topics,
                                                 id2word=dictionary,
                                                 random_seed=1234,
                                                #  alpha=50,
                                                 workers=os.cpu_count())

        model.save(os.path.join(out_model_path, f"sent_lda_topic_{num_topics}.model"))
        lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model)
        # model_list.append(model)
        coherencemodel = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        score = coherencemodel.get_coherence()
        coherence_values.append(score)
        print(f'topic = {num_topics}, coherence score: {score}')

    return model_list, coherence_values


def multi_thread_process_list(input_list, collate_fn, num_worker=os.cpu_count(), keep_order=False):
    if keep_order:
        input_list = list(enumerate(input_list))

    def chunk_list(lst, n):
        size = len(lst)
        chunk_size, remainder = divmod(size, n)
        chunks = [lst[i*chunk_size:(i+1)*chunk_size] for i in range(n)]
        for i in range(remainder):
            chunks[i].append(lst[size-remainder+i])
        return chunks

    """Does not guarantee original element order"""
    input_chunks = chunk_list(input_list, num_worker)
    result_queue = queue.Queue()
    threads = []

    progress_bars = [tqdm(total=len(chunk), position=i) for i, chunk in zip(range(num_worker), input_chunks)]

    def worker(chunk, result_queue, progress_bar):
        worker_result = []
        if keep_order:
            for i, item in chunk:
                worker_result.append((i, collate_fn(item)))
                progress_bar.update(1)
        else:
            for item in chunk:
                worker_result.append(collate_fn(item))
                progress_bar.update(1)
        result_queue.put(worker_result)

    for i, chunk in enumerate(input_chunks):
        thread = threading.Thread(target=worker, args=(chunk, result_queue, progress_bars[i]))
        thread.start()
        threads.append(thread)

    new_res = []
    for thread in threads:
        thread.join()
        while not result_queue.empty():
            result = result_queue.get()
            new_res.extend(result)

    if keep_order:
        new_res.sort(key=lambda x: x[0])
        new_res = [element for index, element in new_res]
    return new_res


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--in_path', type=str, default='dataset/processed/processed.csv')
    parser.add_argument('--out_dir', type=str, default='dataset/processed')
    parser.add_argument('--out_filename', type=str, default='nlp_processed_sent_lda.csv', help='save presentation, qa part')
    parser.add_argument('--comment',
                        type=str,
                        default='Preprocess for LDA and Run LDA topic model',
                        help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--result', type=str, default='result', help='Path for saving running related data.')
    parser.add_argument('--search_model', default=False, help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='LDA_0707', help='A string for documentation purpose')
    parser.add_argument('--need_train', default=False, help='Train model or load pretrained model')
    parser.add_argument('--need_topic_rep', default=True, help='save LDA topic representation data')

    args = parser.parse_args()
    args.log = os.path.join(args.result, 'log', 'lda_topic')
    args.checkpoint = os.path.join(args.result, 'checkpoint', args.doc)

    if not os.path.exists(args.log):
        os.makedirs(args.log)

    # setup logger
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


def get_topic_rep(lda_model, out_path, encoder):
    rep_dict = {}
    for topic in lda_model.show_topics(num_topics=-1, num_words=10, formatted=False):
        topic_words = []
        word_reps = []
        word_weights = []
        topic_id = topic[0]
        for pair in topic[1]:
            topic_words.append(pair[1])
            word_weights.append(pair[1])
            word_reps.append(encoder.encode(pair[0], show_progress_bar=False))
        word_weights = np.expand_dims(np.array([weight / np.sum(word_weights) for weight in word_weights]), axis=1)
        topic_rep = np.sum(word_weights * word_reps, axis=0)
        rep_dict[topic_id] = topic_rep
    print(rep_dict.keys())
    # Create output directory if it doesn't exist
    out_dir = os.path.dirname(out_path)
    if out_dir:  # Only create directory if path contains a directory component
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'wb') as fOut:
        pickle.dump(rep_dict, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"{out_path} has been saved.")


def main():
    args = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Args = {}".format(args))

    '''load processed.csv, then generate new one: nlp_processed.csv'''
    if not os.path.exists(os.path.join(args.out_dir, args.out_filename)):
        logging.info(f'{os.path.join(args.out_dir, args.out_filename)} not exist, creating...')
        df = pd.read_csv(args.in_path)
        df = df[df['year'].isin([2015, 2016, 2017, 2018])]
        paths = df['path'].to_list()

        def collate_func(path):
            cur_df = pd.read_csv(path).dropna()
            # presentation
            pre = cur_df.loc[(cur_df['section'] == 'intro')].dropna(axis=0, subset=['sentence'])['sentence']
            merged_pre = sent_tokenize(' '.join(preprocess(pre)))
            merged_pre_filtered = [sent for sent in merged_pre if len(word_tokenize(sent)) > 15]

            # question-answer
            # q_a = cur_df.loc[(cur_df['section'] == 'qa')].dropna(axis=0, subset=['sentence'])['sentence']
            # merged_qa = ' '.join(preprocess(q_a))
            # new_qa_per_text.append(merged_qa)

            return merged_pre_filtered

        new_sent_copora = []
        processed_res = multi_thread_process_list(paths, collate_fn=collate_func, num_worker=1)
        for doc in processed_res:
            new_sent_copora.extend(doc)

        logging.info(f'Total {len(new_sent_copora)} sentences.')
        # Create output directory if it doesn't exist
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, args.out_filename), 'wb') as fOut:
            pickle.dump(new_sent_copora, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    '''train model and save model file'''
    if args.need_train:
        with open(os.path.join(args.out_dir, args.out_filename), 'rb') as fIn:
            sentences = pickle.load(fIn)

        out_model_path = os.path.join(args.checkpoint, "sent_lda.model")
        out_dict_path = os.path.join(args.checkpoint, "sent_lda_id2word.pkl")

        def drop_punctuation(sentence):
            x = ''.join([i for i in sentence if i not in string.punctuation])
            return re.sub("[^a-zA-Z]+", " ", x).split()

        processed_data = multi_thread_process_list(sentences, collate_fn=drop_punctuation, num_worker=15)

        # processed_data = [text.split() for text in new_sentences]

        if not os.path.exists(out_dict_path):
            print('generating the dictionary...')
            # Create checkpoint directory if it doesn't exist
            os.makedirs(args.checkpoint, exist_ok=True)
            id2word = corpora.Dictionary(processed_data)
            id2word.filter_extremes(no_below=15, no_above=0.3, keep_n=50000)
            with open(out_dict_path, 'wb') as fOut:
                pickle.dump(id2word, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('loading the dictionary...')
            with open(os.path.join(args.checkpoint, "sent_lda_id2word.pkl"), 'rb') as fIn:
                id2word = pickle.load(fIn)

        logging.info("doc2bow...")
        corpus = [id2word.doc2bow(text) for text in tqdm(processed_data)]

        logging.info("Loading LDA model...")
        logging.info(f"Corpus length: {len(corpus)}")
        if args.search_model:
            logging.info("Finding best LDA model...")
            limit = 70
            start = 20
            step = 5
            model_list, coherence_values = compute_coherence_values(dictionary=id2word,
                                                                    corpus=corpus,
                                                                    texts=processed_data,
                                                                    start=start,
                                                                    limit=limit,
                                                                    step=step,
                                                                    out_model_path=args.checkpoint)

            logging.info(f"LDA model training finished. Coherence_values: {coherence_values}")

            # limit = 61
            # start = 36
            # step = 4
            x = range(start, limit, step)
            plt.plot(x, coherence_values)
            plt.xlabel("Num Topics")
            plt.ylabel("Coherence score")
            plt.legend(("coherence_values"), loc='best')
            # Create figure directory if it doesn't exist
            os.makedirs("result/figure", exist_ok=True)
            plt.savefig("result/figure/sent_lda_coherence_0707.png", dpi=500)
            # plt.show()

            for m, cv in zip(x, coherence_values):
                print(f"Num Topics = {m} has Coherence Value of {round(cv, 4)}")
        else:
            mallet_path = 'packages/mallet-2.0.8/bin/mallet'
            model = gensim.models.wrappers.LdaMallet(mallet_path,
                                                     corpus=corpus,
                                                     num_topics=50,
                                                     id2word=id2word,
                                                     random_seed=1234,
                                                     alpha=5,
                                                     workers=os.cpu_count()-1)

            # Create checkpoint directory if it doesn't exist
            os.makedirs(args.checkpoint, exist_ok=True)
            model.save(out_model_path)

    '''get topic model representation'''
    model = gensim.models.wrappers.LdaMallet.load(os.path.join(args.checkpoint, "sent_lda.model"))
    lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model)
    
    top_k = 10
    # beta = lda_model.get_topics()
    # print(beta.shape)  # [50, 30952]
    
    # with open(os.path.join(args.checkpoint, "sent_lda_id2word.pkl"), 'rb') as fIn:
    #             id2word = pickle.load(fIn)
    
    # def print_topic_to_list_with_rerank(beta, vocab, top_k=20):
    #     topic_id = 0
    #     topwords = []
    #     word_sum_prob = np.sum(beta, axis=0)
    #     normalized_beta = beta / word_sum_prob
    #     print(beta.shape)
    #     for t, topic_word in enumerate(normalized_beta):
    #         term_idx = np.argsort(topic_word)
    #         topKwords = []
    #         for j in np.flip(term_idx[-top_k:]):
    #             topKwords.append(vocab[j])
    #         print('topic', topic_id,':', ' '.join(topKwords))
    #         topic_id += 1
    #         topwords.append(topKwords)
    #     return topwords
    
    # topwords = print_topic_to_list_with_rerank(beta, id2word)
    for idx, topic in lda_model.print_topics(num_topics=-1, num_words=top_k):
        print(f"Topic {idx + 1}: {topic}")
    
    
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    get_topic_rep(lda_model, os.path.join(args.out_dir, "sent_lda_rep_v2_0707.pkl"), encoder)

    # exit()

    '''for each earning call, find presentation, question-answer pair'''
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    # model_pre = gensim.models.wrappers.LdaMallet.load(os.path.join(args.checkpoint, "lda_pre.model"))
    # model_qa = gensim.models.wrappers.LdaMallet.load(os.path.join(args.checkpoint, "lda_qa.model"))
    # model_all = gensim.models.wrappers.LdaMallet.load(os.path.join(args.checkpoint, "lda_all.model"))
    # lda_model_pre = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model_pre)
    # lda_model_qa = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model_qa)
    # lda_model_all = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model_all)
    # with open(os.path.join(args.checkpoint, "lda_pre_id2word.pkl"), 'rb') as fIn:
    #     id2word_pre = pickle.load(fIn)
    # with open(os.path.join(args.checkpoint, "lda_qa_id2word.pkl"), 'rb') as fIn:
    #     id2word_qa = pickle.load(fIn)
    # with open(os.path.join(args.checkpoint, "lda_all_id2word.pkl"), 'rb') as fIn:
    #     id2word_all = pickle.load(fIn)
    with open(os.path.join(args.checkpoint, "sent_lda_id2word.pkl"), 'rb') as fIn:
        id2word_all = pickle.load(fIn)

    ori_df = pd.read_csv(args.in_path)
    filtered_df = deepcopy(ori_df[ori_df['year'].isin([2015, 2016, 2017, 2018])])
    del ori_df

    for year in [2015, 2016, 2017, 2018]:
        df = filtered_df[filtered_df['year'] == year]
        # df = df[300:310]
        all_data = []
        for path, post_10, post_20, post_60 in tqdm(zip(df['path'],
                                                        df['firm_std_10_post'],
                                                        df['firm_std_20_post'],
                                                        df['firm_std_60_post'], ),
                                                    total=len(df['firm_std_10_post'])):
            cur_data = {}
            cur_data['path'] = path
            cur_data['label'] = {
                'firm_std_10_post': post_10,
                'firm_std_20_post': post_20,
                'firm_std_60_post': post_60
            }

            cur_df = pd.read_csv(path)
            cur_df.dropna(inplace=True)
            cur_df.reset_index(drop=True, inplace=True)

            # presentation
            pre = cur_df.loc[(cur_df['section'] == 'intro')]['sentence']
            
            merged_pre = " ".join(pre)
            pre_sentences = sent_tokenize(merged_pre)
            pre_reps = encoder.encode(pre_sentences, show_progress_bar=False)
            
            # Experiment 0721: Replace sentenceBERT with LDA topic vector (commented out)
            # pre_rep_list = []
            # for pre_sentence in pre_sentences:
            #     preprocessed_pre = ' '.join(preprocess_text(pre_sentence)).split()
            #     corpus = id2word_all.doc2bow(preprocessed_pre)
            #     pre_topic_pro = lda_model.get_document_topics(corpus)
            #     topic_vecotr = [0.0] * 50
            #     for index, value in pre_topic_pro:
            #         topic_vecotr[index] = value
            #     pre_rep_list.append(topic_vecotr)
            # new_pre_reps = np.stack(pre_rep_list, axis=0)
            
            cur_data['pre_reps'] = pre_reps  # Use sentence-BERT embeddings
            # cur_data['pre_reps'] = new_pre_reps  # LDA topic vector (commented out)

            preprocessed_pre = ' '.join(preprocess(pre)).split()
            corpus = id2word_all.doc2bow(preprocessed_pre)
            # pre_topic_pro = lda_model_all.get_document_topics(corpus)
            pre_topic_pro = lda_model.get_document_topics(corpus)
            cur_data['pre_topic_pro'] = pre_topic_pro

            round_qr = []  # save all round q representations
            round_ar = []  # save all round a representations
            round_qt = []  # save all round q topic probabilities
            round_at = []  # save all round a topic probabilities
            round_qat = []
            # question-answer drop the final question, always thanks. if index i+1 - index i == 1, drop it
            q_index = cur_df.loc[(cur_df['section'] == 'qa') & (cur_df['role'] == 'analyst')].index
            for i in range(len(q_index) - 1):
                if q_index[i + 1] - q_index[i] > 1:  ## > 1 represents containing "a"
                    merged_q = " ".join(cur_df.loc[[q_index[i]]]['sentence'])
                    round_q_sentences = sent_tokenize(merged_q)
                    round_q_reps = encoder.encode(round_q_sentences, show_progress_bar=False)
                    
                    # Experiment 0721: Replace sentenceBERT with LDA topic vector (commented out)
                    # q_rep_list = []
                    # for round_q_sentence in round_q_sentences:
                    #     preprocessed_q = ' '.join(preprocess_text(round_q_sentence)).split()
                    #     corpus = id2word_all.doc2bow(preprocessed_q)
                    #     q_topic_pro = lda_model.get_document_topics(corpus)
                    #     topic_vecotr = [0.0] * 50
                    #     for index, value in q_topic_pro:
                    #         topic_vecotr[index] = value
                    #     q_rep_list.append(topic_vecotr)
                    # new_round_q_reps = np.stack(q_rep_list, axis=0)

                    round_qr.append(round_q_reps)  # Use sentence-BERT embeddings
                    # round_qr.append(new_round_q_reps)  # LDA topic vector (commented out)

                    preprocessed_q = ' '.join(preprocess(cur_df.loc[[q_index[i]]]['sentence'])).split()
                    # corpus = id2word_all.doc2bow(preprocessed_q)
                    # q_topic_pro = lda_model_all.get_document_topics(corpus)
                    # round_qt.append(q_topic_pro)

                    a_index = list(range(q_index[i] + 1, q_index[i + 1]))
                    merged_a = " ".join(cur_df.loc[a_index]['sentence'])
                    round_a_sentences = sent_tokenize(merged_a)
                    round_a_reps = encoder.encode(round_a_sentences, show_progress_bar=False)
                    
                    # Experiment 0721: Replace sentenceBERT with LDA topic vector (commented out)
                    # a_rep_list = []
                    # for round_a_sentence in round_a_sentences:
                    #     preprocessed_a = ' '.join(preprocess_text(round_a_sentence)).split()
                    #     corpus = id2word_all.doc2bow(preprocessed_a)
                    #     a_topic_pro = lda_model.get_document_topics(corpus)
                    #     topic_vecotr = [0.0] * 50
                    #     for index, value in a_topic_pro:
                    #         topic_vecotr[index] = value
                    #     a_rep_list.append(topic_vecotr)
                    # new_round_a_reps = np.stack(a_rep_list, axis=0)
                    
                    round_ar.append(round_a_reps)  # Use sentence-BERT embeddings
                    # round_ar.append(new_round_a_reps)  # LDA topic vector (commented out)

                    preprocessed_a = ' '.join(preprocess(cur_df.loc[a_index]['sentence'])).split()
                    # corpus = id2word_all.doc2bow(preprocessed_a)
                    # a_topic_pro = lda_model_all.get_document_topics(corpus)
                    # round_at.append(a_topic_pro)
                    corpus = id2word_all.doc2bow(preprocessed_q + preprocessed_a)
                    # qa_topic_pro = lda_model_all.get_document_topics(corpus)
                    qa_topic_pro = lda_model.get_document_topics(corpus)
                    round_qat.append(qa_topic_pro)

            cur_data['q_reps'] = round_qr
            cur_data['a_reps'] = round_ar
            # cur_data['q_topic_pro'] = round_qt
            # cur_data['a_topic_pro'] = round_at
            cur_data['qa_topic_pro'] = round_qat
            # print(cur_data.keys())
            # for key, item in cur_data.items():
            #     print(f"{key}: {len(item)}")
            # exit()
            all_data.append(cur_data)

        # Create processed directory if it doesn't exist
        os.makedirs("dataset/processed", exist_ok=True)
        with open(f"dataset/processed/topic_sent_bert_based_data_{str(year)}.pkl", 'wb') as fOut:
            pickle.dump(all_data, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"dataset/processed/topic_sent_bert_based_data_{str(year)}.pkl has been saved.")
        gc.collect()
        

if __name__ == '__main__':
    main()
