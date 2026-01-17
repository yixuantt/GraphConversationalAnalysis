import gc
import logging
from copy import deepcopy

import os
import math
import queue
import threading
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import gensim
from tqdm import tqdm
from model import *
import torch.optim as optim
import shutil
import tensorboardX
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from torch.nn.utils.rnn import pad_sequence
from sklearn.cluster import MiniBatchKMeans
import scipy.stats as stats

from model.MRQASelector import MRQA_simple

# NLP imports
import re
import contractions
import demoji
import string
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

"""
Three sections: presentation, question, answers
Considering the consistency of all benchmarks, we use the sentence transformer as a substitute for Bi-lstm+attention

Todo List:
-[ ] check whether the dataset has been generated, if not, generate it.
"""
prefix_len = len('dataset/processed/')
label_source_df = pd.read_csv(
    'dataset/processed/update_index_Volatility.csv')

with open('dataset/stopwords.txt', 'r', encoding='utf-8') as file:
            custom_stopwords = set([line.strip() for line in file])


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

def read_dict(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            words = [line.strip().lower() for line in file]
        return words
    
def count_emotional_words(text_words, positive_words, negative_words):
    positive_count = sum(1 for word in text_words if word in positive_words)
    negative_count = sum(1 for word in text_words if word in negative_words)
    return positive_count, negative_count

def preprocess_dataset(path, label_name):
    '''
        Actually for topic_based method, we need 3 input, pre sentence, q-a pair
        For the first simplest version, we need pre sentences(N), pre-topic(28), and q-a pair(k)
        :param paths:
        :param label_name:
        :return:
        '''
    checkpoint_path = './result/checkpoint/MRQA/best_checkpoint.pth'
    selector = MRQA_simple().cuda()
    pretrained_data = torch.load(checkpoint_path)
    selector.load_state_dict(pretrained_data)

    X_pre = []
    X_q = []
    X_a = []
    Y = []

    with open(path, 'rb') as fIn:
        stored_datas = pickle.load(fIn)
        for stored_data in tqdm(stored_datas):
            if (len(stored_data['pre_reps']) != 0) and (len(stored_data['q_reps']) != 0) and (len(stored_data['a_reps']) != 0):
                if label_name in ['firm_std_3_post', 'firm_std_7_post', 'firm_std_15_post']:
                    full_path = stored_data['path']
                    basename = os.path.basename(full_path)  
                    basename_no_ext = os.path.splitext(basename)[0]

                    matches = label_source_df[label_source_df['path'].str.contains(basename_no_ext)]

                    if len(matches) == 0:
                        print('N/A')
                        continue
                    else:
                        cur_val = matches[label_name]
                        if len(cur_val.values) == 0 or cur_val.values[0] == 0.:
                            print('N/A')
                            continue
                        else:
                            Y.append(cur_val.values[0])
                else:
                    Y.append(stored_data['label'][label_name])

                q_rounds = []
                a_rounds = []
                for round_q, round_a in zip(stored_data['q_reps'], stored_data['a_reps']):
                    q = torch.from_numpy(round_q).float().cuda()
                    a = torch.from_numpy(round_a).float().cuda()

                    q_select, a_select = selector(
                        q.unsqueeze(0), a.unsqueeze(0))
                    q_selected_sentences = q[q_select.bool()]
                    a_selected_sentences = a[a_select.bool()]
                    if q_selected_sentences.shape[0] == 0 or a_selected_sentences.shape[0] == 0:
                        pass
                    else:
                        q_selected_sentences = q_selected_sentences.cpu().numpy()
                        a_selected_sentences = a_selected_sentences.cpu().numpy()
                        q_rounds.append(
                            np.mean(q_selected_sentences, axis=0).squeeze())
                        a_rounds.append(
                            np.mean(a_selected_sentences, axis=0).squeeze())

                if len(q_rounds) == 0 or len(a_rounds) == 0:
                    Y.pop()
                else:
                    X_pre.append(stored_data['pre_reps'])
                    X_q.append(np.stack(q_rounds, axis=0))
                    X_a.append(np.stack(a_rounds, axis=0))
    return {
        'pre': X_pre,
        'q': X_q,
        'a': X_a,
        'label': Y
    }


class TopicProfetGNNDataset(Dataset):
    def __init__(self, paths, label_name='firm_std_10_post', clustering=False, lock_target=False):
        super(TopicProfetGNNDataset, self).__init__()
        self.label_name = label_name
        self.clustering_samples = []
        self.clustering = clustering
        self.clustering_path = "dataset/processed/clustering_res.pkl"
        self.lock_target = lock_target
        data = self.loading_train_dataset(paths, label_name)

        self.input_path = data['path']
        self.input_pre = data['pre']
        self.input_q = data['q']
        self.input_a = data['a']
        self.label = np.log(data['label'])
        self.input_pre_topic = data['pre_topic']
        self.input_qa_topic = data['qa_topic']
        self.clusters_centriod = None
        if self.clustering and not os.path.exists(self.clustering_path):
            self.clustering_samples = np.concatenate(
                self.clustering_samples, axis=0)
            self.clustering_alg()
        else:
            print('skip clustering.')
            self.clustering = False

        print("Label stats BEFORE log:", pd.Series(data['label']).describe())
        print("Label <= 0:", (np.array(data['label']) <= 0).sum())


    def clustering_alg(self):
        
        kmeans = MiniBatchKMeans(n_clusters=200, batch_size=50, verbose=1)
        kmeans.fit(self.clustering_samples)
        with open(self.clustering_path, 'wb') as fOut:
            pickle.dump(kmeans.cluster_centers_, fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print("Kmeans Clustering finished.")

    def multi_thread_load_dataset(self, paths, label_name):
        X_pre = []
        X_q = []
        X_a = []
        Y = []
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(preprocess_dataset, paths[0], label_name)
            future2 = executor.submit(preprocess_dataset, paths[1], label_name)

            result1 = future1.result()
            result2 = future2.result()

            X_pre.extend(result1['pre'])
            X_q.extend(result1['q'])
            X_a.extend(result1['a'])
            Y.extend(result1['label'])

            X_pre.extend(result2['pre'])
            X_q.extend(result2['q'])
            X_a.extend(result2['a'])
            Y.extend(result2['label'])

        saved_data = {
            'pre': X_pre,
            'q': X_q,
            'a': X_a,
            'label': Y
        }

        with open(f"dataset/processed/mr_qa_train_data.pkl", 'wb') as fOut:
            pickle.dump(saved_data, fOut, protocol=pickle.HIGHEST_PROTOCOL)

        return saved_data

    def loading_train_dataset(self, paths: list, label_name: str) -> dict:
        '''
        Actually for topic_based method, we need 3 input, pre sentence, q-a pair
        For the first simplest version, we need pre sentences(N), pre-topic(28), and q-a pair(k)
        :param paths:
        :param label_name:
        :return:
        '''

        df_stacking_csv = pd.read_csv('dataset/processed/update_index_Volatility.csv')
        sample_path_set = set(df_stacking_csv['path'].tolist())
        print(f"sample_path_set size: {len(sample_path_set)}")
        X_pre = []
        X_q = []
        X_a = []
        Y = []
        X_qa_topic = []
        X_pre_topic = []
        X_path = []
        for path in paths:
            with open(path, 'rb') as fIn:
                stored_datas = pickle.load(fIn)
                print(f"Loading {path} with {len(stored_datas)} samples.")
                if self.lock_target:
                    for stored_data in stored_datas:  # call
                        if stored_data['path'] == 'dataset/earningscall/0/4169992-materialises-mtls-ceo-wilfried-vancraen-q1-2018-results-earnings-call-transcript.csv':
                            q_rounds = []
                            a_rounds = []
                            for round_q, round_a in zip(stored_data['q_reps'], stored_data['a_reps']):
                                q_rounds.append(np.mean(round_q, axis=0))
                                a_rounds.append(np.mean(round_a, axis=0))
                            if (len(stored_data['pre_reps']) != 0) and (len(stored_data['q_reps']) != 0) and (len(stored_data['a_reps']) != 0):

                                X_pre.append(stored_data['pre_reps'])
                                X_q.append(np.stack(q_rounds, axis=0))
                                X_a.append(np.stack(a_rounds, axis=0))
                                Y.append(stored_data['label'][label_name])
                else:
                    for stored_data in tqdm(stored_datas):
                        if (len(stored_data['pre_reps']) != 0) and (len(stored_data['q_reps']) != 0) and (len(stored_data['a_reps']) != 0):
                            stored_path = os.path.basename(stored_data['path'])[:-4]
                            condition = df_stacking_csv['path'].str.contains(stored_path)

                            selected_rows = df_stacking_csv[condition]

                            if selected_rows.empty:
                                continue
                            if label_name in ['firm_std_3_post', 'firm_std_7_post', 'firm_std_15_post']:
                                full_path = stored_data['path']
                                basename = os.path.basename(full_path)  
                                basename_no_ext = os.path.splitext(basename)[0]

                                matches = label_source_df[label_source_df['path'].str.contains(basename_no_ext)]

                                if len(matches) == 0:
                                    print('N/A')
                                    continue
                                else:
                                    cur_val = matches[label_name]
                                    if cur_val.values[0] == 0.:
                                        print('N/A')
                                        continue
                                    else:
                                        Y.append(cur_val.values[0])
                                        X_path.append(stored_data['path'])
                            else:
                                Y.append(stored_data['label'][label_name])
                                X_path.append(stored_data['path'])

                            q_rounds = []
                            a_rounds = []
                            qa_topics = []
                            for round_q, round_a, qa_topic in zip(stored_data['q_reps'], stored_data['a_reps'], stored_data['qa_topic_pro']):
                            
                                vector = [0.0] * 50
                                for index, value in qa_topic:
                                    vector[index] = value
                                qa_topics.append(vector)
                                
                                q_rounds.append(np.mean(round_q, axis=0))
                                a_rounds.append(np.mean(round_a, axis=0))

                            X_q.append(np.stack(q_rounds, axis=0))
                            X_a.append(np.stack(a_rounds, axis=0))
                            X_pre.append(stored_data['pre_reps'])
                            
                            X_qa_topic.append(np.stack(qa_topics, axis=0))
                            vector = [0.0] * 50
                            for index, value in stored_data['pre_topic_pro']:
                                vector[index] = value
                            X_pre_topic.append(np.array(vector))

                if self.clustering:
                    self.clustering_samples.append(np.expand_dims(
                        stored_data['pre_reps'].mean(axis=0), axis=0))
                    self.clustering_samples.append(
                        (np.array(q_rounds) + np.array(a_rounds)) / 2)

        saved_data = {
            'path': X_path, 
            'pre': X_pre,
            'q': X_q,
            'a': X_a,
            'label': Y,
            'pre_topic': X_pre_topic,
            'qa_topic': X_qa_topic
        }


        return saved_data

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return {
            'path': self.input_path[index], 
            'pre': torch.tensor(np.array(self.input_pre[index])),
            'q': torch.from_numpy(self.input_q[index]).float(),
            'a': torch.from_numpy(self.input_a[index]).float(),
            'label': torch.tensor(np.array(self.label[index])),
            'pre_topic': torch.from_numpy(self.input_pre_topic[index]).float(),
            'qa_topic': torch.from_numpy(self.input_qa_topic[index]).float()
        }


class ModelWrappedWithMSELoss(nn.Module):
    def __init__(self, device):
        super(ModelWrappedWithMSELoss, self).__init__()
        self.model = None
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.device = device

    def init_model(self, args):
        self.model = TopicProfetGNN(**vars(args)).to(self.device)

    def forward(self, inputs, target):
        output = self.model(*inputs)
        target = target.view(-1).to(torch.float32)
        output = output.view(target.size(0), -1).to(torch.float32)
        if output.size(1) == 1:
            output = output.view(target.size(0))
        select_loss = self.criterion(output, target)
        select_loss = select_loss.mean().view(1, 1)
        backward_loss = select_loss
        return backward_loss, select_loss, output


def pack_to_longest(batch):
    path_list = []
    pre_list = []
    q_list = []
    a_list = []
    label_list = []
    pre_topic_list = []
    qa_topic_list = []
    for b in batch:
        path_list.append(b['path'])
        pre_list.append(b['pre'])
        q_list.append(b['q'])
        a_list.append(b['a'])
        label_list.append(b['label'])
        
        identity_matrix = torch.eye(50)
        pre_topic_list.append(identity_matrix)
        qa_topic_list.append(b['qa_topic'])

    len_pre = torch.as_tensor([v.size(0) for v in pre_list])
    len_qa = torch.as_tensor([v.size(0) for v in q_list])
    pre_mask = generate_mask(len_pre)
    qa_mask = generate_mask(len_qa)

    new_pre = pad_sequence(pre_list, batch_first=True)
    new_q = pad_sequence(q_list, batch_first=True)
    new_a = pad_sequence(a_list, batch_first=True)
    
    new_pre_topic = pad_sequence(pre_topic_list, batch_first=True)
    new_qa_topic = pad_sequence(qa_topic_list, batch_first=True)

    label = torch.tensor(label_list)

        pre_len = 50
        qa_max_len = len_qa.max()
        gnn_masks = []
        for gnn_qa_real_len in len_qa:
            gnn_mask = np.tril(
                np.ones((pre_len + qa_max_len, pre_len + qa_max_len)))
            gnn_qa_mask = np.zeros((pre_len + qa_max_len, pre_len + qa_max_len))
            gnn_qa_mask[0:pre_len + gnn_qa_real_len,
                        0:pre_len + gnn_qa_real_len] = 1
            gnn_mask = gnn_mask * gnn_qa_mask
            gnn_mask[:pre_len, :pre_len] = np.eye(pre_len)
            gnn_mask[:pre_len, pre_len:] = 0
            
            gnn_masks.append(gnn_mask)

    gnn_masks = torch.tensor(np.array(gnn_masks)).float()

    return {
        'path': path_list,
        'pre': new_pre,
        'q': new_q,
        'a': new_a,
        'label': label,
        'pre_mask': pre_mask,
        'qa_mask': qa_mask,
        'gnn_mask': gnn_masks,
        'padded_len': len_qa,
        'pre_topic': new_pre_topic,
        'qa_topic': new_qa_topic
    }


def generate_mask(len_list):
    '''
    need input like this: tensor([124, 59, 15, 177,...])
    :param len_list:
    :return:
    '''
    sequence_length = torch.LongTensor(len_list)
    batch_size = len_list.size(0)
    max_len = len_list.max()
    seq_range = torch.arange(0, max_len)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = sequence_length.unsqueeze(
        1).expand_as(seq_range_expand)

    ''' or '''
    mask = torch.zeros(batch_size, max_len, dtype=torch.float)
    for e_id, src_len in enumerate(len_list):
        mask[e_id, :src_len] = 1

    return seq_range_expand < seq_length_expand


class GCATrainer(object):
    def __init__(self,
                 args,
                 config,
                 grad_clip=None,
                 patience_epochs=10
                 ):
        logging.info(f"initialize {self.__class__.__name__}")
        self.args = deepcopy(args)
        self.config = deepcopy(config)
        torch.set_num_threads(os.cpu_count())

        # tensorboard
        tb_path = os.path.join(self.args.result, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)
        os.makedirs(tb_path)
        self.tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

        # model
        self.model_with_loss = ModelWrappedWithMSELoss(self.config.device)
        self.model_with_loss.init_model(self.config.model)
        self.model, self.criterion = self.model_with_loss.model, self.model_with_loss.criterion

        num_parameters = sum([l.nelement() for l in self.model.parameters()])
        logging.info("number of parameters: %d", num_parameters)

        optim_params = vars(self.config.optim)
        if optim_params['optimizer'] == 'Adagrad':
            del optim_params['optimizer']
            optimizer = optim.Adagrad(self.model.parameters(), **optim_params)
        elif optim_params['optimizer'] == 'Adam':
            del optim_params['optimizer']
            optimizer = optim.Adam(self.model.parameters(), **optim_params)
        else:
            raise AssertionError(
                'According to the original paper, you should use "Adagrad" as the optimizer')
        self.optimizer = optimizer

        if self.config.scheduler.is_scheduler:
            logging.info('Use lr_scheduler.')
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=self.config.scheduler.lr_dc_step,
                                                             gamma=self.config.scheduler.lr_dc)

    def train(self):
        train_step = 0
        test_step = 0

        best_score = float('inf')

        train_paths = self.config.data.train_path[:-1] if len(self.config.data.train_path) > 1 else self.config.data.train_path
        val_paths = [self.config.data.train_path[-1]] if len(self.config.data.train_path) > 1 else self.config.data.test_path[:1]
        
        train_dataset = TopicProfetGNNDataset(train_paths,
                                              label_name=self.config.data.label)
        print(f"Train dataset size: {train_dataset.__len__()}") 

        eval_dataset = TopicProfetGNNDataset(val_paths,
                                             label_name=self.config.data.label)

        logging.info(f"Training. Total {train_dataset.__len__()} train data, total {eval_dataset.__len__()} valid data.")
        
        for epoch in range(self.config.train.n_epochs):
            logging.info(f"Epoch {epoch}")

            train_dataloader = DataLoader(train_dataset,
                                          batch_size=self.config.train.batch_size,
                                          shuffle=True,
                                          num_workers=1,
                                          pin_memory=True,
                                          collate_fn=pack_to_longest,
                                          drop_last=True)
            logging.info(
                f"Train dataloader size: {len(train_dataloader)} batches, each batch size: {self.config.train.batch_size}")
            eval_dataloader = DataLoader(eval_dataset,
                                         batch_size=self.config.train.batch_size,
                                         shuffle=True,
                                         num_workers=1,
                                         pin_memory=True,
                                         collate_fn=pack_to_longest,
                                         drop_last=True)

            '''train'''
            self.model.train()
            total_train_loss = []
            for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                train_step += 1
                self.optimizer.zero_grad()
                input = [data['pre'].to(self.config.device),
                         data['q'].to(self.config.device),
                         data['a'].to(self.config.device),
                         data['pre_mask'].to(self.config.device),
                         data['qa_mask'].to(self.config.device),
                         data['gnn_mask'].to(self.config.device),
                         data['padded_len'],
                         data['pre_topic'].to(self.config.device),
                         data['qa_topic'].to(self.config.device),
                         ]
                label = data['label'].to(self.config.device)
                backward_loss, select_loss, output = self.model_with_loss(
                    input, label)
                backward_loss.backward()
                self.optimizer.step()
                total_train_loss.append(backward_loss.item())

                self.tb_logger.add_scalar(
                    'train_loss', backward_loss, global_step=train_step)
                self.tb_logger.add_scalar(
                    'learning_rate', self.optimizer.param_groups[0]['lr'], global_step=train_step)

            '''eval'''
            self.model.eval()
            total_eval_loss = []
            cur_eval_score = []
            with torch.no_grad():
                for i, data in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
                    test_step += 1
                    input = [data['pre'].to(self.config.device),
                             data['q'].to(self.config.device),
                             data['a'].to(self.config.device),
                             data['pre_mask'].to(self.config.device),
                             data['qa_mask'].to(self.config.device),
                             data['gnn_mask'].to(self.config.device),
                             data['padded_len'],
                             data['pre_topic'].to(self.config.device),
                             data['qa_topic'].to(self.config.device),
                             ]
                    label = data['label'].to(self.config.device)
                    backward_loss, select_loss, output = self.model_with_loss(
                        input, label)
                    total_eval_loss.append(backward_loss.item())
                    cur_eval_score.append(select_loss.item())

                    self.tb_logger.add_scalar(
                        'eval_loss', backward_loss, global_step=test_step)
            cur_score = np.mean(cur_eval_score)
            if cur_score < best_score:
                logging.info(
                    f"best score is: {best_score}, current score is: {cur_score}, save best_checkpoint.pth")
                best_score = cur_score
                states = [
                    self.model.state_dict(),
                    self.optimizer.state_dict(),
                ]
                save_path = os.path.join(self.args.checkpoint, 'best_checkpoint.pth')
                print(save_path)
                torch.save(states, os.path.join(
                    self.args.checkpoint, 'best_checkpoint.pth'))

            if self.config.scheduler.is_scheduler:
                self.scheduler.step()
                logging.info(
                    f'Learning rate = {self.scheduler.get_last_lr()[0]}')

        states = [
            self.model.state_dict(),
            self.optimizer.state_dict(),
        ]
        torch.save(states, os.path.join(
            self.args.checkpoint, 'latest_checkpoint.pth'))
        gc.collect()

    def test(self, load_pre_train=False):
        if load_pre_train:
            if self.config.checkpoint.choice == 'best':
                pretrained_data = torch.load(os.path.join(
                    self.args.checkpoint, 'best_checkpoint.pth'))
            elif self.config.checkpoint.choice == 'latest':
                pretrained_data = torch.load(os.path.join(
                    self.args.checkpoint, 'latest_checkpoint.pth'))
            self.model.load_state_dict(pretrained_data[0])

        test_dataset = TopicProfetGNNDataset(self.config.data.test_path,
                                             label_name=self.config.data.label, lock_target=False)

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=self.config.train.batch_size,
                                     shuffle=False,
                                     num_workers=2,
                                     pin_memory=True,
                                     collate_fn=pack_to_longest,
                                     drop_last=False)

        logging.info(f"Testing. Total {test_dataset.__len__()} data.")
        self.model.eval()
        mse_list = []
        mae_list = []
        x = []
        y = []
        data_path = []
        with torch.no_grad():
            for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                data_path.extend(data['path'])
                self.optimizer.zero_grad()
                input = [data['pre'].to(self.config.device),
                         data['q'].to(self.config.device),
                         data['a'].to(self.config.device),
                         data['pre_mask'].to(self.config.device),
                         data['qa_mask'].to(self.config.device),
                         data['gnn_mask'].to(self.config.device),
                         data['padded_len'],
                         data['pre_topic'].to(self.config.device),
                         data['qa_topic'].to(self.config.device),
                         ]
                label = data['label'].to(self.config.device)
                backward_loss, select_loss, output = self.model_with_loss(
                    input, label)
                mse_list.append(backward_loss.item())
                mae_list.append(mean_absolute_error(
                    output.cpu().numpy(), label.cpu().numpy()))
                x.extend(output.cpu().numpy().tolist())
                y.extend(label.cpu().numpy().tolist())
            spearman_corr, spearman_pvalue = stats.spearmanr(x, y)
            print(f"Spearman's correlation: {spearman_corr}")
            print(f"Spearman's p-value: {spearman_pvalue}")

            # Calculate Kendall's tau coefficient
            kendall_tau, kendall_pvalue = stats.kendalltau(x, y)
            print(f"Kendall's tau: {kendall_tau}")
            print(f"Kendall's p-value: {kendall_pvalue}")
            
            df = pd.DataFrame(
                {
                    'file_path': data_path,
                    'output': x,
                    'label': y
                }
            )

        logging.info(
            f'label is {self.config.data.label}, mse is {np.mean(mse_list)}, mae is {np.mean(mae_list)}')
    
    @torch.inference_mode()
    def test_one_firm_with_interpretability(self):
        '''load model & lda_model'''
        if self.config.checkpoint.choice == 'best':
            pretrained_data = torch.load(os.path.join(
            self.args.checkpoint, 'best_checkpoint.pth'))
            self.model.load_state_dict(pretrained_data[0])
            print(f"load model from: {os.path.join(self.args.checkpoint, 'best_checkpoint.pth')}")
        elif self.config.checkpoint.choice == 'latest':
            pretrained_data = torch.load(os.path.join(
            self.args.checkpoint, 'latest_checkpoint.pth'))
            self.model.load_state_dict(pretrained_data[0])
            print(f"load model from: {os.path.join(self.args.checkpoint, 'latest_checkpoint.pth')}")
        
        self.model.eval()
        
        lda_model_path = os.path.join('result', 'checkpoint', 'LDA', 'sent_lda.model')
        model = gensim.models.wrappers.LdaMallet.load(lda_model_path)
        lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model)
        
        lda_dict_path = os.path.join('result', 'checkpoint', 'LDA', 'sent_lda_id2word.pkl')
        with open(lda_dict_path, 'rb') as fIn:
            id2word = pickle.load(fIn)
        
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        results_path = os.path.join('result', 'output', 'results_60d.csv')
        df = pd.read_csv(results_path)
        sorted_df = df.sort_values(by='label_60d', ascending=False)
        
        positive_words = read_dict(os.path.join('data', 'LM_positive.txt'))
        negative_words = read_dict(os.path.join('data', 'LM_negative.txt'))
        
        counter = 0
        
        inter_topic_neg_list = []
        
        for file_path, firm_risk in tqdm(zip(sorted_df['file_path'], sorted_df['label_60d']), total=len(sorted_df['file_path'])):
            cur_df = pd.read_csv(file_path)
            cur_df.dropna(inplace=True)
            cur_df.reset_index(drop=True, inplace=True)
            
            cur_data = {}
            cur_data['path'] = file_path
            cur_data['label'] = firm_risk
            
            pre = cur_df.loc[(cur_df['section'] == 'intro')]['sentence']
            merged_pre = " ".join(pre)
            pre_sentences = sent_tokenize(merged_pre)
            pre_reps = encoder.encode(pre_sentences, show_progress_bar=False)
            cur_data['pre_reps'] = pre_reps

            preprocessed_pre = ' '.join(preprocess(pre)).split()
            corpus = id2word.doc2bow(preprocessed_pre)
            pre_topic_pro = lda_model.get_document_topics(corpus)
            cur_data['pre_topic_pro'] = pre_topic_pro

            round_qr = []  # save all round q representations
            round_ar = []  # save all round a representations
            round_qat = []
            q_words = []
            a_words = []
            q_index = cur_df.loc[(cur_df['section'] == 'qa') & (cur_df['role'] == 'analyst')].index
            for i in range(len(q_index) - 1):
                if q_index[i + 1] - q_index[i] > 1:  ## > 1 represents containing "a"
                    merged_q = " ".join(cur_df.loc[[q_index[i]]]['sentence'])
                    round_q_sentences = sent_tokenize(merged_q)
                    round_q_reps = encoder.encode(round_q_sentences, show_progress_bar=False)
                    round_qr.append(round_q_reps)

                    preprocessed_q = ' '.join(preprocess(cur_df.loc[[q_index[i]]]['sentence'])).split()
                    q_words.append(preprocessed_q)
                        
                    a_index = list(range(q_index[i] + 1, q_index[i + 1]))
                    merged_a = " ".join(cur_df.loc[a_index]['sentence'])
                    round_a_sentences = sent_tokenize(merged_a)
                    round_a_reps = encoder.encode(round_a_sentences, show_progress_bar=False)
                    round_ar.append(round_a_reps)

                    preprocessed_a = ' '.join(preprocess(cur_df.loc[a_index]['sentence'])).split()
                    a_words.append(preprocessed_a)
                    
                    corpus = id2word.doc2bow(preprocessed_q + preprocessed_a)
                    qa_topic_pro = lda_model.get_document_topics(corpus)
                    round_qat.append(qa_topic_pro)

            q_sample_topic_senti = []
            for i, q_round_words in enumerate(q_words):
                positive_count, negative_count = count_emotional_words(q_round_words, positive_words, negative_words)
                q_sample_topic_senti.append(round((positive_count - negative_count) / (positive_count + negative_count + 1e-7), 3))
            
            a_sample_topic_senti = []
            for i, a_round_words in enumerate(a_words):
                positive_count, negative_count = count_emotional_words(a_round_words, positive_words, negative_words)
                a_sample_topic_senti.append(round((positive_count - negative_count) / (positive_count + negative_count + 1e-7), 3))

            cur_data['q_reps'] = round_qr
            cur_data['a_reps'] = round_ar
            cur_data['qa_topic_pro'] = round_qat
            
            X_path = []
            X_pre = []
            X_q = []
            X_a = []
            X_qa_topic = []
            X_pre_topic = []
            Y = []
            
            for _ in range(64):
                X_path.append(cur_data['path'])
                Y.append(cur_data['label'])
                q_rounds = []
                a_rounds = []
                qa_topics = []
                for round_q, round_a, qa_topic in zip(cur_data['q_reps'], cur_data['a_reps'], cur_data['qa_topic_pro']):
                    vector = [0.0] * 50
                    for index, value in qa_topic:
                        vector[index] = value
                    qa_topics.append(vector)
                    q_rounds.append(np.mean(round_q, axis=0))
                    a_rounds.append(np.mean(round_a, axis=0))
                X_q.append(np.stack(q_rounds, axis=0))
                X_a.append(np.stack(a_rounds, axis=0))
                X_pre.append(cur_data['pre_reps'])
                
                X_qa_topic.append(np.stack(qa_topics, axis=0))
                
                identity_matrix = torch.eye(50)
                X_pre_topic.append(identity_matrix)
            
            len_pre = torch.as_tensor([len(v) for v in X_pre])
            len_qa = torch.as_tensor([len(v) for v in X_q])
            pre_mask = generate_mask(len_pre)
            qa_mask = generate_mask(len_qa)

            new_pre = pad_sequence(torch.tensor(X_pre), batch_first=True)
            new_q = pad_sequence(torch.tensor(X_q), batch_first=True)
            new_a = pad_sequence(torch.tensor(X_a), batch_first=True)
            
            new_pre_topic = pad_sequence(X_pre_topic, batch_first=True)
            new_qa_topic = pad_sequence(torch.tensor(X_qa_topic), batch_first=True)
            
            label = torch.tensor(Y)
            
            pre_len = 50
            qa_max_len = len_qa.max()
            gnn_masks = []
            for gnn_qa_real_len in len_qa:
                gnn_mask = np.tril(
                    np.ones((pre_len + qa_max_len, pre_len + qa_max_len)))
                gnn_qa_mask = np.zeros((pre_len + qa_max_len, pre_len + qa_max_len))
                gnn_qa_mask[0:pre_len + gnn_qa_real_len,
                            0:pre_len + gnn_qa_real_len] = 1
                gnn_mask = gnn_mask * gnn_qa_mask
                gnn_mask[:pre_len, :pre_len] = np.eye(pre_len)
                gnn_mask[:pre_len, pre_len:] = 0
                gnn_masks.append(gnn_mask)

            gnn_masks = torch.tensor(np.array(gnn_masks)).float()
            
            inputs = [new_pre.to(self.config.device),
                    new_q.to(self.config.device),
                    new_a.to(self.config.device),
                    pre_mask.to(self.config.device),
                    qa_mask.to(self.config.device),
                    gnn_masks.to(self.config.device),
                    len_qa,
                    new_pre_topic.to(self.config.device),
                    new_qa_topic.to(self.config.device)
                    ]
            
            label = label.to(self.config.device)
            
            output = self.model(*inputs)
            target = label.view(-1).to(torch.float32)
            output = output.view(target.size(0), -1).to(torch.float32)
            if output.size(1) == 1:
                output = output.view(target.size(0))
            ori_loss = F.mse_loss(output, target)
            
            if not ori_loss.item() < 0.1619:
                continue
            
            node_score = {}
            
            for i in range(qa_max_len):
                modified_q = new_q
                modified_q[:, i, :] = 0
                modified_a = new_a
                modified_a[:, i, :] = 0
                
                inputs = [new_pre.to(self.config.device),
                        modified_q.to(self.config.device),
                        modified_a.to(self.config.device),
                        pre_mask.to(self.config.device),
                        qa_mask.to(self.config.device),
                        gnn_masks.to(self.config.device),
                        len_qa,
                        new_pre_topic.to(self.config.device),
                        new_qa_topic.to(self.config.device)
                        ]
                
                label = label.to(self.config.device)
                
            
                output = self.model(*inputs)
                target = label.view(-1).to(torch.float32)
                output = output.view(target.size(0), -1).to(torch.float32)
                if output.size(1) == 1:
                    output = output.view(target.size(0))
                loss = F.mse_loss(output, target)
                
                node_score[str(i)] = np.abs(loss.item()-ori_loss.item())
            
            sorted_index = sorted(node_score.keys(), key=lambda k: node_score[k], reverse=True)
            
            inter_topic_neg = {}
            inter_topic_neg['path'] = file_path
            inter_topic_neg['label'] = firm_risk
            inter_topic_neg['loss'] = ori_loss.item()
            inter_topic_neg['q_senti'] = q_sample_topic_senti
            inter_topic_neg['a_senti'] = a_sample_topic_senti
            inter_topic_neg['node_score'] = node_score
            inter_topic_neg['sorted_index'] = sorted_index
            
            
            inter_topic_neg_list.append(inter_topic_neg)
            counter += 1
        
        output_path = os.path.join('result', 'output', 'inter_topic_neg.pkl')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as file: 
            pickle.dump(inter_topic_neg_list, file)
        
        print(f'Total {counter} samples saved to {output_path}')

