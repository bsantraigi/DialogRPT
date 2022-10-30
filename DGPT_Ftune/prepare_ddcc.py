import json
import os
import sys
import random
import re

# import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
# from torch.utils.data import ChainDataset, DataLoader, ConcatDataset, IterableDataset, Sampler, DistributedSampler
import glob

import logging
import coloredlogs
from transformers import AutoTokenizer, BlenderbotTokenizer, \
    BertTokenizer, BertTokenizerFast, \
    RobertaTokenizer, RobertaTokenizerFast

# from .r1m_preprocess import filter_dialogs
# from .callbacks import RandomNegativeSampler, FaissResponseSampler, NucleusResponseSampler
C_MAX_LEN = 300
R_MAX_LEN = 60

###############
# LOGGING SETUP
###############
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


class BaseDataClass(Dataset):
    def prep_tokenizer_info(self, tokenizer, max_ctx_len, max_resp_len):
        self.tokenizer = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_resp_len = max_resp_len

        if isinstance(tokenizer, BlenderbotTokenizer):
            self.CLS = tokenizer.bos_token_id
            self.EOU = tokenizer.sep_token
        elif isinstance(tokenizer, BertTokenizer) or isinstance(tokenizer, BertTokenizerFast) \
            or isinstance(tokenizer, RobertaTokenizerFast) or isinstance(tokenizer, RobertaTokenizer):
            # NO NEW TOKEN added because we may init the model with actual Roberta/bert weights
            self.CLS = tokenizer.cls_token_id
            self.EOU = tokenizer.sep_token
        else:
            logger.error("Tokenizer not supported.")
            raise NotImplementedError(f"Tokenizer {tokenizer} is not supported.")

        self.pad_token_id = tokenizer.pad_token_id

        assert self.CLS is not None, "CLS token not found."
        assert self.EOU is not None, "EOU token not found."

        logger.debug(f"[TOKENIZER] cls: {self.CLS}, sep: {self.EOU}, pad: {self.pad_token_id}")
        logger.debug(f"[TOKENIZER] {tokenizer}")

    def collate_fn(self, batch):
        morphed_batch = pd.DataFrame(batch).to_dict(orient="list")
        final_batch = {
            "premise": pad_sequence(morphed_batch["premise"], batch_first=True, padding_value=self.pad_token_id),
            "hypothesis": pad_sequence(morphed_batch["hypothesis"], batch_first=True, padding_value=self.pad_token_id),
            "premise_length": torch.tensor(morphed_batch["premise_length"]),
            "hypothesis_length": torch.tensor(morphed_batch["hypothesis_length"]),
            "label": torch.tensor(morphed_batch["label"]),
            "index": torch.tensor(morphed_batch["index"])
        }
        return final_batch

    def _preprocess(self, C, R=None):
        # should be on cpu to support multiple workers in dataloader
        # for blender
        # c = self.tokenizer.encode("<s> " + C)
        # r = self.tokenizer.encode("<s> " + R)

        # for bert
        # c = self.tokenizer.encode(C)
        # r = self.tokenizer.encode(R)

        c = self.tokenizer.encode(C, add_special_tokens=False)
        l1 = len(c)
        if l1 >= self.max_ctx_len:
            c = c[l1 - self.max_ctx_len + 1:]
        c = [self.CLS] + c
        c = torch.tensor(c)

        if R is not None:
            r = self.tokenizer.encode(R, add_special_tokens=False)
            l2 = len(r)
            if l2 >= self.max_resp_len:
                r = r[:self.max_resp_len - 1]
            r = [self.CLS] + r
            r = torch.tensor(r)
        else:
            r = None

        return c, r
    
    def __len__(self):
        return len(self.data)


class DialogData(BaseDataClass):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, data_path, tokenizer, prob_positive,
                 max_ctx_len=C_MAX_LEN, max_resp_len=R_MAX_LEN, min_dial_len=2, resp_sampler_callback=None):
        """
        @param tokenizer: A huggingface tokenizer
        @param neg_per_positive: (npp) can be between 0 to 1 or any integer greater than 1.
        """
        super(DialogData, self).__init__()
        _file = data_path

        logger.debug(f"File: {_file}")

        self.prep_tokenizer_info(tokenizer, max_ctx_len, max_resp_len)

        # npp
        assert prob_positive > 0, "Probability of positive dialogs must be greater than 0." 
        self.prob_positive = prob_positive
        logger.debug(f"Probability of positive sample: {self.prob_positive}")

        self.dial_data = []

        with open(_file) as f:
            for line in tqdm(f, desc="Loading data"):
                # if len(self.data) > max_items:
                #     break  # Enough for now
                Full_D = line.strip().strip("__eou__").split(" __eou__ ")
                self.dial_data.append(Full_D)
        
        # need this to match the testset with other libraries!
        self.min_dial_len = min_dial_len
        self.num_positives = -1
        self.extract_cr_pairs()

    def extract_cr_pairs(self):
        self.data = []
        self.data_only_positives = []
        MIN_DIAL_LEN = self.min_dial_len

        for Full_D in tqdm(self.dial_data, desc="Unrolling dialogs"):
            if len(Full_D) >= MIN_DIAL_LEN:
                for j in range(MIN_DIAL_LEN, len(Full_D) + 1):
                    D = Full_D[:j]
                    C = " ".join(D[:-2]).strip() + " "
                    S = D[-2].strip() + f" "
                    R = D[-1].strip() + f" "
                    # mid = len(D)//2
                    # C = " __eou__ ".join(D[:mid])
                    # R = " __eou__ ".join(D[mid:])

                    # For 1 item in wo_negatives
                    self.data_only_positives.append([C, R])
                    pos_item_index = len(self.data_only_positives) - 1
                    self.data.append([C, S, R, pos_item_index])

        self.num_positives = len(self.data_only_positives)
        logger.debug(f"Loaded {len(self.data_only_positives)} (+) CR-samples.")
        logger.debug(f"Generated {len(self.data)} (+/-) CR-samples.")
        logger.debug(f"Samples: {self.data[random.randint(0, len(self.data))]}")

    def __getitem__(self, index):
        C, S, R, pos_item_index = self.data[index]
        
        # positive sample
        # c, r = self._preprocess(C, R)
        # label = 1

        return [C,S,R]

if __name__=="__main__":
    logger.debug("Running Unit Tests")

    # vocab_model_reference = 'facebook/blenderbot-3B'
    vocab_model_reference = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(vocab_model_reference, use_fast=True, verbose=False)    
        
    # Datasets and Dataloaders
    dataset = "dd_cc"
    data_path_root = sys.argv[1]
    output_path_root = sys.argv[2]
    
    # data_path_root = "/home/bishal/HULK/Projects/DialogRPT/data/ijcnlp_dailydialog_cc/"
    # output_path_root = "/home/bishal/HULK/Projects/DialogRPT/preprocessed_data/"
    
    # http://10.5.30.155:9155/lab/tree/HULK/Projects/DialogRPT/data/ijcnlp_dailydialog_cc/validation/dialogues_validation.txt
    
    train_path = os.path.join(data_path_root, f'train/dialogues_train.txt')
    valid_path = os.path.join(data_path_root, f'validation/dialogues_validation.txt')
    test_path = os.path.join(data_path_root, f'test/dialogues_test.txt')

    valid_dataset = DialogData(valid_path, tokenizer, prob_positive=1.0, min_dial_len=3)
    train_dataset = DialogData(train_path, tokenizer, prob_positive=1.0, min_dial_len=3)
    test_dataset = DialogData(test_path, tokenizer, prob_positive=1.0, min_dial_len=3)

    print(len(train_dataset), len(valid_dataset), len(test_dataset))

    if not os.path.isdir(output_path_root):
        os.makedirs(output_path_root)

    with open(os.path.join(output_path_root, "dd_train_input.tsv"), "w") as f:
        for c,s,r in train_dataset:
            f.write(f"{c}\t{s}\t{r}\n")

    with open(os.path.join(output_path_root, "dd_validation_input.tsv"), "w") as f:
        for c,s,r in valid_dataset:
            f.write(f"{c}\t{s}\t{r}\n")

    with open(os.path.join(output_path_root, "dd_test_input.tsv"), "w") as f:
        for c,s,r in test_dataset:
            f.write(f"{c}\t{s}\t{r}\n")