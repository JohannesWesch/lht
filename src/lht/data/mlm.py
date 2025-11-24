import logging

import torch
from datasets import load_dataset

# from src.HDT import HDTTokenizer
from torch.utils.data.dataloader import DataLoader

# from src.utils import module_to_dict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from .basic import BasicDataModule

# import configs as CONFIG
# from src.utils import *

log = logging.getLogger(__name__)


class MLMDataModule(BasicDataModule):
    _name_ = "mlm"

    def __init__(self, config):
        super().__init__(config)

    def prepare_data(self) -> None:
        if not self.prepared:
            # We assume ds_info is a list of dicts with dataset args
            if self.config.data.ds_info:
                # TODO: handle cache_dir from config or default
                pass

            # Since we are using streaming or on-the-fly, maybe we don't need to save tokenizer here
            # But we can check if we need to train a tokenizer
            pass
            self.prepared = True

    def setup(self, stage: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.data.tokenizer_name_or_path
        )
        # self.data_collator = DataCollatorForMaskedLanguageModeling(self.tokenizer, CONFIG.cfg_data.mlm_probability, input_max_length=CONFIG.cfg_data.model_max_length, hierarchical=CONFIG.hierarchical)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.config.training.mlm_probability,
        )

        if stage == "fit":
            corpus_list = []
            if self.config.data.ds_info:
                for cfg_dict in self.config.data.ds_info:
                    # raw_dataset = load_dataset(**cfg_dict, cache_dir=CONFIG.cache_dir)
                    raw_dataset = load_dataset(**cfg_dict)
                    corpus_list.append(raw_dataset)
                self.data_train = self._concatenate_datasets(corpus_list)
                self._log_tokenization(self.data_train)
            else:
                # Fallback or error if no dataset info
                pass
            # self.data_train = preprocess(self.data_train, self.tokenizer, self.cfg_data)
        elif stage == "test":
            ## Validation set always use AG_news
            test_dataset = load_dataset(
                "ag_news", split="train", num_proc=self.config.data.num_workers
            )
            self.data_test = test_dataset.remove_columns(
                [col for col in test_dataset.column_names if col != "text"]
            )
            # self.data_test = preprocess(test_dataset, self.tokenizer, self.cfg_data)

    def val_dataloader(self) -> DataLoader:
        # Just taking a slice for validation as per original code
        # TODO: Better validation split
        return DataLoader(
            self.data_train,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            collate_fn=self.data_collator,
        )  # self.data_train[-1000:]


class HierarchicalMLMDataModule(MLMDataModule):
    _name_ = "hierarchical_mlm"
    special_tokens = dict(cls_token="<cls>", sec_token="<sec>", doc_token="<doc>")

    def _log_tokenization(self, train_dataset):
        # hdt_tokenizer = HDTTokenizer(self.tokenizer, CONFIG.cfg_data.model_max_length)
        # 4) Log overviews so we always know what's going on with weird tokenization tricks
        random_sentence_idx = torch.randint(0, len(train_dataset), (1,)).item()
        input_data = train_dataset[random_sentence_idx]["text"]
        # .squeeze()  # squeeze because hf has leading dim
        dataset_size = len(train_dataset)

        log.info(
            f"Random sentence with seq_length {self.config.data.max_seq_len} from dataset of size {dataset_size:,}: ..."
        )
        log.info(input_data)
        log.info("... is tokenized into ...")
        # tokenized_doc = hdt_tokenizer(input_data)["input_ids"]
        tokenized_doc = self.tokenizer(input_data)["input_ids"]
        log.info(" ".join(self.tokenizer.decode(t) for t in tokenized_doc))
