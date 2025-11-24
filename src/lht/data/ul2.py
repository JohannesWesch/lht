import torch

# from src.data.basic import BasicDataModule
# from src.HDT import HDTTokenizer
# from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from datasets import IterableDataset, load_dataset
from torch.utils.data.dataloader import DataLoader

# from src.utils import module_to_dict
from transformers import AutoTokenizer

from .basic import BasicDataModule

# from src.utils.data_collators import DataCollatorForHierarchicalUL2
# from src.utils.tokenization import construct_tokenizer
# import configs as CONFIG
# from src.utils import *


class UL2DataModule(BasicDataModule):
    _name_ = "ul2"
    additional_special_tokens = dict(
        **{
            "additional_special_tokens": ["<doc>", "<sec>", "<NLU>", "<NLG>", "<S2S>"]
            + [f"<extra_id_{i}>" for i in range(500, -1, -1)]
        }
    )

    def __init__(self, config):
        super().__init__(config)

    def _get_tokenizer(self, raw_data):
        # if CONFIG.data_config.tok_name not in ["BPE", "Unigram", "WordLevel", "WordPiece", "WordPieceBERT", "SentencePieceUnigram",
        #                         "SentencePieceBPE"]:
        #     # fix https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/tokenization_t5.py
        #     tokenizer = AutoTokenizer.from_pretrained(CONFIG.cfg_data.tok_name, cls_token="<cls>", bos_token="<s>", additional_special_tokens=self.additional_special_tokens["additional_special_tokens"], extra_ids=0)
        # else:
        #     tokenizer = construct_tokenizer(raw_data, CONFIG.cfg_data.tok_name, CONFIG.cache_dir, CONFIG.cfg_data.preprocess_batch_size, special_tokens={"cls_token": "<cls>", "bos_token": "<bos>"})
        #     tokenizer.add_special_tokens(self.additional_special_tokens, replace_additional_special_tokens=True)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.data.tokenizer_name_or_path
        )
        tokenizer.add_special_tokens(self.additional_special_tokens)
        # tokenizer.save_pretrained(CONFIG.save_dir)
        return tokenizer

    def prepare_data(self) -> None:
        if not self.prepared:
            corpus_list = []
            if self.config.data.ds_info:
                for cfg_dict in self.config.data.ds_info:
                    raw_dataset = load_dataset(**cfg_dict)  # cache_dir=CONFIG.cache_dir
                    corpus_list.append(raw_dataset)
                raw_data = self._concatenate_datasets(corpus_list)
                self.tokenizer = self._get_tokenizer(raw_data)
            # self.tokenizer.save_pretrained(CONFIG.save_dir)
            self.prepared = True

    def setup(self, stage: str) -> None:
        # data_collator = DataCollatorForHierarchicalUL2
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.data.tokenizer_name_or_path
        )
        if stage == "fit":
            corpus_list = []
            if self.config.data.ds_info:
                for cfg_dict in self.config.data.ds_info:
                    raw_dataset = load_dataset(**cfg_dict)  # cache_dir=CONFIG.cache_dir
                    corpus_list.append(raw_dataset)
                self.data_train = self._concatenate_datasets(corpus_list)
            # self.data_collator = data_collator(self.tokenizer, label_max_length=CONFIG.model_config.max_decoder_position_embeddings, input_max_length=CONFIG.model_config.max_encoder_position_embeddings)
            # Using default collator for now as placeholder if DataCollatorForHierarchicalUL2 is missing
            from transformers import DataCollatorForLanguageModeling

            self.data_collator = DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            )

    def train_dataloader(self) -> DataLoader:
        sampler = (
            torch.utils.data.distributed.DistributedSampler(
                self.data_train, drop_last=True
            )
            if self.multi_gpu
            else None
        )
        return DataLoader(
            self.data_train,
            batch_size=self.config.training.batch_size,
            shuffle=(
                sampler is None and not isinstance(self.data_train, IterableDataset)
            ),
            num_workers=self.config.data.num_workers,
            collate_fn=self.data_collator,
            sampler=sampler,
        )
