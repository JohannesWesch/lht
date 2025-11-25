import logging

import torch
from datasets import load_dataset

# from src.HDT import HDTTokenizer
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader

# from src.utils import module_to_dict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from lht.utils.nested_builder import build_coords_from_nested_list

from .basic import BasicDataModule

# import configs as CONFIG
# from src.utils import *

log = logging.getLogger(__name__)


class HierarchicalDataCollator(DataCollatorForLanguageModeling):
    """
    Collator that handles:
    1. Tokenizing nested list data on-the-fly (or receiving pre-tokenized data with positions).
    2. Padding input_ids.
    3. Masking tokens for MLM.
    4. Collate hierarchical positions (pad/batch).
    """

    def __init__(self, tokenizer, mlm_probability=0.15, max_length=8192):
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
        self.max_length = max_length

    def torch_call(self, examples):
        batch_input_ids = []
        batch_coords = []

        for ex in examples:
            if isinstance(ex, dict) and "text" in ex:
                doc = ex["text"]
            else:
                doc = ex

            input_ids, coords = build_coords_from_nested_list(
                doc, self.tokenizer, max_length=self.max_length
            )

            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_coords.append(coords)

        batch_input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        batch = super().torch_call([{"input_ids": t} for t in batch_input_ids_padded])

        max_len = batch_input_ids_padded.size(1)

        from lht.core.attention import HierarchicalPositions

        # Pad/truncate each level's enumeration vectors
        num_levels = batch_coords[0].num_levels if batch_coords else 3
        batched_level_enums = [[] for _ in range(num_levels)]

        for positions in batch_coords:
            for level_idx in range(num_levels):
                level_enum = positions.level_enums[level_idx]
                current_len = len(level_enum)

                if current_len < max_len:
                    # Pad with zeros (non-participants)
                    padding_len = max_len - current_len
                    level_enum = torch.cat(
                        [
                            level_enum,
                            torch.zeros(
                                padding_len,
                                dtype=level_enum.dtype,
                                device=level_enum.device,
                            ),
                        ]
                    )
                elif current_len > max_len:
                    # Truncate
                    level_enum = level_enum[:max_len]

                batched_level_enums[level_idx].append(level_enum)

        # Stack into [B, N] tensors for each level
        stacked_enums = [torch.stack(level_list) for level_list in batched_level_enums]

        batch["positions"] = HierarchicalPositions(stacked_enums)

        return batch


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
            self.config.data.tokenizer_name_or_path,
            model_max_length=self.config.data.max_seq_len,
        )

        if self.config.model.mlswa:
            self.data_collator = HierarchicalDataCollator(
                tokenizer=self.tokenizer,
                mlm_probability=self.config.training.mlm_probability,
                max_length=self.config.data.max_seq_len,
            )
        else:
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

        # Handle IterableDataset which doesn't support len() or random access
        try:
            # Try to take the first item
            if isinstance(train_dataset, IterableDataset):
                input_data = next(iter(train_dataset))["text"]
                dataset_size_str = "unknown (streaming)"
            else:
                random_sentence_idx = torch.randint(0, len(train_dataset), (1,)).item()
                input_data = train_dataset[random_sentence_idx]["text"]
                dataset_size_str = f"{len(train_dataset):,}"

        except (TypeError, StopIteration):
            # Fallback if empty or not iterable
            log.warning(
                "Could not log tokenization sample: dataset is empty or not iterable."
            )
            return

        log.info(
            f"Sample sentence with seq_length {self.config.data.max_seq_len} from dataset of size {dataset_size_str}: ..."
        )
        # input_data is a list of lists (nested structure), convert to string for logging
        log.info(str(input_data)[:500] + "...")
        log.info("... is tokenized into ...")

        # Build coords to get flat input_ids for logging
        input_ids, _ = build_coords_from_nested_list(
            input_data, self.tokenizer, max_length=512  # Limit for logging
        )
        log.info(" ".join(self.tokenizer.decode(t) for t in input_ids))
