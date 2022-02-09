from transformers.optimization import AdamW, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from typing import List, Union, Optional
import torch
from .arguments import args
from .techniques import tech2, eda


class Trainer:
    def __init__(self, model, tokenizer, len_dataloader):
        self.model = model
        self.model.train()
        self.model.to(args.device)
        self.model.resize_token_embeddings(len(tokenizer))  # vocab size대로 embedding metrix resize
        self.tokenizer = tokenizer
        self.optim = AdamW(self.model.parameters(), lr=args.lr)
        if args.lr_cycle == 2:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optim, num_warmup_steps=100//args.batch_size, num_training_steps=len_dataloader*args.n_epochs-100,
                num_cycles=2
            )
        else:
            self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                self.optim, num_warmup_steps=100//args.batch_size, num_training_steps=len_dataloader*args.n_epochs-100,
                num_cycles=args.lr_cycle
            )

    def encode(self, input_texts: List[str], target_texts: List[str]):
        if isinstance(input_texts, tuple):
            input_texts = list(input_texts)
        if isinstance(target_texts, tuple):
            target_texts = list(target_texts)

        inputs_batch = self.tokenizer(input_texts, return_tensors='pt', padding=True)  # {(batch_size, tokens), ..}
        targets_batch = self.tokenizer(target_texts, return_tensors='pt', padding=True)  # {(batch_size, tokens), ..}

        max_length = self.tokenizer.max_model_input_sizes[args.model_name]
        while inputs_batch['input_ids'].shape[-1] > max_length and len(input_texts) > 1:
            idx = sorted(list(range(len(input_texts))), key=lambda x: -len(input_texts[x]))[0]  # 가장 긴 텍스트
            del input_texts[idx]
            del target_texts[idx]
            inputs_batch = self.tokenizer(input_texts, return_tensors='pt', padding=True)  # {(batch_size, tokens), ..}
            targets_batch = self.tokenizer(target_texts, return_tensors='pt',
                                           padding=True)  # {(batch_size, tokens), ..}

        if inputs_batch['input_ids'].shape[-1] > max_length or \
                targets_batch['input_ids'].shape[-1] > max_length:
            return None, None
        targets_batch['input_ids'][targets_batch['attention_mask'] == 0] = -100  # padding은 label 마스킹

        # print(inputs_batch)
        # print(targets_batch)
        inputs_batch.to(args.device)
        targets_batch.to(args.device)
        return inputs_batch, targets_batch

    def train_step(self, input_texts: List[str], target_texts: List[str]):
        # tech2
        if args.tech2:
            input_texts, target_texts = \
                tuple(zip(*[tech2(*tech2_args) for tech2_args in zip(input_texts, target_texts)]))
        if args.eda:
            input_texts = [eda(t) for t in input_texts]
        inputs_batch, targets_batch = self.encode(input_texts, target_texts)
        if inputs_batch is None:
            return None

        outputs = self.model(**inputs_batch, labels=targets_batch['input_ids'])

        self.optim.zero_grad()
        outputs.loss.backward()
        self.optim.step()
        self.scheduler.step()

        loss = float(str(float(outputs.loss.cpu().detach().numpy())))
        del outputs.loss, outputs
        del inputs_batch, targets_batch
        # torch.cuda.empty_cache()
        return loss

    def validate(self, input_texts: List[str], target_texts: List[str]):
        # validation 실시
        with torch.no_grad():
            inputs_batch, targets_batch = self.encode(input_texts, target_texts)
            if inputs_batch is None:
                return None

            outputs = self.model(**inputs_batch, labels=targets_batch['input_ids'])

        loss = float(str(float(outputs.loss.cpu().detach().numpy())))
        del outputs.loss, outputs
        return loss

    def generate(self, input_text: Union[List[str], str]) -> (Optional[Union[List[str], str]], List[str]):
        with torch.no_grad():
            drops: List[str] = []

            inputs = self.tokenizer(input_text, return_tensors='pt', padding=True)

            max_length = self.tokenizer.max_model_input_sizes[args.model_name]

            if isinstance(input_text, list):
                # 길이 max_length 초과하는 텍스트 삭제
                while inputs['input_ids'].shape[-1] > max_length and len(input_text) > 1:
                    idx = sorted(list(range(len(input_text))), key=lambda x: -len(input_text[x]))[0]  # 삭제할 인덱스
                    drops.append(input_text[idx])
                    del input_text[idx]
                    inputs = self.tokenizer(input_text, return_tensors='pt', padding=True)
            if inputs['input_ids'].shape[-1] > max_length:
                return None, drops

            inputs.to(args.device)

            tokens = self.model.generate(
                **inputs)
            if tokens.ndim == 2:
                return [self.tokenizer.decode(t).replace('<pad>', '').replace('<s>', '').replace('</s>', '').replace('<extra_id_31>', '') for t in tokens], drops
            else:
                return self.tokenizer.decode(tokens).replace('<pad>', '').replace('<s>', '').replace('</s>', '').replace('<extra_id_31>', ''), drops
