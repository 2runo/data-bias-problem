import torch
from rouge_score import rouge_scorer

from .dataset import DataLoader
from .model import model, tokenizer
from .trainer import Trainer
from .arguments import args
from .techniques import tech1
from .techniques import tech2


# 모델 이름에 'tech1'이 들어있으면 args.tech1 == True여야함.
if args.tech1:
    assert 'tech1' in args.ckpt_path
else:
    assert 'tech1' not in args.ckpt_path
# 모델 이름에 'tech2'이 들어있으면 args.tech2 == True여야함.
if args.tech2:
    assert 'tech2' in args.ckpt_path
else:
    assert 'tech2' not in args.ckpt_path
# 모델 이름에 'eda'가 들어있으면 args.eda == True여야함.
if args.eda:
    assert 'eda' in args.ckpt_path
else:
    assert 'eda' not in args.ckpt_path


if args.tech1:
    # tokenizer에 special token 추가
    tokenizer.add_tokens(['<nnp{}>'.format(i) for i in range(16)], special_tokens=True)


test_dataloader = DataLoader(args.val_batch_size, 'test', shuffle=False)
trainer = Trainer(model, tokenizer, len(test_dataloader))
trainer.model.load_state_dict(torch.load(args.ckpt_path))
trainer.model.eval()


scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

rouge1_logs = []
rouge2_logs = []
rougeL_logs = []
generated_n_words = []  # 생성한 요약문 단어 개수 목록
target_n_words = []  # 정답 요약문 단어 개수 목록
skip_history = []
first_print = True
for i, (input_texts, target_texts) in enumerate(test_dataloader.iterate()):
    if i > 1000:
        break

    input_texts = list(input_texts)
    target_texts = list(target_texts)

    generated_summaries, drops = trainer.generate(input_texts.copy())
    if generated_summaries is None:
        skip_history.append(i)
        continue

    for drop in drops:
        idx = input_texts.index(drop)
        del input_texts[idx]
        del target_texts[idx]

    assert len(generated_summaries) == len(target_texts)

    generated_n_words.extend([sent.count(' ') + 1 for sent in generated_summaries])
    target_n_words.extend([sent.count(' ') + 1 for sent in target_texts])

    for j in range(len(generated_summaries)):
        scores = [i.fmeasure for i in scorer.score(target_texts[j], generated_summaries[j]).values()]
        rouge1_logs.append(scores[0])
        rouge2_logs.append(scores[1])
        rougeL_logs.append(scores[2])
    if first_print:
        print(args.ckpt_path, '\t', args.dataset_name)
        first_print = False
    print(f"{i} {sum(rouge1_logs) / len(rouge1_logs)} {sum(rouge2_logs) / len(rouge2_logs)} {sum(rougeL_logs) / len(rougeL_logs)} {sum(generated_n_words) / len(generated_n_words)} {sum(target_n_words) / len(target_n_words)}", end='\r')


print()
print()
print(skip_history)
