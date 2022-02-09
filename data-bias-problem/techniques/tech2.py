from typing import List, Optional
from rouge_score import rouge_scorer
import numpy as np
import nltk
import string
from nltk.tokenize.treebank import TreebankWordDetokenizer

from .tfidf import TfIdf
from ..arguments import args


class Tech2:
    def __init__(self):
        self.tfidf = TfIdf()
        self.tfidf.load()
        self.scorer = rouge_scorer.RougeScorer(['rouge2'])
        self.detokenizer = TreebankWordDetokenizer()

    def deflate_words(self, text: str, n: int) -> str:
        # TF-IDF 낮은 단어 n개 삭제
        doc: List[str] = self.tfidf.tokenize(text)
        processed_doc: List[str] = [self.tfidf.preprocessing(i) for i in doc]
        tfidf_scores: List[float] = []
        for i, token in enumerate(processed_doc):
            tfidf_scores.append(self.tfidf(token, processed_doc))

        for _ in range(n):
            idx = int(np.argmin(tfidf_scores))
            del doc[idx]
            del tfidf_scores[idx]
        return self.detokenizer.detokenize(doc)

    def deflate_sentences(self, text: str, target: str, n: int) -> str:
        # rouge score 낮은 문장 n개 삭제
        sents: List[str] = nltk.sent_tokenize(text)

        rouge_scores: List[float] = []
        for sent in sents:
            rouge_scores.append(list(self.scorer.score(target, sent).values())[0].fmeasure)

        for _ in range(n):
            idx = int(np.argmin(rouge_scores))
            del sents[idx]
            del rouge_scores[idx]
        return ' '.join(sents)

    def inflate_words(self, text: str, n: int) -> str:
        # 무작위 단어 n개 골라서 무작위 위치에 삽입
        tokens: List[str] = nltk.word_tokenize(text)

        for _ in range(n):
            word = np.random.choice(tokens)
            cnt = 0
            while not any([c in word for c in string.ascii_letters]):
                word = np.random.choice(tokens)
                cnt += 1
                if cnt >= 100:
                    break
            tokens.insert(int(np.random.randint(0, len(tokens))), word)
        return self.detokenizer.detokenize(tokens)

    @staticmethod
    def inflate_sentences(text: str, n: int) -> str:
        # 문장 복제
        sents: List[str] = nltk.sent_tokenize(text)

        for _ in range(n):
            idx = int(np.random.randint(0, len(sents)))
            sents.insert(idx, sents[idx])
        return ' '.join(sents)

    def inflate(self, text: str, ratio: float):
        sents: List[str] = nltk.sent_tokenize(text)
        n = int(len(sents) * ratio) - len(sents)  # inflate_sentences
        if n > 0:
            text = self.inflate_sentences(text, n)
        words: List[str] = nltk.word_tokenize(text)
        n = int(len(words) * ratio) - len(words)  # inflate_words
        if n > 0:
            text = self.inflate_words(text, n)
        return text

    def deflate(self, text: str, ratio: float, target_text: Optional[str] = None):
        if target_text is not None:
            sents: List[str] = nltk.sent_tokenize(text)
            n = len(sents) - int(len(sents) * ratio)  # deflate_sentences
            if n > 0:
                text = self.deflate_sentences(text, target_text, n)
        words: List[str] = nltk.word_tokenize(text)
        n = len(words) - int(len(words) * ratio)  # deflate_words
        if n > 0:
            text = self.deflate_words(text, n)
        return text

    def __call__(self, input_text: str, target_text: str):
        rand: float = float(np.random.rand())
        if 0 <= rand <= args.tech2_pass_p:
            # 50% -> 그냥 pass
            pass
        elif args.tech2_pass_p < rand <= args.tech2_pass_p + (1-args.tech2_pass_p)/2:
            # 25% -> inflate
            ratio = float(np.random.uniform(1, 1.5))  # inflate 비율
            input_text = self.inflate(input_text, ratio)
            target_text = self.inflate(target_text, ratio)
        elif args.tech2_pass_p + (1-args.tech2_pass_p)/2 < rand <= 1:
            # 25% -> deflate
            ratio = float(np.random.uniform(0.5, 1))  # deflate 비율
            input_text = self.deflate(input_text, ratio, target_text=target_text)
            target_text = self.deflate(target_text, ratio)
        return input_text, target_text


tech2 = Tech2()
