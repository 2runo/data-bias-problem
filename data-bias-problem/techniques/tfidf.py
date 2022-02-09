from typing import List, Optional
from math import log
import multiprocessing
import nltk
import pickle

from ..utils.cache import Cache


class TfIdf:
    def __init__(self):
        self.df_cache: Cache = Cache()
        self.len_docs: int = 0

    @staticmethod
    def tf(token: str, doc: List[str]) -> int:
        # 특정 문서에서의 특정 단어의 등장 횟수
        return doc.count(token)

    @staticmethod
    def df(token: str, docs: List[List[str]]):
        # 특정 단어가 등장한 문서의 수
        df = 0
        for doc in docs:
            df += token in doc
        return df

    def idf(self, token: str, docs: Optional[List[List[str]]] = None, fitting: bool = False):
        if self.df_cache.include(token):
            # 캐시에 있으면? -> 그거 사용
            df = self.df_cache[token]
        else:
            if fitting:
                assert docs is not None
                # 학습 중이라면? -> 몇개인지 직접 count
                df = self.df(token, docs)
                self.df_cache[token] = df
            else:
                # 실전이라면? -> 처음 보는 단어로 간주 -> 0개
                df = 0
        if fitting:
            return None
        return log(self.len_docs / (df + 1))

    def tfidf(self, token: str, doc: List[str], docs: Optional[List[List[str]]] = None):
        return self.tf(token, doc) * self.idf(token, docs)

    @staticmethod
    def preprocessing(text: str) -> str:
        # 전처리
        return text.lower()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return nltk.word_tokenize(text)

    def fit(self, corpus: List[str], logging: bool = False):
        # 학습
        processes = 30
        pool = multiprocessing.Pool(processes=processes)
        corpus = pool.map(self.preprocessing, corpus)
        docs: List[List[str]] = pool.map(self.tokenize, corpus)

        self.len_docs = len(docs)
        for i, doc in enumerate(docs):
            for token in doc:
                self.idf(token, docs[i:], fitting=True)

            if logging:
                print(f"(fitting) df caching... {i}/{self.len_docs} {len(self.df_cache)}", end='\r')
        print()
        print(f"fitting finished. length of cache : {len(self.df_cache)}")

    def save(self, path: str = 'tfidf.pkl'):
        # df_cache 저장
        with open(path, 'wb') as f:
            pickle.dump((self.df_cache, self.len_docs), f)

    def load(self, path: str = 'tfidf.pkl'):
        # df_cache 불러오기
        with open(path, 'rb') as f:
            self.df_cache, self.len_docs = pickle.load(f)

    def __call__(self, *args):
        return self.tfidf(*args)


def fit_tfidf():
    raise NotImplementedError("말뭉치 이슈로 인해 TF-IDF 학습 기능은 제공하지 않습니다.")
    # import requests
    # corpus: List[str] = []

    # tfidf = TfIdf()
    # tfidf.fit(corpus, logging=True)
    # print('saving..', end='\r')
    # tfidf.save()
    # print('finished!')
