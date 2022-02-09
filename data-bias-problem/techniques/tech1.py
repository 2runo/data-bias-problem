from typing import List
import nltk
import string
from nltk.corpus import stopwords

nltk_words = nltk.corpus.words.words()


class Tech1:
    @staticmethod
    def is_nnp(word: str) -> bool:
        if len(word) == 1:
            # 한 글자라면 -> 대문자 알파벳이어야 함
            return word.isupper()
        if word.islower():
            if word in nltk_words:
                return False
        return any([c in word for c in string.ascii_letters])  # 알파벳이 하나라도 들어가야 함

    @staticmethod
    def or_is_nnp(word: str) -> bool:
        return any([word.startswith(c) for c in string.ascii_letters]) \
               and any([word.endswith(c) for c in string.ascii_letters]) \
               and any([str(c) in word for c in range(10)]) \
               and all([c not in word for c in '!@#$%^&*()-=_+\'"'])  # ex: word2vec

    def get_nnps(self, text: str) -> List[str]:
        # 고유명사 목록 반환
        nnps: List[str] = []

        text = text.replace("'", '').replace('"', '').replace("^", '').replace("[", '').replace("]", '').replace("*", '')
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            # words = [word for word in words if word not in set(stopwords.words('english'))]
            tagged = nltk.pos_tag(words)
            for i, (word, tag) in enumerate(tagged):
                if tag == 'NNP' or self.or_is_nnp(word):
                    if self.is_nnp(word):
                        nnps.append(word)  # 고유명사
        return list(set(nnps))

    def __call__(self, *texts, mask: str = "<nnp{}>") -> List[str]:
        # 기법1) 고유명사 마스킹
        # ex) f("I have Mac, Apple's laptop.", "Mac is pretty.")
        # -> ("I have <nnp1>, <nnp0>'s laptop.", "<nnp1> is pretty.")
        if len(texts) == 1 and (isinstance(texts, list) or isinstance(texts, tuple)):
            texts = texts[0]
        if isinstance(texts, str):
            texts = [texts]
        nnps = []
        for text in texts:
            nnps.extend(self.get_nnps(text))  # 고유명사 목록
        nnps: List[str] = sorted(nnps, key=lambda x: -len(x))  # 길이 기준 정렬
        for i, nnp in enumerate(nnps):
            texts = [text.replace(nnp, mask.format(i)).replace(' <nnp', '<nnp') for text in texts]
        return texts


tech1 = Tech1()
