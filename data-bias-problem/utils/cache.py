from typing import Optional, Dict, Union, List


class Cache:
    def __init__(self, max_len: Optional[int] = None):
        self.max_len: Optional[int] = max_len
        self.key_history: List[str] = []  # 캐시에 등록했던 key 기록
        self.cache: Dict[str, Union] = {}

    def __setitem__(self, key: str, value):
        # 추가 or 값 설정
        assert isinstance(key, str)
        assert len(key) > 0
        head = key[0]
        tail = key[1:]
        if head not in self.cache:
            self.cache[head] = {}
        self.cache[head][tail] = value
        if self.max_len is not None:
            self.key_history.append(key)
            self.check_len()

    def __getitem__(self, key: str):
        # 값 반환
        head = key[0]
        tail = key[1:]
        return self.cache[head][tail]

    def include(self, key: str) -> bool:
        assert isinstance(key, str)
        assert len(key) > 0
        head = key[0]
        tail = key[1:]
        return head in self.cache and tail in self.cache[head]

    def delete(self, key: str):
        # 삭제
        assert isinstance(key, str)
        assert len(key) > 0
        head = key[0]
        tail = key[1:]
        del self.cache[head][tail]

    def __len__(self):
        # 길이 반환
        length: int = 0
        for v in self.cache.values():
            length += len(v)
        return length

    def check_len(self):
        # 최대 길이를 넘겼는지 확인 & 넘겼으면 삭제
        while len(self) > self.max_len:
            self.delete(self.key_history[0])
            del self.key_history[0]
