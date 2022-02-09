# Training Techniques for Data Bias Problem on Deep Learning Text Summarization

This is the official implementation of the paper *Training Techniques for Data Bias Problem on Deep Learning Text Summarization*.



## 코드 설명

#### 주요 코드

- `techniques/tech1.py` : **고유명사 마스킹 기법**을 구현한 코드입니다. 
- `techniques/tech2.py` : **길이 조절 기법**을 구현한 코드입니다. 

#### 기타 코드

- `__main__.py` : 학습, 테스트 등을 진행하는 코드입니다.
- `arguments.py` : 학습 관련 파라미터가 지정돼 있는 코드입니다.
- `dataset.py` : 데이터셋을 불러오고 전처리하는 코드입니다. ([Datasets](https://github.com/huggingface/datasets) 라이브러리 사용)
- `model.py` : 언어 모델을 정의하는 코드입니다. ([Transformers](https://github.com/huggingface/transformers) 라이브러리 사용)
- `test.py` : 학습한 언어 모델을 테스트하는 코드입니다.
- `trainer.py` : 학습 관련 함수들을 정의하는 코드입니다. (train step, validate 등)
- `utils/cache.py` : 속도 향상을 위한 자체 제작 캐시 모듈입니다.
- `techniques/eda.py` : [EDA](https://arxiv.org/pdf/1901.11196.pdf) 논문 구현 코드입니다. (성능 비교를 위해 만들어 두었습니다)
- `techniques/tfidf.py` : TF-IDF 알고리즘을 구현한 코드입니다.



## 참고

본 레포지토리는 아카이브 목적으로 제작되었습니다. 코드가 실제로 잘 동작하지 않을 수 있습니다.

체크포인트는 파일은 크기 문제로 인해 제공하지 않습니다.

