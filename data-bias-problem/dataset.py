from datasets import load_dataset
import torch.utils.data
import multiprocessing
from .arguments import args
from .techniques import tech1


class DataLoader:
    def __init__(self, batch_size: int, type_: str = 'train', shuffle: bool = False):
        if args.dataset_name == 'scitldr':
            self.dataset = load_dataset('scitldr', 'Abstract')
            self.data = list(zip([' '.join(i) for i in self.dataset[type_]['source']],
                                 [' '.join(i) for i in self.dataset[type_]['target']]))
        elif args.dataset_name == 'xsum':
            self.dataset = load_dataset('xsum')
            self.data = list(zip(self.dataset[type_]['document'],
                                 self.dataset[type_]['summary']))
        elif args.dataset_name == 'reddit_tifu':
            self.dataset = load_dataset('reddit_tifu', 'long')
            self.data = list(zip(self.dataset['train']['documents'],
                                 self.dataset['train']['tldr']))
            # 데이터셋 내에 train/val/test split 안 돼 있으므로 수동 split
            if type_ == 'train':
                self.data = self.data[:int(len(self.data)*0.7)]
            elif type_ == 'validation':
                self.data = self.data[int(len(self.data)*0.7):int(len(self.data)*0.85)]
            elif type_ == 'test':
                self.data = self.data[int(len(self.data)*0.85):]

        self.preprocessing()  # 전처리

        self.torch_dl = torch.utils.data.DataLoader(self.data, batch_size=batch_size, shuffle=shuffle)

    def preprocessing(self):
        # 전처리
        if args.tech1 and not args.test:
            processes = 50
            pool = multiprocessing.Pool(processes=processes)
            temp = []
            for i in range(0, len(self.data), processes):
                print(f'preprocessing..({i}/{len(self.data)})', end='\r')
                inputs = self.data[i:i+processes]
                temp.extend(pool.map(tech1, inputs))
            pool.close()
            self.data = temp

    def iterate(self):
        return self.torch_dl

    def __len__(self):
        return len(self.torch_dl)
