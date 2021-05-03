import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sentences1, sentences2, encodings, labels):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).unsqueeze(0)
        item['sentence1'] = self.sentences1[idx]
        item['sentence2'] = self.sentences2[idx]

        return item

    def __len__(self):
        return len(self.labels)