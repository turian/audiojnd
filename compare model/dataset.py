from torchopenl3.utils import preprocess_audio_batch

import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

import config


class AudioJNDDataset(Dataset):
    def __init__(self, files, y):
        self.files = files
        self.y = y

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fileA, fileB = self.files[idx]
        audioA, srA = sf.read(fileA)
        audioB, srB = sf.read(fileB)

        assert audioA.shape == audioB.shape
        assert srA == srB

        audioA = preprocess_audio_batch(torch.tensor(audioA).unsqueeze(0), srA).to(
            torch.float32
        )
        audioB = preprocess_audio_batch(torch.tensor(audioB).unsqueeze(0), srB).to(
            torch.float32
        )

        return {
            "audioA": audioA,
            "audioB": audioB,
            "similarity": torch.tensor(self.y[idx], dtype=torch.long),
            "sr": srA,
        }


def create_dataloaders(trainset, testset):
    trainfiles, train_y = trainset
    train_dset = AudioJNDDataset(trainfiles, train_y)
    train_dataloader = DataLoader(
        train_dset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    testfiles, test_y = testset
    test_dset = AudioJNDDataset(testfiles, test_y)
    test_dataloader = DataLoader(
        test_dset,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    return {"train": train_dataloader, "test": test_dataloader}
