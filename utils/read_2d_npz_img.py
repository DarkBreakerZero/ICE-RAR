import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
class npz_reader_2d(torch.utils.data.Dataset):
    def __init__(self, paired_data_txt):
        super(npz_reader_2d, self).__init__()
        self.paired_files = open(paired_data_txt).readlines()

    def __len__(self):
        return len(self.paired_files)

    def to_tensor(self, data):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            # data = data.transpose(2, 0, 1)
            # data = data[np.newaxis, ...]
            pass

        data = torch.FloatTensor(data)

        return data

    def to_numpy(self, data):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if data.ndim == 3:
            data = data.transpose(1, 2, 0)

        return data
    
    def get(self, paired_files):

        npzData = np.load(paired_files[:-1])

        rdctData = npzData['label'] * 100
        ldctData = npzData['input'] * 100
        rdctData = np.array(rdctData).astype(np.float32)
        ldctData = np.array(ldctData).astype(np.float32)
        rdctData = self.to_tensor(rdctData)
        ldctData = self.to_tensor(ldctData)

        return {"rdct": rdctData, "ldct": ldctData}

    def __getitem__(self, index):
        paired_files = self.paired_files[index]
        return self.get(paired_files)


class npz_reader_2d_polar(torch.utils.data.Dataset):
    def __init__(self, paired_data_txt):
        super(npz_reader_2d_polar, self).__init__()
        self.paired_files = open(paired_data_txt).readlines()

    def __len__(self):
        return len(self.paired_files)

    def to_tensor(self, data):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            # data = data.transpose(2, 0, 1)
            # data = data[np.newaxis, ...]
            pass

        data = torch.FloatTensor(data)

        return data

    def to_numpy(self, data):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if data.ndim == 3:
            data = data.transpose(1, 2, 0)

        return data

    def get(self, paired_files):

        npzData = np.load(paired_files[:-1])

        rdctData = npzData['label'] * 50
        ldctData = npzData['input'] * 50
        rdctData = np.array(rdctData).astype(np.float32)
        ldctData = np.array(ldctData).astype(np.float32)
        rdctData = self.to_tensor(rdctData)
        ldctData = self.to_tensor(ldctData)

        return {"rdct": rdctData, "ldct": ldctData}

    def __getitem__(self, index):
        paired_files = self.paired_files[index]
        return self.get(paired_files)
    
class npz_reader_2d_proj(torch.utils.data.Dataset):
    def __init__(self, paired_data_txt):
        super(npz_reader_2d_proj, self).__init__()
        self.paired_files = open(paired_data_txt).readlines()

    def __len__(self):
        return len(self.paired_files)

    def to_tensor(self, data):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            # data = data.transpose(2, 0, 1)
            # data = data[np.newaxis, ...]
            pass

        data = torch.FloatTensor(data)

        return data

    def to_numpy(self, data):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if data.ndim == 3:
            data = data.transpose(1, 2, 0)

        return data

    def get(self, paired_files):

        npzData = np.load(paired_files[:-1])

        rdctData = npzData['label'] * 10
        ldctData = npzData['input'] * 10
        rdctData = np.array(rdctData).astype(np.float32)
        ldctData = np.array(ldctData).astype(np.float32)
        rdctData = self.to_tensor(rdctData)
        ldctData = self.to_tensor(ldctData)

        return {"rdct": rdctData, "ldct": ldctData}

    def __getitem__(self, index):
        paired_files = self.paired_files[index]
        return self.get(paired_files)


class npz_reader_2d_CNN(torch.utils.data.Dataset):
    def __init__(self, paired_data_txt):
        super(npz_reader_2d_CNN, self).__init__()
        self.paired_files = open(paired_data_txt).readlines()

    def __len__(self):
        return len(self.paired_files)

    def to_tensor(self, data):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            # data = data.transpose(2, 0, 1)
            # data = data[np.newaxis, ...]
            pass

        data = torch.FloatTensor(data)

        return data

    def to_numpy(self, data):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if data.ndim == 3:
            data = data.transpose(1, 2, 0)

        return data

    def get(self, paired_files):

        npzData = np.load(paired_files[:-1])

        rdctData = npzData['label'] * 100
        ldctData = npzData['input'] * 100
        wfData = npzData['input_WF'] * 100
        rdctData = np.array(rdctData).astype(np.float32)
        ldctData = np.array(ldctData).astype(np.float32)
        wfData = np.array(wfData).astype(np.float32)

        ldctData = ldctData[np.newaxis, :, :]
        wfData = wfData[np.newaxis, :, :]

        mixData = np.concatenate([ldctData, wfData], axis=0)  # 2x512x512

        rdctData = self.to_tensor(rdctData)
        mixData = self.to_tensor(mixData)


        return {"rdct": rdctData, "ldct": mixData}

    def __getitem__(self, index):
        paired_files = self.paired_files[index]
        return self.get(paired_files)

class npz_reader_2d_pi(torch.utils.data.Dataset):
    def __init__(self, paired_data_txt):
        super(npz_reader_2d_pi, self).__init__()
        self.paired_files = open(paired_data_txt).readlines()

    def __len__(self):
        return len(self.paired_files)

    def to_tensor(self, data):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            data = data.transpose(2, 0, 1)

        data = torch.FloatTensor(data)

        return data

    def to_numpy(self, data):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if data.ndim == 3:
            data = data.transpose(1, 2, 0)

        return data
    
    def get(self, paired_files):

        npzData = np.load(paired_files[:-1])

        rdctData = npzData['label'] * 100
        ldctData = npzData['input'] * 100
        pictData = npzData['prior'] * 100
        rdctData = np.array(rdctData).astype(np.float32)
        ldctData = np.array(ldctData).astype(np.float32)
        pictData = np.array(pictData).astype(np.float32)
        rdctData = self.to_tensor(rdctData)
        ldctData = self.to_tensor(ldctData)
        pictData = self.to_tensor(pictData)

        return {"label": rdctData, 'prior': pictData, "input": ldctData}

    def __getitem__(self, index):
        paired_files = self.paired_files[index]
        return self.get(paired_files)
    

# class npz_reader_2d_sino(torch.utils.data.Dataset):
    


if __name__ =='__main__':
    train_dataset = npz_reader_2d_CNN(paired_data_txt='/mnt/no1/liuguannan/RingExp/txt/train_img_CNN.txt')
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=64, shuffle=True)

    valid_dataset = npz_reader_2d_CNN(paired_data_txt='/mnt/no1/liuguannan/RingExp/txt/valid_img_CNN.txt')
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=8, shuffle=True)

    for (index, data) in enumerate(train_loader):
        print(data['ldct'].shape)
        print(data['rdct'].shape)