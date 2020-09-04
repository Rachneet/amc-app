import sys
from lightning_resnet import *
from resnet import *
import pandas as pd
from sklearn import preprocessing


import pprint

def complex_to_iq(complex):
    iq = []
    signal=[]
    for sample in complex:
        re = sample.real
        img = sample.imag
        iq.append(np.array([re,img]))
    return np.array(iq)


class DatasetFromNpz(Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.file = np.load(self.filename)
        self.key = self.file.files[0]

    def __len__(self):
        lens = len(self.file[self.key])
        return lens

    def __getitem__(self, item):
        
        data = self.file[self.key][item]

        data = complex_to_iq(data)
        data = preprocessing.scale(data,with_mean=False).astype(np.float32)
        data = data.astype(np.float32)

        return data

# print(sys.path)
def prediction(file):
    mods = ['SC BPSK', 'SC QPSK', 'SC 16-QAM', 'SC 64-QAM',
            'OFDM BPSK', 'OFDM QPSK', 'OFDM 16-QAM', 'OFDM 64-QAM']
    iqs, preds = [], []
    data = DatasetFromNpz(file)

    test_set = DataLoader(data, batch_size=512,
                shuffle=False,num_workers=1)
    model = LightningResnet.load_from_checkpoint('epoch=7.ckpt', map_location='cpu')
    model.eval()

    for batch in test_set:
        y_hat = model(batch)
        batch_pred = [mods[torch.argmax(val).numpy()] for val in y_hat]
        preds.extend(batch_pred)
        iqs.extend(batch.numpy())

    df = pd.DataFrame(list(zip(iqs, preds)), columns=['I/Q', 'Modulation'])
    return df


if __name__ == "__main__":
    # path = "/home/rachneet/rf_dataset_inets/vsg_no_intf_all_normed.h5"
    # file = h5.File(path, 'r')
    # iq = file['iq']
    # sample = iq[1]
    # # np.set_printoptions(threshold=sys.maxsize)
    # # pprint.pprint(sample)
    # predict = prediction(sample)
    pass
