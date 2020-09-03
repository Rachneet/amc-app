import sys
from lightning_resnet import *
from resnet import *
import pandas as pd

import pprint

def complex_to_iq(complex):
    iq = []
    signal=[]
    for sample in complex:
        re = sample.real
        img = sample.imag
        iq.append(np.array([re,img]))
    return np.array(iq)

# print(sys.path)
def prediction(file):
    mods = ['SC BPSK', 'SC QPSK', 'SC 16-QAM', 'SC 64-QAM',
            'OFDM BPSK', 'OFDM QPSK', 'OFDM 16-QAM', 'OFDM 64-QAM']
    iqs, preds = [], []
    data = np.load(file)
    key = data.files[0]
    num_samples = data[key].shape[0]
    for i in range(num_samples):
        iq = data[key][i]
        iq = np.array(iq)
        iq  = complex_to_iq(iq)
        iq = np.squeeze(iq)
        iq = preprocessing.scale(iq, with_mean=False).astype(np.float32)
        iq = torch.Tensor(iq)
        iq = iq.unsqueeze(dim=0)
        model = LightningResnet.load_from_checkpoint('epoch=7.ckpt', map_location='cpu')
        # print(model.load_state_dict)
        model.eval()
        y_hat = model(iq)
        pred = torch.argmax(y_hat)
        pred = mods[pred]
        iqs.append(complex_to_iq(data[key][i]))
        preds.append(pred)

    df = pd.DataFrame(list(zip(iqs, preds)), columns=['I/Q', 'Modulation'])

    return df


if __name__ == "__main__":
    path = "/home/rachneet/rf_dataset_inets/vsg_no_intf_all_normed.h5"
    file = h5.File(path, 'r')
    iq = file['iq']
    sample = iq[1]
    # np.set_printoptions(threshold=sys.maxsize)
    # pprint.pprint(sample)
    predict = prediction(sample)
