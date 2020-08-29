import sys
from lightning_resnet import *
from resnet import *

import pprint

# print(sys.path)
def prediction(iq):
    mods = ['SC BPSK', 'SC QPSK', 'SC 16-QAM', 'SC 64-QAM',
            'OFDM BPSK', 'OFDM QPSK', 'OFDM 16-QAM', 'OFDM 64-QAM']
    # norm if needed
    iq = np.array(iq)
    iq = np.squeeze(iq)
    iq = preprocessing.scale(iq, with_mean=False).astype(np.float32)
    iq = torch.Tensor(iq)
    iq = iq.unsqueeze(dim=0)
    model = LightningResnet.load_from_checkpoint('epoch=7.ckpt')
    # print(model.load_state_dict)
    model.eval()
    y_hat = model(iq)
    pred = torch.argmax(y_hat)
    pred = mods[pred]
    print(pred)
    return pred


if __name__ == "__main__":
    path = "/home/rachneet/rf_dataset_inets/vsg_no_intf_all_normed.h5"
    file = h5.File(path, 'r')
    iq = file['iq']
    sample = iq[1]
    # np.set_printoptions(threshold=sys.maxsize)
    # pprint.pprint(sample)
    predict = prediction(sample)
