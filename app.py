from flask import Flask, request, jsonify, session, Response
from flask_cors import CORS
import os
import numpy as np
from predict import prediction
import time as time

#csv.field_size_limit(sys.maxsize)

# -------------------------------------flask backend-----------------------------------------------

flask_app = Flask(__name__, static_folder="build/", static_url_path="/")
CORS(flask_app)
flask_app.debug = 'DEBUG' in os.environ

UPLOAD_FOLDER = '/home/rachneet/PycharmProjects/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

flask_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@flask_app.route('/', methods=["GET"])
def index():
    return flask_app.send_static_file('index.html')

@flask_app.route('/favicon.ico', methods=["GET"])
def favicon():
    return flask_app.send_static_file('favicon.ico')


@flask_app.route("/api/upload", methods=["POST"])
def make_prediction():
    file = request.files['file']
    df = prediction(file)
    # print(df.head())
    # target = os.path.join(UPLOAD_FOLDER, 'test_docs')
    # if not os.path.isdir(target):
    #     os.mkdir(target)
    #
    # file = request.files['file']
    # print(file)
    # filename = secure_filename(file.filename)
    # destination = "/".join([target, filename])
    # file.save(destination)
    # session['uploadFilePath'] = destination
    response = Response(df.to_json(orient="records"), mimetype='application/json')
    print(response)
    return response


# -----------------------------------------------------MAIN------------------------------------------------------

def iq_to_complex(iq_sample):
    complex = []
    for sample in iq_sample:
        re = sample[0]
        img = sample[1]
        signal = np.complex(re + 1j * img)
        complex.append(signal)
    return complex
#
#
# def complex_to_iq(complex):
#     iq = []
#     signal=[]
#     for sample in complex:
#         re = sample.real
#         img = sample.imag
#         iq.append(np.array([re,img]))
#     return np.array(iq)


if __name__ == "__main__":

    if (os.environ.get('PORT')):
        port = int(os.environ.get('PORT'))
        host = "0.0.0.0"
    else:
        host = "127.0.0.1"
        port = 5000
    flask_app.run(host=host, debug=True, port=port)


    # import h5py as h5
    # import numpy as np
    # # from numpy import asarray, savez
    # import pandas as pd
    #
    # file = h5.File("/home/rachneet/rf_dataset_inets/vsg_no_intf_all_normed.h5", "r")
    # iq = file['iq']
    # num_samples = 2
    # num_iq = 1024
    # matrix = np.zeros((num_samples, num_iq), dtype=np.complex64)
    # # print(matrix.shape)
    # # print(iq[0].shape)
    #
    # for i in range(num_samples):
    #     matrix[i,:] = iq_to_complex(iq[i])
    #
    # print(matrix)
    # print(matrix.shape)
    #
    # np.savez("/home/rachneet/rf_dataset_inets/test1.npz", matrix=matrix)
    # path = "/home/rachneet/rf_dataset_inets/test.npz"
    # # data = np.load("/home/rachneet/rf_dataset_inets/test.npz")
    # start = time.time()
    # prediction(path)
    # stop = time.time()
    # print(stop-start)
    # print(df.head())
    # with np.load(path) as file:
    #     print(len(file['matrix']))


