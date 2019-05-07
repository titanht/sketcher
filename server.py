#!/usr/bin/env python

from flask import Flask, url_for, send_from_directory, request
from flask_cors import CORS, cross_origin
import logging, os
from werkzeug import secure_filename
from flask import render_template

import argparse
import os
import scipy.misc
import numpy as np
import cv2

from model import pix2pix
import tensorflow as tf


app = Flask(__name__)
cors = CORS(app, resources={r"/": {"origins": "*", "Access-Control-Allow-Origin": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='facades', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')

args = parser.parse_args()

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
print(PROJECT_HOME)
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = None
sess = None
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
if not os.path.exists(args.sample_dir):
    os.makedirs(args.sample_dir)
if not os.path.exists(args.test_dir):
    os.makedirs(args.test_dir)

sess = tf.Session()
print("Building Model")
model = pix2pix(sess, image_size=args.fine_size, batch_size=args.batch_size,
                output_size=args.fine_size, dataset_name=args.dataset_name,
                checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir)
print("Model type 1: ", end='')
print(type(model))
print("Model built")

def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

@app.route('/', methods=['GET'])
def test():
    return render_template("index.html")

@app.route('/color', methods = ['POST', 'GET'])
# @cross_origin(origin='*',headers=['Content-Type','Authorization'])
def api_root():

    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.files['image']: 
        print("Model type 2: ", end='')
        print(type(model))
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']
        img_name = secure_filename(img.filename)

        print("Img: " + str(img_name))
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info("saving {}".format(saved_path))

        print("Try color")
        img.save(saved_path)

        img = cv2.imread(saved_path)
        img = cv2.resize(img, (256, 256))

        crow = img.shape[0]
        ccol = img.shape[1]

        conc = np.zeros(shape=(crow, ccol*2, 3), dtype=np.uint8)

        conc[:crow, :ccol] = img
        conc[:crow, ccol:] = img
        cv2.imwrite('static/imgs/sketch.jpg', conc)
        cv2.imwrite('static/imgs/file.jpg', img)

        # print('Name: ' + saved_path)

        model.colorize('static/imgs/sketch.jpg')
        # return send_from_directory('./','colored.png', as_attachment=True)
        return render_template('result.html', sketch_name='http://127.0.0.1:5000/static/imgs/file.jpg')
    else:
    	return "Where is the image?"

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == '__main__':

    app.run(debug=True, use_reloader=False)