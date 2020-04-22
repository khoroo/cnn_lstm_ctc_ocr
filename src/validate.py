# CNN-LSTM-CTC-OCR
# Copyright (C) 2017,2018 Jerod Weinman, Abyaya Lamsal, Benjamin Gafford
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# validate.py - Run model directly on from paths to image filenames. 
# NOTE: assumes mjsynth files are given by adding an extra row of padding

import sys
import numpy as np
from PIL import Image
import tensorflow as tf

import model_fn
import charset

import mjsynth
import pipeline
import data_tools

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string( 'model','../data/model',
                            """Directory for model checkpoints""" )
tf.app.flags.DEFINE_boolean( 'print_score', False,
                             """Print log probability scores with predictions""" )
tf.app.flags.DEFINE_string( 'lexicon','',
                """File containing lexicon of image words""" )
tf.app.flags.DEFINE_float( 'lexicon_prior',None,
                """Prior bias [0,1] for lexicon word""" )


tf.logging.set_verbosity( tf.logging.INFO )


def _process_padding(im, divby=32):
    def _add_margin(pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result
    
    width,height = im.size
    if ((width*height) % divby) == 0:
        return im
    top, right, bottom, left = 0, 0, 0, 0
    w_pad = divby - (width % divby) - 1
    h_pad = divby - (height % divby) - 1
    if w_pad < h_pad:
        if (w_pad % 2) == 0:
            right = left = int(w_pad/2)
        else:
            left = int(w_pad)
    else:
        if (h_pad % 2) == 0:
            top = bottom = int(h_pad/2)
        else:
            bottom = int(h_pad)
    result = _add_margin(im, top, right, bottom, left, (255,255,255))
    return result
    

def _get_image( filename ):
    """Load image data for placement in graph"""

    image = Image.open( filename )
    image = _process_padding(image)
    image = np.array( image )
    # in mjsynth, all three channels are the same in these grayscale-cum-RGB data
    image = image[:,:,:1] # so just extract first channel, preserving 3D shape
    return image

def _get_stdin():
    """Create a dataset of images by reading from stdin"""

    # Eliminate any trailing newline from filename
    image_data = _get_image( raw_input().rstrip() )

    # Initializing the dataset with one image
    dataset = tf.data.Dataset.from_tensors( image_data )

    # Add the rest of the images to the dataset (if any)
    for line in sys.stdin:
        image_data = _get_image( line.rstrip() )
        temp_dataset = tf.data.Dataset.from_tensors( image_data )
        dataset = dataset.concatenate( temp_dataset )
    return dataset

def _get_input():
    
    # dataset = _get_stdin()
    
    
    rects_file = '/content/output/Burnley/1909056140151.txt'
    image_file = '/content/ChimneyData/Sample/Raw/Burnley/1909056140151.tif'
    dataset = _get_prediction_dataset(image_file, rects_file)
    
    # mjsynth input images need preprocessing transformation (shape, range)
    dataset = dataset.map( mjsynth.preprocess_image )

    # pack results for model_fn.predict 
    dataset = dataset.map ( pipeline.pack_image )
    return dataset

def _get_prediction_dataset(image_file, rects_file):
    """
    Get a dataset from the lines in rects_file where each element is a
    tuple with the predicted rectangle (4x2 float32 tensor with points in CCW
    and bottom-left as the first point) and the cropped, rotated image (RGB uint8)
    corresponding to that rectangle as extracted from image_file.

    Parameters
      image_file: Path to the image to read rectangles from
      rects_file: Path to the file containing predicted rectangles
      **kwargs:   Optional arguments to data_tools.normalize_box
    Returns
      dataset:    tf.data.Dataset with (rectangle, image) tuples 
    """
    
    def gen_rect_image(): # Generator for yielding (rect,cropped) tuples
        image = np.array(Image.open(image_file).convert(mode='RGB'))
        rects = data_tools.parse_boxes_from_text(rects_file,slice_first=True)[0]

        for rect in rects:
            cropped = data_tools.normalize_box(image,rect)
            yield cropped

    # Construct the Dataset object from the generator
    dataset = tf.data.Dataset.from_generator (
        generator=gen_rect_image,
        output_types=(tf.uint8),
        output_shapes=(tf.TensorShape([None,None,1])))
    

    return dataset



def _get_config():
    """Setup session config to soften device placement"""

    device_config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False )

    custom_config = tf.estimator.RunConfig( session_config=device_config ) 

    return custom_config


def main(argv=None):
    
    classifier = tf.estimator.Estimator( config=_get_config(),
                                         model_fn=model_fn.predict_fn(
                                             FLAGS.lexicon,
                                             FLAGS.lexicon_prior), 
                                         model_dir=FLAGS.model )
    
    predictions = classifier.predict( input_fn=_get_input )
    
    # Get all the predictions in string format
    while True:
        try:
            results = next( predictions )
            print('results =',results)
            pred_str = charset.label_to_string( results['labels'] )
            if FLAGS.print_score:
                print(pred_str, results['score'][0])
            else:
                print(pred_str)
        except StopIteration:
            sys.exit()
    
if __name__ == '__main__':
    tf.app.run()
