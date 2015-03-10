__author__ = 'rahul chandran'

import theano
from theano import tensor as T
from theano.tensor.nnet import conv

import numpy as np
import pylab
from PIL import Image

rng = np.random.RandomState(23444)

# Initialize 4D tensor for input

input = T.tensor4(name ='input')

#initialize shared variables for weight

w_shp = 2,3,9,9
w_bound = np.sqrt(3*9*9)
W  = theano.shared(np.asarray(rng.uniform(low = -1/w_bound,high= 1/w_bound,size=w_shp),dtype=input.dtype),name = 'W')

# initialize shared variables for bias
b_shp = 2,
b = theano.shared(np.asarray(rng.uniform(low=-.5,high = .5,size=b_shp),dtype=input.dtype),name='b')

# build symbolic expression that computes the convolution of input with filters in w
conv_out = conv.conv2d(input, W)

output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
# create theano function to compute filtered images
f = theano.function([input], output)

img = Image.open(open('/home/rahul/undir.jpg'))
print img.size
img = np.asarray(img, dtype='float64') / 256

# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)
filtered_img = f(img_)

# plot original image and first and second components of output
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray()

# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()




