from re import T
from keras import layers
import tensorflow as tf
import tflearn
from keras.layers import Conv3D, Activation, UpSampling3D,BatchNormalization,Conv3DTranspose,Add,Concatenate,Dropout
from tflearn.initializations import normal
from .spatial_transformer import Dense3DSpatialTransformer
from .utils import Network, ReLU, LeakyReLU,Softmax,Sigmoid
#from .IN import InstanceNormalization
from tensorflow.contrib.layers import instance_norm
import keras.backend as K
import keras

#from keras.layers.core import Lambda

def convolve(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False, weights_init='uniform_scaling'):
    return tflearn.layers.conv_3d(inputLayer, outputChannel, kernelSize, strides=stride,
                                  padding='same', activation='linear', bias=True, scope=opName, reuse=reuse, weights_init=weights_init)

def leakyReLU(inputLayer,opName, alpha = 0.1):
    return LeakyReLU(inputLayer,alpha,opName+'_leakilyrectified')

def inLeakyReLU(inputLayer,opName,alpha = 0.1):
    #IN = InstanceNormalization()(inputLayer)
    IN = instance_norm(inputLayer, scope=opName+'_IN')
    return LeakyReLU(IN,alpha,opName+'_leakilyrectified')

def convInLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, alpha=0.1, stddev=1e-2, reuse=False):
    conv = convolve(opName, inputLayer,outputChannel, kernelSize, stride, stddev, reuse)
    #conv_In = InstanceNormalization()(conv)
    conv_In = instance_norm(conv, scope=opName+'_IN')
    return LeakyReLU(conv_In,alpha,opName+'_leakilyrectified')

def convolveReLU(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False):
    return ReLU(convolve(opName, inputLayer,
                         outputChannel,
                         kernelSize, stride, stddev=stddev, reuse=reuse),
                opName+'_rectified')

def convolveSoftmax(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False):
    return Softmax(convolve(opName, inputLayer,
                         outputChannel,
                         kernelSize, stride, stddev=stddev, reuse=reuse),
                opName+'_softmax')

def convolveSigmoid(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False):
    return Sigmoid(convolve(opName, inputLayer,
                         outputChannel,
                         kernelSize, stride, stddev=stddev, reuse=reuse),
                opName+'_sigmoid')

def convolveLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, alpha=0.1, stddev=1e-2, reuse=False):
    return LeakyReLU(convolve(opName, inputLayer,
                              outputChannel,
                              kernelSize, stride, stddev, reuse),
                     alpha, opName+'_leakilyrectified')

def upconvolve(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, stddev=1e-2, reuse=False, weights_init='uniform_scaling'):
    return tflearn.layers.conv.conv_3d_transpose(inputLayer, outputChannel, kernelSize, targetShape, strides=stride,
                                                 padding='same', activation='linear', bias=False, scope=opName, reuse=reuse, weights_init=weights_init)

def upconvolveLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, alpha=0.1, stddev=1e-2, reuse=False):
    return LeakyReLU(upconvolve(opName, inputLayer,
                                outputChannel,
                                kernelSize, stride,
                                targetShape, stddev, reuse),
                     alpha, opName+'_rectified')

def upconvolveInLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, alpha=0.1, stddev=1e-2, reuse=False):
    up_in = instance_norm(upconvolve(opName, inputLayer,outputChannel,kernelSize, stride,targetShape, stddev, reuse), scope=opName+'_IN')
    return LeakyReLU(up_in,alpha, opName+'_rectified')


    
class DUAL(Network):
    def __init__(self, name, flow_multiplier=1., channels=8, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels
        self.reconstruction = Dense3DSpatialTransformer()

    def build(self,img1, img2 ):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        #T1 Encoder
        dims = 3
        c = self.channels
        def resblock(inputLayer,opName,channel):
            residual = inputLayer
            conv1_1 = convolve(opName+'_1',inputLayer, channel,   3, 1)
            conv1_2 = convolve(opName+'_2',conv1_1, channel,   3, 1)
            add1 = Add()([conv1_2, residual])
            return add1

        conv0_fixed = convolve('conv0_fixed',   img1, c,   3, 1)
        conv0_fixed = BatchNormalization()(conv0_fixed)
        conv0_fixed = Activation('relu')(conv0_fixed)

        conv1_fixed = convolve('conv1_fixed',   conv0_fixed, c,   3, 2)#80
        conv1_fixed = BatchNormalization()(conv1_fixed)
        conv1_fixed = Activation('relu')(conv1_fixed)
        conv1_fixed = convolve('conv1_fixed_',   conv1_fixed, c*2,   3, 1)  
        conv1_fixed = resblock(conv1_fixed,'conv1_fixed_1',c*2)
        conv1_fixed = resblock(conv1_fixed,'conv1_fixed_2',c*2)
        
        conv2_fixed = convolve('conv2_fixed',   conv1_fixed, c*2,   3, 2)#40
        conv2_fixed = BatchNormalization()(conv2_fixed)
        conv2_fixed = Activation('relu')(conv2_fixed)
        conv2_fixed = convolve('conv2_fixed_',   conv2_fixed, c*4,   3, 1) 
        conv2_fixed = resblock(conv2_fixed,'conv2_fixed_1',c*4)
        conv2_fixed = resblock(conv2_fixed,'conv2_fixed_2',c*4)
        
        conv3_fixed = convolve('conv3_fixed',   conv2_fixed, c*4,   3, 2)#20
        conv3_fixed = BatchNormalization()(conv3_fixed)
        conv3_fixed = Activation('relu')(conv3_fixed)
        conv3_fixed = convolve('conv3_fixed_',   conv3_fixed, c*4,   3, 1) 
        conv3_fixed = resblock(conv3_fixed,'conv3_fixed_1',c*4)
        conv3_fixed = resblock(conv3_fixed,'conv3_fixed_2',c*4)

        conv4_fixed = convolve('conv4_fixed',   conv3_fixed, c*4,   3, 2)#10

        #T2 Encoder
        conv0_float = convolve('conv0_float',  img2, c,   3, 1)
        conv0_float = BatchNormalization()(conv0_float)
        conv0_float = Activation('relu')(conv0_float)

        conv1_float = convolve('conv1_float',   conv0_float, c,   3, 2)#80
        conv1_float = BatchNormalization()(conv1_float)
        conv1_float = Activation('relu')(conv1_float)
        conv1_float = convolve('conv1_float_',   conv1_float, c*2,   3, 1)  
        conv1_float = resblock(conv1_float,'conv1_float_1',c*2)
        conv1_float = resblock(conv1_float,'conv1_float_2',c*2)
        
        conv2_float = convolve('conv2_float',   conv1_float, c*2,   3, 2)#40
        conv2_float = BatchNormalization()(conv2_float)
        conv2_float = Activation('relu')(conv2_float)
        conv2_float = convolve('conv2_float_',   conv2_float, c*4,   3, 1) 
        conv2_float = resblock(conv2_float,'conv2_float_1',c*4)
        conv2_float = resblock(conv2_float,'conv2_float_2',c*4)
        
        conv3_float = convolve('conv3_float',   conv2_float, c*4,   3, 2)#20
        conv3_float = BatchNormalization()(conv3_float)
        conv3_float = Activation('relu')(conv3_float)
        conv3_float = convolve('conv3_float_',   conv3_float, c*4,   3, 1) 
        conv3_float = resblock(conv3_float,'conv3_float_1',c*4)
        conv3_float = resblock(conv3_float,'conv3_float_2',c*4)

        conv4_float = convolve('conv4_float',   conv3_float, c*4,   3, 2)#10

        concat_bottleNeck = tf.concat([conv4_fixed,conv4_float],4,'concat_bottleNeck')

        #   warping scale 3   
        pred4 = convolve('pred4', concat_bottleNeck, dims, 3, 1)
        warping_field_3 = UpSampling3D()(pred4)

        conv3_float_up = UpSampling3D()(conv4_float)
        conv3_fixed_up = UpSampling3D()(conv4_fixed)
        conv3_float_up = convolveLeakyReLU('decode3_conv1', conv3_float_up, c*4, 1, 1,reuse = None)
        conv3_fixed_up = convolveLeakyReLU('decode3_conv1', conv3_fixed_up, c*4, 1, 1,reuse = True)

        concat3_float = tf.concat([conv3_float,conv3_float_up], 4, 'concat3_float')
        concat3_fixed = tf.concat([conv3_fixed,conv3_fixed_up], 4, 'concat3_fixed')

        deconv3_float = convolveLeakyReLU('decode3', concat3_float, c*4, 3, 1,reuse = None)
        deconv3_fixed = convolveLeakyReLU('decode3', concat3_fixed, c*4, 3, 1,reuse = True)

        conv3_float_rc = self.reconstruction([deconv3_float,warping_field_3])
        concat_3_rc = tf.concat([conv3_float_rc,deconv3_fixed], 4, 'concat_3_rc')
        
        #   warping scale 2   
        pred3 = convolve('pred3', concat_3_rc, dims, 3, 1)
        warping_field_2 = UpSampling3D()(pred3)

        conv2_float_up = UpSampling3D()(conv3_float)
        conv2_fixed_up = UpSampling3D()(conv3_fixed)
        conv2_float_up = convolveLeakyReLU('decode2_conv1', conv2_float_up, c*4, 1, 1,reuse = None)
        conv2_fixed_up = convolveLeakyReLU('decode2_conv1', conv2_fixed_up, c*4, 1, 1,reuse = True)

        concat2_float = tf.concat([conv2_float,conv2_float_up], 4, 'concat2_float')
        concat2_fixed = tf.concat([conv2_fixed,conv2_fixed_up], 4, 'concat2_fixed')

        deconv2_float = convolveLeakyReLU('decode2', concat2_float, c*4, 3, 1,reuse = None)
        deconv2_fixed = convolveLeakyReLU('decode2', concat2_fixed, c*4, 3, 1,reuse = True)

        conv2_float_rc = self.reconstruction([deconv2_float,warping_field_2])
        concat_2_rc = tf.concat([conv2_float_rc,deconv2_fixed], 4, 'concat_2_rc')

        #   warping scale 1   
        pred2 = convolve('pred2', concat_2_rc, dims, 3, 1)
        warping_field_1 = UpSampling3D()(pred2)

        conv1_float_up = UpSampling3D()(conv2_float)
        conv1_fixed_up = UpSampling3D()(conv2_fixed)
        conv1_float_up = convolveLeakyReLU('decode1_conv1', conv1_float_up, c*2, 1, 1,reuse = None)
        conv1_fixed_up = convolveLeakyReLU('decode1_conv1', conv1_fixed_up, c*2, 1, 1,reuse = True)

        concat1_float = tf.concat([conv1_float,conv1_float_up], 4, 'concat1_float')
        concat1_fixed = tf.concat([conv1_fixed,conv1_fixed_up], 4, 'concat1_fixed')

        deconv1_float = convolveLeakyReLU('decode1', concat1_float, c*2, 3, 1,reuse = None)
        deconv1_fixed = convolveLeakyReLU('decode1', concat1_fixed, c*2, 3, 1,reuse = True)

        conv1_float_rc = self.reconstruction([deconv1_float,warping_field_1])
        concat_1_rc = tf.concat([conv1_float_rc,deconv1_fixed], 4, 'concat_1_rc')

        #   warping scale 0  
        pred1 = convolve('pred1', concat_1_rc, dims, 3, 1)
        warping_field_0 = UpSampling3D()(pred1)

        conv0_float_up = UpSampling3D()(conv1_float)
        conv0_fixed_up = UpSampling3D()(conv1_fixed)
        conv0_float_up = convolveLeakyReLU('decode0_conv1', conv0_float_up, c, 1, 1,reuse = None)
        conv0_fixed_up = convolveLeakyReLU('decode0_conv1', conv0_fixed_up, c, 1, 1,reuse = True)

        concat0_float = tf.concat([conv0_float,conv0_float_up], 4, 'concat0_float')
        concat0_fixed = tf.concat([conv0_fixed,conv0_fixed_up], 4, 'concat0_fixed')

        deconv0_float = convolveLeakyReLU('decode0', concat0_float, c, 3, 1,reuse = None)
        deconv0_fixed = convolveLeakyReLU('decode0', concat0_fixed, c, 3, 1,reuse = True)

        conv0_float_rc = self.reconstruction([deconv0_float,warping_field_0])
        concat_0_rc = tf.concat([conv0_float_rc,deconv0_fixed], 4, 'concat_0_rc')

        pred0 = convolve('pred0', concat_0_rc, dims, 3, 1)

        progress_3 = self.reconstruction([UpSampling3D()(pred4),pred3])+pred3
        progress_2 = self.reconstruction([UpSampling3D()(progress_3),pred2])+pred2
        progress_1 = self.reconstruction([UpSampling3D()(progress_2),pred1])+pred1
        progress_0 = self.reconstruction([UpSampling3D()(progress_1),pred0])+pred0
        
        return {'flow': progress_0}

def affine_flow(W, b, len1, len2, len3):
    b = tf.reshape(b, [-1, 1, 1, 1, 3])
    xr = tf.range(-(len1 - 1) / 2.0, len1 / 2.0, 1.0, tf.float32)
    xr = tf.reshape(xr, [1, -1, 1, 1, 1])
    yr = tf.range(-(len2 - 1) / 2.0, len2 / 2.0, 1.0, tf.float32)
    yr = tf.reshape(yr, [1, 1, -1, 1, 1])
    zr = tf.range(-(len3 - 1) / 2.0, len3 / 2.0, 1.0, tf.float32)
    zr = tf.reshape(zr, [1, 1, 1, -1, 1])
    wx = W[:, :, 0]
    wx = tf.reshape(wx, [-1, 1, 1, 1, 3])
    wy = W[:, :, 1]
    wy = tf.reshape(wy, [-1, 1, 1, 1, 3])
    wz = W[:, :, 2]
    wz = tf.reshape(wz, [-1, 1, 1, 1, 3])
    return (xr * wx + yr * wy) + (zr * wz + b)

def det3x3(M):
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    return tf.add_n([
                M[0][0] * M[1][1] * M[2][2],
                M[0][1] * M[1][2] * M[2][0],
                M[0][2] * M[1][0] * M[2][1]
            ]) - tf.add_n([
                M[0][0] * M[1][2] * M[2][1],
                M[0][1] * M[1][0] * M[2][2],
                M[0][2] * M[1][1] * M[2][0]
            ])


class VTNAffineStem(Network):
    def __init__(self, name, flow_multiplier=1., **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier

    def build(self, img1, img2):
        
            #img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        
        concatImgs = tf.concat([img1, img2], 4, 'coloncatImgs')

        dims = 3
        conv1 = convolveLeakyReLU(
            'conv1',   concatImgs, 16,   3, 2)  # 64 * 64 * 64
        conv2 = convolveLeakyReLU(
            'conv2',   conv1,      32,   3, 2)  # 32 * 32 * 32
        conv3 = convolveLeakyReLU('conv3',   conv2,      64,   3, 2)
        conv3_1 = convolveLeakyReLU(
            'conv3_1', conv3,      64,   3, 1)
        conv4 = convolveLeakyReLU(
            'conv4',   conv3_1,    128,  3, 2)  # 16 * 16 * 16
        conv4_1 = convolveLeakyReLU(
            'conv4_1', conv4,      128,  3, 1)
        conv5 = convolveLeakyReLU(
            'conv5',   conv4_1,    256,  3, 2)  # 8 * 8 * 8
        conv5_1 = convolveLeakyReLU(
            'conv5_1', conv5,      256,  3, 1)
        conv6 = convolveLeakyReLU(
            'conv6',   conv5_1,    512,  3, 2)  # 4 * 4 * 4
        conv6_1 = convolveLeakyReLU(
            'conv6_1', conv6,      512,  3, 1)
        ks = conv6_1.shape.as_list()[1:4]
        conv7_W = tflearn.layers.conv_3d(
            conv6_1, 9, ks, strides=1, padding='valid', activation='linear', bias=False, scope='conv7_W')
        conv7_b = tflearn.layers.conv_3d(
            conv6_1, 3, ks, strides=1, padding='valid', activation='linear', bias=False, scope='conv7_b')

        I = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        W = tf.reshape(conv7_W, [-1, 3, 3]) * self.flow_multiplier
        b = tf.reshape(conv7_b, [-1, 3]) * self.flow_multiplier
        #print(W.shape.as_list(),b.shape.as_list(),'^'*40)
        A = W + I 
        # the flow is displacement(x) = place(x) - x = (Ax + b) - x
        # the model learns W = A - I.

        sx, sy, sz = img1.shape.as_list()[1:4]
        flow = affine_flow(W, b, sx, sy, sz)
        #print(flow.shape.as_list(),'&'*30)
        # determinant should be close to 1
        det = det3x3(A)
        det_loss = tf.nn.l2_loss(det - 1.0)
        # should be close to being orthogonal
        # C=A'A, a positive semi-definite matrix
        # should be close to I. For this, we require C
        # has eigen values close to 1 by minimizing
        # k1+1/k1+k2+1/k2+k3+1/k3.
        # to prevent NaN, minimize
        # k1+eps + (1+eps)^2/(k1+eps) + ...
        eps = 1e-5
        epsI = [[[eps * elem for elem in row] for row in Mat] for Mat in I]
        C = tf.matmul(A, A, True) + epsI

        def elem_sym_polys_of_eigen_values(M):
            M = [[M[:, i, j] for j in range(3)] for i in range(3)]
            sigma1 = tf.add_n([M[0][0], M[1][1], M[2][2]])
            sigma2 = tf.add_n([
                M[0][0] * M[1][1],
                M[1][1] * M[2][2],
                M[2][2] * M[0][0]
            ]) - tf.add_n([
                M[0][1] * M[1][0],
                M[1][2] * M[2][1],
                M[2][0] * M[0][2]
            ])
            sigma3 = tf.add_n([
                M[0][0] * M[1][1] * M[2][2],
                M[0][1] * M[1][2] * M[2][0],
                M[0][2] * M[1][0] * M[2][1]
            ]) - tf.add_n([
                M[0][0] * M[1][2] * M[2][1],
                M[0][1] * M[1][0] * M[2][2],
                M[0][2] * M[1][1] * M[2][0]
            ])
            return sigma1, sigma2, sigma3
        s1, s2, s3 = elem_sym_polys_of_eigen_values(C)
        ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
        ortho_loss = tf.reduce_sum(ortho_loss)

        return {
            'flow': flow,
            'W': W,
            'b': b,
            'det_loss': det_loss,
            'ortho_loss': ortho_loss
        }
