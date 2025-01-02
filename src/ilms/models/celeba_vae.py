from flax import linen as nn
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import random
import distrax


ENCODER_CONV_UNITS = [[128,(4,4),(1,1), nn.activation.relu],   
                      [128,(4,4),(2,2), nn.activation.relu],   
                      [256,(4,4),(2,2), nn.activation.relu], 
                      [256,(4,4),(2,2), nn.activation.relu], 
                      [256,(4,4),(1,1), nn.activation.relu]]  

ENCODER_DENSE_UNITS = [[256, nn.activation.relu], 
                       [256 * 2, None]]

DECODER_DENSE_UNITS = [[256, nn.activation.relu], 
                       [256 * 8 * 8, nn.activation.relu]]

DECODER_CONV_UNITS = [[256,(4,4),(1,1), nn.activation.relu],
                      [256,(4,4),(2,2), nn.activation.relu],
                      [256,(4,4),(2,2), nn.activation.relu],
                      [128,(4,4),(2,2), nn.activation.relu],
                      [3*2,(4,4),(1,1), None]]


def unflatten_shape(INPUT_SHAPE=(64,64,3), BATCH_SIZE=32):
    h,w = INPUT_SHAPE[:2]
    for stride in ENCODER_CONV_UNITS:
        h = h//stride[2][0]
        w = w//stride[2][1]   
    
    assert DECODER_DENSE_UNITS[-1][0] % (h*w) == 0
    unflatten_C = DECODER_DENSE_UNITS[-1][0] // (h*w)
    x=(BATCH_SIZE, h, w, unflatten_C)
    return x

def z_shape(BATCH_SIZE=32):
    return [BATCH_SIZE, ENCODER_DENSE_UNITS[-1][0] // 2]



class EncoderConvs(nn.Module):
    """
    Encoder Block.

    The Block is made of convolutions that downsample the image
    resolution until a certain point, after which we flatten the image
    and use a stack of Dense layers to get the posterior distribution q(z|x).
    """
    def setup(self):
        convs_list = []        
        for filters, kernel_size, stride, activation in ENCODER_CONV_UNITS:
            convs_list.append(nn.Conv(filters, kernel_size, stride))
            if activation is not None:
                convs_list.append(activation)
        self.convs_list = convs_list

        dense_list = []
        for filters, activation in ENCODER_DENSE_UNITS:
            dense_list.append(nn.Dense(filters))
            if activation is not None:
                dense_list.append(activation)

        self.dense_list = dense_list
        
                
    def __call__(self,x):
        for conv in self.convs_list:
            x = conv(x)

        # (B, h, w, C) -> (B, h*w*C)
        x = jnp.reshape(x, shape=(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
        
        for dense in self.dense_list:
            x = dense(x)
        return x  

class ResizeAndConv(nn.Module):
    """
    Resize-Conv Block.

    A simple Nearest-Neighbord upsampling + Conv block, used to upsample images instead of Deconv layers.
    This block is useful to avoid checkerboard artifacts: https://distill.pub/2016/deconv-checkerboard/
    """
    filters: int
    kernel_size: Tuple[int]
    stride: Tuple[int]

    def setup(self):
        self.conv = nn.Conv(self.filters, self.kernel_size, (1,1))
        
    def __call__(self,x):
        if self.stride != (1,1):
            x = jax.image.resize(x, (x.shape[0], x.shape[1]*self.stride[0],x.shape[2]*self.stride[1],x.shape[3]), method = 'nearest')
        x = self.conv(x)
        return x
    
class DecoderConvs(nn.Module):
    """
    Decoder Block.

    The Decoder block starts off with Dense layers to process the sample z,
    followed by an unflatten (reshape) operation into an activation of shape (B, h, w, C).
    The activation is then upsampled back to the original image size using a stack
    of resize-conv blocks.
    """
    def setup(self, INPUT_SHAPE=(64,64,3), BATCH_SIZE=32):
        convs_list=[]        
        for filters, kernel_size, stride, activation in DECODER_CONV_UNITS:            
            convs_list.append(ResizeAndConv(filters, kernel_size, stride))
            if activation is not None:
                convs_list.append(activation)
        self.convs_list = convs_list 

        dense_list = []
        for filters, activation in DECODER_DENSE_UNITS:
            dense_list.append(nn.Dense(filters))
            if activation is not None:
                dense_list.append(activation)

        self.dense_list = dense_list
        self.INPUT_SHAPE = INPUT_SHAPE
        self.BATCH_SIZE = BATCH_SIZE
              
    def __call__(self,x):
        for dense in self.dense_list:
            x = dense(x)

        # (B, C) -> (B, h, w, new_C)
        x = jnp.reshape(x,shape=unflatten_shape(self.INPUT_SHAPE, self.BATCH_SIZE))
        
        for conv in self.convs_list:
            x = conv(x)
        
        return x   

class VAEModel(nn.Module):
    """
    VAE model.

    A simple Encoder-Decoder architecture where both Encoder and Decoder model multivariate
    gaussian distributions.
    """
    def setup(self, BATCH_SIZE=32):
        self.encoder_convs = EncoderConvs()
        self.decoder_convs = DecoderConvs()
        self.batch_size = BATCH_SIZE

    def __call__(self, key, inputs):
        enc_mean, enc_logstd = self.encode(inputs)
        epsilon = random.normal(key, enc_mean.shape)
        z = epsilon * jnp.exp(enc_logstd) + enc_mean
        dec_mean, dec_logstd = self.decode(z)
        return enc_mean, enc_logstd, dec_mean, dec_logstd

    def encode(self, inputs):
        enc_x = self.encoder_convs(inputs)
        enc_mean, enc_logstd = jnp.split(enc_x, 2, axis=-1)
        return enc_mean, enc_logstd

    def decode(self, z):
        dec_x = self.decoder_convs(z)
        dec_mean, dec_logstd = jnp.split(dec_x, 2, axis=-1)
        return dec_mean, dec_logstd
    
    def generate(self, key, z_temp=1., x_temp=1.):
        """
        Randomly sample z from the prior distribution N(0, 1) and generate the image x from z.

        z_temp: float, defines the temperature multiplier of the encoder stddev. 
            Smaller z_temp makes the generated samples less diverse and more generic
        x_temp: float, defines the temperature multiplier of the decoder stddev.
            Smaller x_temp makes the generated samples smoother, and loses small degree of information.
        """
        # Generate random samples from the prior distribution N(0, 1)
        key1, key2 = random.split(key)

        z = random.normal(key1, z_shape(self.batch_size))
        z = z * z_temp  # "Reparametrization" to N(0, 1 * z_temp)
        
        dec_mean, dec_logstd = self.decode(z)
        return distrax.Normal(dec_mean, jnp.exp(dec_logstd) * x_temp).sample(seed=key2)