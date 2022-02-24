from tensorflow import keras
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D,MaxPooling2D,Conv1D,Cropping2D,Concatenate
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.initializers import RandomNormal
from keras.layers import Concatenate
from keras.models import load_model
from keras.initializers import RandomNormal

def UNET(input_shape=(572,572,1)):
  init = RandomNormal(stddev=0.02)
  input=Input(shape=input_shape)

  '''block 1'''
  l1=Conv2D(1, (3,3), activation ='relu', kernel_initializer=init)(input)
  l1=Conv2D(64, (3,3),  activation ='relu',kernel_initializer=init)(l1)

  crop_1=Cropping2D(88)(l1)

  '''block 2'''
  l2 =MaxPooling2D((2,2))(l1)
  l2=Conv2D(128, (3,3),   activation ='relu',kernel_initializer=init)(l2)
  l2=Conv2D(128, (3,3),   activation ='relu',kernel_initializer=init)(l2)

  crop_2=Cropping2D(40)(l2)

  '''block 3'''
  l3 =MaxPooling2D((2,2))(l2)
  l3=Conv2D(256, (3,3),  activation ='relu',kernel_initializer=init)(l3)
  l3=Conv2D(256, (3,3),  activation ='relu',kernel_initializer=init)(l3)

  crop_3=Cropping2D(16)(l3)

  '''block 4'''
  l4 =MaxPooling2D((2,2))(l3)
  l4=Conv2D(512, (3,3),  activation ='relu', kernel_initializer=init)(l4)
  l4=Conv2D(512, (3,3),  activation ='relu',kernel_initializer=init)(l4)

  crop_4=Cropping2D(4)(l4)
  '''block 5'''
  l5 =MaxPooling2D((2,2))(l4)
  l5=Conv2D(1024, (3,3), activation ='relu',kernel_initializer=init)(l5)
  l5=Conv2D(1024, (3,3), activation ='relu',kernel_initializer=init)(l5)


  '''block 6'''
  u0=Conv2DTranspose(512, (2,2), strides=(2, 2), kernel_initializer=init)(l5)
  u0=Concatenate()([u0,crop_4])
  u0=Conv2D(512, (3,3), activation ='relu', kernel_initializer=init)(u0)
  u0=Conv2D(512, (3,3), activation ='relu', kernel_initializer=init)(u0)

  '''block 7'''
  u1=Conv2DTranspose(256, (2,2), strides=(2, 2), kernel_initializer=init)(u0)
  u1=Concatenate()([u1,crop_3])
  u1=Conv2D(256, (3,3), activation ='relu', kernel_initializer=init)(u1)
  u1=Conv2D(256, (3,3), activation ='relu', kernel_initializer=init)(u1)

  '''block 8'''
  u2=Conv2DTranspose(128, (2,2),strides=(2, 2), kernel_initializer=init)(u1)
  u2=Concatenate()([u2,crop_2])
  u2=Conv2D(128, (3,3), activation ='relu', kernel_initializer=init)(u2)
  u2=Conv2D(128, (3,3),  activation ='relu', kernel_initializer=init)(u2)

  '''block 9'''
  u3=Conv2DTranspose(64, (2,2), strides=(2, 2),kernel_initializer=init)(u2)

  u3=Concatenate()([u3,crop_1])
  u3=Conv2D(64, (3,3),  activation ='relu', kernel_initializer=init)(u3)
  u3=Conv2D(64, (3,3),  activation ='relu', kernel_initializer=init)(u3)
  op=Conv2D(2, (1,1),  activation ='relu', kernel_initializer=init)(u3)
 

  model = Model(input, op)
    
  return model