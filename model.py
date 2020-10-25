import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense,LeakyReLU,Input
from keras.optimizers import Adam

batch_size = 16
step_per_epoch = 3750
epochs = 10

(x_train,x_test),(y_train,y_test) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = x_train.reshape(-1,28*28*1)


'''
import matplotlib.pyplot as plt
plt.imshow(x_train[200].reshape(28,28))
plt.show()
'''

def Generator():
  generator = Sequential()
  generator.add(Dense(210,input_dim=100))
  generator.add(LeakyReLU(0.2))

  generator.add(Dense(510))
  generator.add(LeakyReLU(0.2))

  generator.add(Dense(810))
  generator.add(LeakyReLU(0.2))

  generator.add(Dense(1010))
  generator.add(LeakyReLU(0.2))

  generator.add(Dense(28*28*1,activation="tanh"))

  generator.compile(optimizer=Adam(0.0002,0.5),loss="binary_crossentropy")

  return generator

def Descriminator():
  descriminator = Sequential()
  descriminator.add(Dense(1010,input_dim=28*28*1))
  descriminator.add(LeakyReLU(0.2))

  descriminator.add(Dense(810))
  descriminator.add(LeakyReLU(0.2))

  descriminator.add(Dense(510))
  descriminator.add(LeakyReLU(0.2))

  descriminator.add(Dense(210))
  descriminator.add(LeakyReLU(0.2))

  descriminator.add(Dense(1,activation="sigmoid"))
  descriminator.compile(optimizer=Adam(0.0002,0.5),loss="binary_crossentropy")
  return descriminator
  
desc = Descriminator()
gen = Generator()
desc.trainable = False
gan_input = Input(shape=(100,))
fake_img = gen(gan_input)
gan_output = desc(fake_img)
gan = Model(gan_input,gan_output)
gan.compile(loss="binary_crossentropy",optimizer=Adam(0.0002,0.5))
for epoch in range(epochs):
  for batch in range(step_per_epoch):
    noise = np.random.normal(0,1,size=(batch_size,100))
    fake = gen.predict(noise)
    real = x_train[np.random.randint(0,x_train.shape[0],size=batch_size)]
    x = np.concatenate((real,fake))
    label_real = np.ones(2*batch_size)
    label_real[:batch_size] = 0.9
    desc_loss = desc.train_on_batch(x,label_real)
    label_fake = np.zeros(batch_size)
    gen_loss = gan.train_on_batch(noise,label_fake)
  print(f"Epoch : {epoch} / Descriminator Loss : {desc_loss} / Generator Loss : {gen_loss}")
def show_image(noise):
 images = gen.predict(noise)
 plt.figure(figsize=(5,4))
 for i,image in enumerate(images):
  plt.subplot(5,4,i+1)
  plt.imshow(images[i].reshape((28,28)))
 plt.show()
noise = np.random.normal(0,1,size=(20,100))
show_image(noise)
 
  
