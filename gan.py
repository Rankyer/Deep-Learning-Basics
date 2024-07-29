import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import mnist
from tqdm import tqdm
#from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.layers import LeakyReLU
## Use this for non-M1/M2 Macs and Intel machines
from tensorflow.keras.optimizers import Adam
## Use this for M1/M2 Macs
#from tensorflow.keras.optimizers.legacy import Adam

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    
    # Reshape from 28x28 image to 784 element vector
    x_train = x_train.reshape(x_train.shape[0], 784)

    return (x_train, y_train, x_test, y_test)

(x_train, y_train, x_test, y_test) = load_data()

def adam_optimizer():
    return Adam(learning_rate = 0.0002, beta_1 = 0.5)

def create_generator():
    generator = Sequential()
    generator.add(Dense(units = 256, input_dim = 100))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(units = 512))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(units = 784, activation = 'tanh'))
    generator.compile(loss = 'binary_crossentropy', optimizer = adam_optimizer())

    return generator

def create_discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(units = 1024, input_dim = 784))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(units = 512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    
    discriminator.add(Dense(units = 256))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Dense(units = 1, activation = 'sigmoid'))

    discriminator.compile(loss = 'binary_crossentropy', optimizer = adam_optimizer())
    return discriminator

# Create the GAN
def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape = (100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs = gan_input, outputs = gan_output)
    gan.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    return gan

# Display images
def plot_generated_images(epoch, generator, examples = 100, dim = (10, 10), figsize = (10, 10)):
    noise = np.random.normal(loc = 0, scale = 1, size = [examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100, 28, 28)
    plt.figure(figsize = figsize)

    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation = 'nearest')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('gan_generated_image %d.png' %epoch)

def training(epochs = 1, batch_size = 128):
    # Load the training data
    (X_train, y_train, X_test, y_test) = load_data()
    batch_count = X_train.shape[0] / batch_size

    # Create the GAN
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(discriminator, generator)

    for e in range(1, epochs + 1):
        print("Epoch %d." % e)
        for _ in tqdm(range(batch_size)):

            # Generate random noise
            noise = np.random.normal(0, 1, [batch_size, 100])

            # Generate fake images
            generated_images = generator.predict(noise)
            
            # Get a random batch of real images
            image_batch = X_train[np.random.randint(low = 0, high = X_train.shape[0], size = batch_size)]
            X = np.concatenate([image_batch, generated_images])

            # Labels for the real and fake data
            y_dis = np.zeros(2 * batch_size)

            # Real data labels
            y_dis[:batch_size] = 0.9

            # Pretrain the discriminator on real and fake images 
            # before starting the GAN

            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Create fake data and fake labels
            noise = np.random.normal(0, 1, [batch_size, 100])
            y_gen = np.ones(batch_size)

            # We freeze the discriminator while training the GAN
            # So we alternate between training the discriminator
            # and the GAN

            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 20 == 0:
            plot_generated_images(e, generator)

training(epochs = 400, batch_size = 128)
