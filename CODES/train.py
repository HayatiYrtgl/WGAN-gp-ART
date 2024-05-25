from generator_discriminator import create_models
from model import WGAN_gp, Monitor
from losses_optimizers import create_loss_optimizer
from keras.utils import image_dataset_from_directory
import os
from keras.models import load_model

concated = os.path.join("../DATASETS/Portraits")

# create dataset
dataset = image_dataset_from_directory(directory=concated, image_size=(128, 128),
                                       color_mode="rgb", label_mode=None, batch_size=16)

dataset = dataset.map(lambda x: (x-127.5)/127.5)

# epochs
epochs = 500

# generator & discriminator
generator = load_model("generator.h5")

discriminator = load_model("discriminator.h5")

# losses
gen_loss = create_loss_optimizer.generator_loss
disc_loss = create_loss_optimizer.discriminator_loss

# optimizers
d_opt = create_loss_optimizer.discriminator_optimizer
g_opt = create_loss_optimizer.generator_optimizer

# gan model
wgan = WGAN_gp(generator=generator, discriminator=discriminator, latent_dim=128)
callbacks = [Monitor(latent_dim=128, num_img=10)]

# fit
wgan.compile(g_loss=gen_loss, d_loss=disc_loss, d_opt=d_opt, g_opt=g_opt)

# fit
wgan.fit(dataset, callbacks=callbacks, epochs=epochs)
generator.save("generator_updated.h5")
discriminator.save("discriminator_updated.h5")