import tensorflow as tf
from keras.callbacks import Callback
from keras.models import Model
from keras.metrics import Mean
from keras.utils import array_to_img


# class for overriding
class WGAN_gp(Model):
    """This method for overriding to keras original model to create wgan with gradient penalty"""

    # initialize model
    def __init__(self, generator, discriminator, latent_dim):
        # override
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = generator
        self.discriminator = discriminator
        self.gp_weights = 10
        self.d_extra_steps = 5

    # compile
    def compile(self, g_loss, d_loss, d_opt, g_opt):
        super().compile()

        self.generator_optimzer = g_opt
        self.generator_loss = g_loss

        self.discriminator_loss = d_loss
        self.discriminator_optimizer = d_opt

        # metrics
        self.d_metrics = Mean(name="discriminator loss")
        self.g_metrics = Mean(name="generator loss")

    # gradient penalty
    def gradient_penalty(self, batch_size, real_images, fake_images):
        # alpha value
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images

        interpolation = real_images + alpha * diff

        # gradient calculation
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolation)

            # pred
            prediction = self.discriminator(interpolation, training=True)

        # calculate
        grads = gp_tape.gradient(prediction, [interpolation])[0]

        # norm
        norm = tf.sqrt(tf.reduce_mean(tf.square(grads), axis=[1, 2, 3]))

        # apply
        gradient_penalty = tf.reduce_mean((norm - 1) ** 2)

        return gradient_penalty

    # train step
    def train_step(self, real_images):

        # tuple control
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # batchsize
        batch_size = tf.shape(real_images)[0]

        # run main section
        for i in range(self.d_extra_steps):
            # create noise
            random_latent_vector = tf.random.normal([batch_size, self.latent_dim])

            # gradient calculation
            with tf.GradientTape() as d_tape:
                # fake images
                fake_images = self.generator(random_latent_vector, training=True)

                # get logits
                fake_logits = self.discriminator(fake_images, training=True)

                # real logits
                real_logits = self.discriminator(real_images, training=True)

                # calculate loss
                d_cost = self.discriminator_loss(real_output=real_logits, fake_output=fake_logits)

                # gradient penalty
                gp = self.gradient_penalty(batch_size=batch_size, real_images=real_images, fake_images=fake_images)

                # original loss concat
                d_loss = d_cost + gp * self.gp_weights

            # update gradient
            grads = d_tape.gradient(d_loss, self.discriminator.trainable_weights)

            # optimize
            self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

            # generator training
            random_latent_vector = tf.random.normal([batch_size, self.latent_dim])

            with tf.GradientTape() as g_tape:
                # generated images
                generate_images = self.generator(random_latent_vector, training=True)

                gen_logits = self.discriminator(generate_images, training=True)

                g_loss = self.generator_loss(gen_logits)

            # get the gradient
            grads = g_tape.gradient(g_loss, self.generator.trainable_weights)

            # optimize
            self.generator_optimzer.apply_gradients(zip(grads, self.generator.trainable_weights))

            # update metrics
            self.g_metrics.update_state(g_loss)
            self.d_metrics.update_state(d_loss)

            return {"discriminator_loss: ": self.d_metrics.result(),
                    "generator_loss": self.g_metrics.result()}


# class for monitoring
class Monitor(Callback):
    def __init__(self, num_img, latent_dim):
        self.num_img = num_img
        self.latent = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_vector = tf.random.normal([self.num_img, self.latent])
        generated_images = self.model.generator(random_vector)
        generated_images = (generated_images*127.5) + 127.5

        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = array_to_img(img)
            img.save(f"created_images/epoch_{epoch}_step_{i}.png")

        with open("epochs.txt", "a") as csv:
            csv.write(f"{epoch+300}/{self.model.d_metrics.result()}/{self.model.g_metrics.result()}\n")