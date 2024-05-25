from keras.optimizers import Adam
import tensorflow as tf


# class for optimizer and losses
class LossOptimizer:
    """This class creates optimizers and losses"""

    def __init__(self):
        self.generator_optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        self.discriminator_optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

    # loss
    @staticmethod
    def generator_loss(fake_output):
        return -1 * tf.reduce_mean(fake_output)

    @staticmethod
    def discriminator_loss(real_output, fake_output):
        real_loss = tf.reduce_mean(real_output)
        fake_loss = tf.reduce_mean(fake_output)
        return fake_loss - real_loss


create_loss_optimizer = LossOptimizer()
