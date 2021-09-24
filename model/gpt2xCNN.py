
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .gpt2cgan import gpt2cgan
from .classifier import get_pretrained_classfier

class gpt2xcnn(tf.keras.Model):

    def __init__(self, config = None, generator = None, classifier = None, noise_len: int = 784, noise_dim: int = 32, d_extra_steps: int = 5, last_dim: int = 3, **kwargs):

        super(gpt2xcnn, self).__init__()

        if generator is None:
            self.gptGenerator = gpt2cgan(config, noise_len = noise_len, noise_dim = noise_dim, d_extra_steps = d_extra_steps, last_dim = last_dim).generator
        else:
            self.gptGenerator = generator

        self.classifier = classifier

    def compile(self, optimizer, loss_fn):
        super(gpt2xcnn, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):

        seeds, labels = data
        # seeds = np.asarray(seeds)
        # labels = np.asarray(labels)

        # print("gpt2xcnn input signals shape: {}".format(seeds.shape))
        # print("gpt2xcnn input labels shape: {}".format(labels.shape))

        signals = tf.constant([])
        generate_count = 16
        generate_round = int(seeds.shape[0] / generate_count)

        for idx in range(generate_round):
            sigs = self.gptGenerator(seeds[idx * generate_count:(idx + 1) * generate_count])

            print("idx: {}, shape of sigs: {}".format(idx, sigs.shape))

            if signals.shape[0] == 0:
                signals = sigs
            else:
                signals = tf.concat([signals, sigs], axis = 0)
        
        print("signals shape: {}".format(signals.shape))

        signals_shape = signals.shape
        signals = tf.reshape(signals, shape = [signals_shape[0], signals_shape[1], signals_shape[2]])

        batch = signals.shape[0]
        images = tf.constant([])

        for idx in range(batch):
            signal = signals[idx]

            Zxx = tf.signal.stft(signal, frame_length = 256, frame_step = 16)
            Zxx = tf.abs(Zxx)
            Zxx = tf.expand_dims(Zxx, axis = 0)

            if images.shape[0] == 0:
                images = Zxx
            else:
                images = tf.concat([images, Zxx], axis = 0)


        # images = np.asarray(images)
        # print("images shape: {}".format(images.shape))
        # print("images tyoe : {}".format(type(images)))

        images = images.numpy()
        # images = np.add(images, 0)

        print("images shape: {}".format(images.shape))
        print("images type: {}".format(type(images)))

        rgb_weights = [0.2989, 0.5870, 0.1140]
        X = None

        for idx in range(images.shape[0]):
            current_image = None
            current_data = images[idx][:, :, :40]

            print("start processing number.{} to stft image".format(idx))

            for channel in range(current_data.shape[0]):
                fig = plt.figure(figsize = (16, 40), dpi = 1)
                plot = fig.add_subplot(111)

                log_spec = np.log(current_data[channel].T)
                height = log_spec.shape[0]
                width = log_spec.shape[1]
                x_axis = np.linspace(0, 2, num = width)
                y_axis = range(height)
                plot.pcolormesh(x_axis, y_axis, log_spec, shading = 'auto')
                plot.axis('off')
                fig.tight_layout(pad = 0)
                fig.canvas.draw()

                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype = np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                plt.close(fig)
                img = np.array(img, dtype = np.float32) / 255

                # convert rgb to gray scale
                img = np.dot(img[...,:3], rgb_weights)
                img = np.expand_dims(img, axis = 2)

                if current_image is None:
                    current_image = img
                else:
                    current_image = np.append(current_image, img, axis = 1)

            current_image = np.expand_dims(current_image, axis = 0)

            if X is None:
                X = current_image
            else:
                X = np.append(X, current_image, axis=0)

        print("Input X shape: {}".format(X.shape))

        # strategy = tf.distribute.MirroredStrategy(['GPU:0'])
        # def func():
        #     replica_context = tf.distribute.get_replica_context()
        #     return replica_context.replica_id_in_sync_group
        # strategy.run(func)

        with tf.GradientTape() as tape:
            y_pred = self.classifier(X)
            # loss, acc = self.classifier.evaluate(X, labels, verbose = 1)
            loss = self.loss_fn(labels, y_pred)
        
        grads = tape.gradients(loss, self.gptGenerator.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.gptGenerator.trainable_weights))

        match = 0

        for x, y in zip(labels, y_pred):
            if x == y:
                match += 1

        return {"loss": loss, "accuracy": float(match) / float(batch)}

        # return {"loss": loss, "accuracy": acc}