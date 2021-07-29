
import os
import time as tt
import numpy as np
import tensorflow as tf

from datetime import datetime
from typing import Any
from alive_progress import alive_bar

from .gpt2 import TFGPT2MainLayer
from .discriminator import Discriminator
from utils.model_monitor import generate_and_save_images, save_loss_record, save_result_as_gif
from utils.output import TFCausalLMOutputWithPast
from utils.model_utils import input_processing, TFPreTrainedModel, TFCausalLanguageModelingLoss

class GPT2WGAN(TFPreTrainedModel, TFCausalLanguageModelingLoss):

    def __init__(self, config, d_extra_steps, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.generator = TFGPT2MainLayer(config, name = "generator")
        self.discriminator = Discriminator(config, name = "discriminator")

        self.d_extra_steps = d_extra_steps
        self.gp_weight = 10.0

        # This method returns a helper function to compute cross entropy loss
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        # training step
        now = datetime.now()
        self.time = now.strftime("%d_%m_%Y_%H_%M_%S")

        self.model_name = "gpt2wgan"

    def get_output_embeddings(self):
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = tf.reduce_mean(real_output)
        fake_loss = tf.reduce_mean(fake_output)
        return fake_loss - real_loss

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def gradient_penalty(self, 
                         real_images, 
                         fake_images, 
                         batch_size: int = 8):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training = True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis = [1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step( self,
                    images, 
                    batch_size: int = 8, 
                    noise_len: int = 784, 
                    noise_dim: int = 32):

        for _ in range(self.d_extra_steps):
            
            noise = tf.random.normal([batch_size, noise_len, noise_dim])

            with tf.GradientTape() as tape:
                # Generate fake images from the random noise
                generated_images = self.generator(noise, training = True)
                
                # Get the logits for the real images
                real_output = self.discriminator(images, training = True)
                # Get the logits for the fake images
                fake_output = self.discriminator(generated_images, training = True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.discriminator_loss(real_output, fake_output)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(images, generated_images, batch_size)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        noise = tf.random.normal([batch_size, noise_len, noise_dim])
        
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(noise, training = True)
            # Get the discriminator logits for fake images
            fake_output = self.discriminator(generated_images, training = True)
            # Calculate the generator loss
            g_loss = self.generator_loss(fake_output)

        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))

        return float(g_loss), float(d_loss)

    def train(  self, 
                dataset, 
                epochs, 
                batch_size, 
                num_examples_to_generate, 
                noise_len, 
                noise_dim):

        seed = tf.random.normal([num_examples_to_generate, noise_len, noise_dim])

        diractory = './checkpoints/gpt2_training_checkpoints/{}'.format(self.time)

        if not os.path.exists(diractory):
            os.mkdir(diractory)

        checkpoint_dir = './checkpoints/gpt2_training_checkpoints/{}'.format(self.time)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer = self.generator_optimizer,
                                         discriminator_optimizer = self.discriminator_optimizer,
                                         generator = self.generator,
                                         discriminator = self.discriminator)

        gen_loss_record = []
        dis_loss_record = []

        with alive_bar(epochs) as bar:
            for i in range(epochs):
                print("epoch: {}".format(i))

                start = tt.time()
                
                for k, image_batch in enumerate(dataset):
                    g_l, d_l = self.train_step( images = image_batch, 
                                                batch_size = batch_size, 
                                                noise_len = noise_len, 
                                                noise_dim = noise_dim)

                    if k % 1500 == 0:
                        gen_loss_record.append(g_l)
                        dis_loss_record.append(d_l)

                generate_and_save_images(self.generator, self.time, i + 1, seed, self.model_name)

                # Save the model every 10 epochs
                if (i + 1) % 10 == 0:
                    checkpoint.save(file_prefix = checkpoint_prefix)

                print ('Time for epoch {} is {} sec'.format(i + 1, tt.time() - start))
            
                bar()

            # Generate after the final epoch
            print("epoch: {}".format(i))
            generate_and_save_images(self.generator, self.time, epochs, seed, self.model_name)
            print("finished generating images")
            print("-" * 80)

        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        assert len(gen_loss_record) == len(dis_loss_record)

        record_range = np.arange(1, len(gen_loss_record) + 1, 1).tolist()

        save_loss_record(record_range, gen_loss_record, dis_loss_record, self.time, self.model_name)
        save_result_as_gif(self.time, self.model_name)

    def call(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )

        generator_outputs = self.generator(
            input_ids=inputs["input_ids"],
            past=inputs["past"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            use_cache=inputs["use_cache"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        hidden_states = generator_outputs[0]
        logits = self.generator.wte(hidden_states, mode="linear")

        loss = None

        if inputs["labels"] is not None:
            # shift labels to the left and cut last logit token
            logits = logits[:, :-1]
            labels = inputs["labels"][:, 1:]
            loss = self.compute_loss(labels, logits)

        if not inputs["return_dict"]:
            output = (logits,) + generator_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFCausalLMOutputWithPast(
            loss = loss,
            logits = logits,
            past_key_values = generator_outputs.past_key_values,
            hidden_states = generator_outputs.hidden_states,
            attentions = generator_outputs.attentions,
        )