
import os
import time as tt
import numpy as np
import random
import tensorflow as tf

from datetime import datetime
from alive_progress import alive_bar
from tensorflow import keras

from .gpt2 import TFGPT2MainLayer
from .discriminator import Discriminator
from utils.model_monitor import generate_and_save_images, save_loss_record, save_result_as_gif
from utils.output import TFCausalLMOutputWithPast
from utils.model_utils import input_processing, TFPreTrainedModel, TFCausalLanguageModelingLoss

class GPT2GAN(TFPreTrainedModel, TFCausalLanguageModelingLoss):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        
        self.generator = TFGPT2MainLayer(config, name = "generator")
        self.discriminator = Discriminator(config, name = "discriminator")

        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        # training step
        now = datetime.now()
        self.time = now.strftime("%d_%m_%Y_%H_%M_%S")

        self.model_name = "gpt2gan"
        self.log_dir = "./logs/gpt2gan/{}".format(self.time)

    def get_output_embeddings(self):
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def Callback_EarlyStopping(self, LossList, min_delta = 0.1, patience = 20):
        #No early stopping for 2 * patience epochs 
        if len(LossList) // patience < 2 :
            return False

        # Mean loss for last patience epochs and second-last patience epochs
        mean_previous = np.mean(LossList[::-1][patience:2*patience]) # second-last
        mean_recent = np.mean(LossList[::-1][:patience]) #last

        # you can use relative or absolute change
        delta_abs = np.abs(mean_recent - mean_previous) #abs change
        delta_abs = np.abs(delta_abs / mean_previous)  # relative change

        if delta_abs < min_delta :
            print("*CB_ES* Loss didn't change much from last %d epochs"%(patience))
            print("*CB_ES* Percent change in loss value:", delta_abs*1e2)
            return True
        else:
            return False

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step( self,
                    images, 
                    batch_size: int = 8, 
                    noise_len: int = 784, 
                    noise_dim: int = 32):

        noise = tf.random.normal([batch_size, noise_len, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training = True)

            real_output = self.discriminator(images, training = True)
            fake_output = self.discriminator(generated_images, training = True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return float(gen_loss), float(disc_loss)

    def train(  self, 
                dataset, 
                epochs, 
                batch_size, 
                num_examples_to_generate, 
                noise_len, 
                noise_dim   ):

        seed = tf.random.normal([num_examples_to_generate, noise_len, noise_dim])

        diractory = './checkpoints/gpt2gan_training_checkpoints/{}'.format(self.time)

        if not os.path.exists(diractory):
            os.mkdir(diractory)
        
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        checkpoint_dir = './checkpoints/gpt2gan_training_checkpoints/{}'.format(self.time)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer = self.generator_optimizer,
                                         discriminator_optimizer = self.discriminator_optimizer,
                                         generator = self.generator,
                                         discriminator = self.discriminator)

        tb_callback = tf.keras.callbacks.TensorBoard(self.log_dir)
        tb_callback.set_model(self)

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

                stopEarly = self.Callback_EarlyStopping(gen_loss_record, min_delta = 0.1, patience = 20)

                if stopEarly:
                    print("Callback_EarlyStopping signal received at epoch = {}/{}".format(i, epochs))
                    print("Terminating training ")
                    break

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