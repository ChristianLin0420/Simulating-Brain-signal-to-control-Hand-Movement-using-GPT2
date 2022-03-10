# Simulating-Brain-signal-to-control-Hand-Movement-using-GPT2

## Abstarct
Recently, Vision Transformer (ViT) has achieved state-of-the-art performance of
image classification in the ImageNet dataset and also proved successful in other
applications, such as object detection and semantic image segmentation. Because
deep learning models are inspired by the structure and function of the human
brain, we are interested in whether they indeed process information like human
brains. In this paper, we make the first attempt to try to understand what deep
learning models learn by looking at the brain signals machines generate. We de-
signed the experiment with an electroencephalogram (EEG) brain-computer inter-
face (BCI), via which we attempt to make machines perform specific tasks such
as imagining hand movements. The proposed target-task classifier was built using
spatio-temporal features of cortical activity in the motor area estimated from hu-
man’s EEG. We utilize a Generative Adversarial Network (GAN) architecture to
build a Generative Pre-trained Transformer (GPT) model that can generate ’ma-
chine’ brain signals spontaneously to control machines. Instead of training a GAN
model with real EEG data of the target task, the proposed framework is first pre-
trained with a different task or subject and later fine-tuned with the target task.
This approach enables us to build a pre-trained model that is general enough to be
later transferred to various kind of application scenarios. In this work, we demon-
strate that the proposed framework can be used for simulating human-like brain
signals and build the EEG classification model that can be transferred between
different tasks and different subjects.


## Configurations
| Configuration  | Value Type    | Description   |
| -------------  | ------------- | ------------- |
| gpu            | Boolean       | Whether using the gpu |
| gpu_id         | Integer       | Identification(index) of the gpu |
| parallel       | Boolean       | Whether using multiple gpus(not available now)
| subject_count  | Integer       | Number of subjects' data to train |
| data_path      | String        | Training data paths name |
| model_name     | String        | Training model name |
| buffer_size    | Integer       | ... |
| batch_size     | Integer       | Number of batch |
| epoches        | Integer       | Number of training epoches |
| rounds         | Integer       | Number of training rounds  |
| learning_rate  | Float         | Learning rate(d_optimizer, g_optimizer, c_optimizer) |
| random_vector_num | Integer    | Number of generated random vectors |
| example_to_generate | Integer  | Number of generated brain signals during training |
| condition_size | Integer       | Size of conditional one-hot vector |
| class_count    | Integer       | Number of the classes |
| noise_variance | Float         | Value of the variance to generate random vectors |
| fine_tune      | Boolena       | Whether using fine-tune training(only for gpt2xcnn, gpt2scnn) |
| pretrained_finetune_path | String | Path of the pretrained generator model |
| pretrained_classifier_path | String | path of the pretrained classifier |
| vocab_size     | Integer       | Vocabulary size of the GPT-2 model  |
| n_positions    | Integer       | The maximum sequence length that this model might ever be used with |
| n_ctx          | Integer       | Dimensionality of the causal mask (usually same as n_positions) |
| n_embd         | Integer       | Dimensionality of the embeddings and hidden states |
| n_layer        | Integer       | Number of hidden layers in the Transformer encoder |
| n_head         | Integer       | Number of attention heads for each attention layer in the Transformer encoder |
| n_inner        | Integer       | Dimensionality of the inner feed-forward layers |
| activation_function | String   | Activation function, to be selected in the list :obj:["relu", "silu", "gelu", "tanh", "gelu_new"] |
| resid_pdrop    | Float         | The dropout probability for all fully connected layers in the embeddings, encoder, and pooler |
| embd_pdrop     | Float         | The dropout ratio for the embeddings |
| attn_pdrop     | Float         | The dropout ratio for the attention |
| layer_norm_epsilon | Float     | The epsilon to use in the layer normalization layers |
| initializer_range  | Float     | The standard deviation of the truncated_normal_initializer for initializing all weight matrices |


## Execution

### Step 1 : Set configuration file(config/config.json)

### Step 2 : Start training mode with command
```
python main.py --mode training(default)
```
