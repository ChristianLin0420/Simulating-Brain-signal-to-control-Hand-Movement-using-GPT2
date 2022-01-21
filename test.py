
import os
import numpy as np
import matplotlib.pyplot as plt
import json 
import tensorflow as tf

def save_loss_range_record(lst_iter, lines, time, model_name, experiment_name, line_names):

    title = "{} of Noise Variance Ablation Experiment".format(experiment_name)

    plt.figure(figsize=(12,5))

    for line, var in zip(lines, line_names):
        l = np.asarray(line)
        mean = np.mean(l, axis = 0)
        standard_dev = np.std(l, axis = 0)

        plt.plot(mean, '-', label = var, linewidth = 1)
        plt.fill_between(lst_iter, mean - standard_dev, mean + standard_dev, alpha = 0.2)

    if experiment_name == "Accuracy": 
        plt.ylim((0.4 ,1.1))

    plt.xlabel("n iteration")
    plt.legend(loc = 'upper left')
    plt.title(title)

    # save image
    plt.savefig("{}.png".format(experiment_name))
    plt.close()


accuracy = []
loss = []

# variances = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
variances = [0.1, 0.25, 0.5, 1.0, 1.5, 5.0, 10.0, 20.0, 50.0]
# iterations = 1000 
# prefix = "" 
iterations = 5000
prefix = "5000_"

for var in variances:

    tmp_accuracy = []
    tmp_loss = []

    for idx in range(3):
        filename = "variance/{}{}_variance_{}.json".format(prefix ,idx+1, var)

        if os.path.exists(filename):

            with open(filename) as f:
                data = json.load(f)

            tmp_accuracy.append(data['accuracy'])
            tmp_loss.append(data['loss'])
    
    accuracy.append(tmp_accuracy)
    loss.append(tmp_loss)

save_loss_range_record(np.arange(iterations), accuracy, "asdf", "gpt2xcnn", "Accuracy", variances)
save_loss_range_record(np.arange(iterations), loss, "asdf", "gpt2xcnn", "Loss", variances)