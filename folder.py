
import os

from datetime import datetime

def is_logs_folder_exist(time: str = None, model_name: str = 'gan'):

    logs_dir = './logs'
    gpt2gan_dir = logs_dir + '/gpt2gan'
    gpt2wgan_dir = logs_dir + '/gpt2wgan'

    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    if not os.path.exists(gpt2gan_dir):
        os.mkdir(gpt2gan_dir)

    if not os.path.exists(gpt2wgan_dir):
        os.mkdir(gpt2wgan_dir)

    time_dir = ''

    if model_name == 'gan':
        time_dir = gpt2gan_dir + '/' + time
    elif model_name == 'wgan':
        time_dir = gpt2wgan_dir + '/' + time
    else:
        print("[ERROR] is_logs_folder_exist() from setup.py has wrong model name input")
    
    if not os.path.exists(time_dir):
        os.mkdir(time_dir)

def is_results_folder_exist(time: str = None, model_name: str = 'gan'):

    results_dir = './results'
    img_dir = results_dir +  '/img_results'
    loss_dir = results_dir + '/training_loss'
    gpt2gan_dir = img_dir + '/gpt2gan'
    gpt2wgan_dir = loss_dir + '/gpt2wgan'

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    if not os.path.exists(loss_dir):
        os.mkdir(loss_dir)

    if not os.path.exists(gpt2gan_dir):
        os.mkdir(gpt2gan_dir)

    if not os.path.exists(gpt2wgan_dir):
        os.mkdir(gpt2wgan_dir)

    time_dir = ''

    if model_name == 'gan':
        time_dir = gpt2gan_dir + '/' + time
    elif model_name == 'wgan':
        time_dir = gpt2wgan_dir + '/' + time
    else:
        print("[ERROR] is_logs_folder_exist() from setup.py has wrong model name input")
    
    if not os.path.exists(time_dir):
        os.mkdir(time_dir)

def is_trained_model_folder_exist(time: str = None, model_name: str = 'gan'):

    trained_model_dir = './trained_model'
    gpt2gan_dir = trained_model_dir + '/gpt2gan'
    gpt2wgan_dir = trained_model_dir + '/gpt2wgan'

    if not os.path.exists(trained_model_dir):
        os.mkdir(trained_model_dir)

    if not os.path.exists(gpt2gan_dir):
        os.mkdir(gpt2gan_dir)

    if not os.path.exists(gpt2wgan_dir):
        os.mkdir(gpt2wgan_dir)

    time_dir = ''

    if model_name == 'gan':
        time_dir = gpt2gan_dir + '/' + time
    elif model_name == 'wgan':
        time_dir = gpt2wgan_dir + '/' + time
    else:
        print("[ERROR] is_logs_folder_exist() from setup.py has wrong model name input")
    
    if not os.path.exists(time_dir):
        os.mkdir(time_dir)

def check_folders(time: str = '', model_name: str = 'gan'):

    is_logs_folder_exist(time = time, model_name = model_name)
    is_results_folder_exist(time = time, model_name = model_name)
    is_trained_model_folder_exist(time = time, model_name = model_name)
