
import os

class DirectoryGenerator:

    def __init__(self, time, config):
        self.time = time
        self.config = config

        self.check_roots()
        self.set_roots()

    def set_roots(self):
        roots_path = "./result/{}/{}".format(self.config.model_name, self.time)

        os.mkdir(roots_path)
        os.mkdir(roots_path + "/config")

        for root in ["/models", "/topography", "/eeg", "/history", "/stft", "/figure"]:
            os.mkdir(roots_path + "{}".format(root))
            for idx in range(self.config.rounds):
                os.mkdir(roots_path + "{}/{}".format(root, idx))

    def check_roots(self):
        if not os.path.exists("./result"):
            os.mkdir("./result")

        if not os.path.exists("./result/{}".format(self.config.model_name)):
            os.mkdir("./result/{}".format(self.config.model_name))

def is_logs_folder_exist(time: str = None, model_name: str = 'gpt2gan'):

    logs_dir = './logs'
    gpt2gan_dir = logs_dir + '/gpt2gan'
    gpt2wgan_dir = logs_dir + '/gpt2wgan'
    gpt2cgan_dir = logs_dir + '/gpt2cgan'
    gpt2xcnn_dir = logs_dir + '/gpt2xcnn'

    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    if not os.path.exists(gpt2gan_dir):
        os.mkdir(gpt2gan_dir)

    if not os.path.exists(gpt2wgan_dir):
        os.mkdir(gpt2wgan_dir)
    
    if not os.path.exists(gpt2cgan_dir):
        os.mkdir(gpt2cgan_dir)

    if not os.path.exists(gpt2xcnn_dir):
        os.mkdir(gpt2xcnn_dir)

    time_dir = ''

    if model_name == 'gpt2gan':
        time_dir = gpt2gan_dir + '/' + time
    elif model_name == 'gpt2wgan':
        time_dir = gpt2wgan_dir + '/' + time
    elif model_name == 'gpt2cgan':
        time_dir = gpt2cgan_dir + '/' + time
    elif model_name == 'gpt2xcnn':
        time_dir = gpt2xcnn_dir + '/' + time
    else:
        print("[ERROR] is_logs_folder_exist() from setup.py has wrong model name input")
    
    if not os.path.exists(time_dir):
        os.mkdir(time_dir)

def is_results_folder_exist(time: str = None, model_name: str = 'gpt2gan'):

    results_dir = './results'
    img_dir = results_dir +  '/img_results'
    loss_dir = results_dir + '/training_loss'
    gpt2gan_img_dir = img_dir + '/gpt2gan'
    gpt2gan_loss_dir = loss_dir + '/gpt2gan'
    gpt2wgan_img_dir = img_dir + '/gpt2wgan'
    gpt2wgan_loss_dir = loss_dir + '/gpt2wgan'
    gpt2cgan_img_dir = img_dir + '/gpt2cgan'
    gpt2cgan_loss_dir = loss_dir + '/gpt2cgan'
    gpt2xcnn_img_dir = img_dir + '/gpt2xcnn'
    gpt2xcnn_loss_dir = loss_dir + '/gpt2xcnn'
    

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    if not os.path.exists(loss_dir):
        os.mkdir(loss_dir)

    if not os.path.exists(gpt2gan_img_dir):
        os.mkdir(gpt2gan_img_dir)

    if not os.path.exists(gpt2gan_loss_dir):
        os.mkdir(gpt2gan_loss_dir)

    if not os.path.exists(gpt2wgan_img_dir):
        os.mkdir(gpt2wgan_img_dir)

    if not os.path.exists(gpt2wgan_loss_dir):
        os.mkdir(gpt2wgan_loss_dir)

    if not os.path.exists(gpt2cgan_img_dir):
        os.mkdir(gpt2cgan_img_dir)

    if not os.path.exists(gpt2cgan_loss_dir):
        os.mkdir(gpt2cgan_loss_dir)

    if not os.path.exists(gpt2xcnn_img_dir):
        os.mkdir(gpt2xcnn_img_dir)

    if not os.path.exists(gpt2xcnn_loss_dir):
        os.mkdir(gpt2xcnn_loss_dir)

    img_time_dir = ''
    loss_time_dir = ''

    if model_name == 'gpt2gan':
        img_time_dir = gpt2gan_img_dir + '/' + time
        loss_time_dir = gpt2gan_loss_dir + '/' + time
    elif model_name == 'gpt2wgan':
        img_time_dir = gpt2wgan_img_dir + '/' + time
        loss_time_dir = gpt2wgan_loss_dir + '/' + time
    elif model_name == 'gpt2cgan':
        img_time_dir = gpt2cgan_img_dir + '/' + time
        loss_time_dir = gpt2cgan_loss_dir + '/' + time
    elif model_name == 'gpt2xcnn':
        img_time_dir = gpt2xcnn_img_dir + '/' + time
        loss_time_dir = gpt2xcnn_loss_dir + '/' + time
    else:
        print("[ERROR] is_logs_folder_exist() from setup.py has wrong model name input")
    
    if not os.path.exists(img_time_dir):
        os.mkdir(img_time_dir)

    if not os.path.exists(loss_time_dir):
        os.mkdir(loss_time_dir)

def is_trained_model_folder_exist(time: str = None, model_name: str = 'gpt2gan'):

    trained_model_dir = './trained_model'
    gpt2gan_dir = trained_model_dir + '/gpt2gan'
    gpt2wgan_dir = trained_model_dir + '/gpt2wgan'
    gpt2cgan_dir = trained_model_dir + '/gpt2cgan'
    gpt2xcnn_dir = trained_model_dir + '/gpt2xcnn'

    if not os.path.exists(trained_model_dir):
        os.mkdir(trained_model_dir)

    if not os.path.exists(gpt2gan_dir):
        os.mkdir(gpt2gan_dir)

    if not os.path.exists(gpt2wgan_dir):
        os.mkdir(gpt2wgan_dir)

    if not os.path.exists(gpt2cgan_dir):
        os.mkdir(gpt2cgan_dir)

    if not os.path.exists(gpt2xcnn_dir):
        os.mkdir(gpt2xcnn_dir)

    time_dir = ''

    if model_name == 'gpt2gan':
        time_dir = gpt2gan_dir + '/' + time
    elif model_name == 'gpt2wgan':
        time_dir = gpt2wgan_dir + '/' + time
    elif model_name == 'gpt2cgan':
        time_dir = gpt2cgan_dir + '/' + time
    elif model_name == 'gpt2xcnn':
        time_dir = gpt2xcnn_dir + '/' + time
    else:
        print("[ERROR] is_logs_folder_exist() from setup.py has wrong model name input")
    
    if not os.path.exists(time_dir):
        os.mkdir(time_dir)

def check_folders(time: str = '', model_name: str = 'gpt2gan'):

    is_logs_folder_exist(time = time, model_name = model_name)
    is_results_folder_exist(time = time, model_name = model_name)
    is_trained_model_folder_exist(time = time, model_name = model_name)
