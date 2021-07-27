# -*- coding: utf-8 -*-

import numpy as np


class DefaultConfig:

    model = 'NARRE'

    # -------------base config-----------------------#
    use_gpu = True
    gpu_id = 0
    multi_gpu = True
    gpu_ids = [0,1]

    seed = 2019
    num_epochs = 100
    num_workers = 0

    optimizer = 'Adam'
    weight_decay = 1e-3  # optimizer rameteri
    lr = 2e-3
    loss_method = 'rmse'
    drop_out = 0.5

    use_word_embedding = True

    id_emb_size = 32
    query_mlp_size = 128
    fc_dim = 32

    doc_len = 500
    filters_num = 100
    kernel_size = 3

    num_fea = 2  # id feature, review feature, doc feature
    use_review = True
    use_doc = True
    self_att = False

    r_id_merge = 'sum'  # review and ID feature
    ui_merge = 'dot'  # cat/add/dot
    output = 'lfm'  # 'fm', 'lfm', 'other: sum the ui_feature'

    fine_step = False  # save mode in step level, defualt in epoch
    pth_path = "checkpoints/*.pth"  # the saved pth path for test

    continue_train=False
    print_opt = 'default'

    addtime = False
    addcnn  = False
    rs_drop = False
    #
    def set_path(self, name):
        '''
        specific
        '''
        self.data_root = f'./dataset/{name}'
        prefix = f'{self.data_root}/train'

        self.user_list_path = f'{prefix}/userReview2Index.npy'
        self.item_list_path = f'{prefix}/itemReview2Index.npy'

        self.user2itemid_path = f'{prefix}/user_item2id.npy'
        self.item2userid_path = f'{prefix}/item_user2id.npy'

        self.user_doc_path = f'{prefix}/userDoc2Index.npy'
        self.item_doc_path = f'{prefix}/itemDoc2Index.npy'

        #wukui
        self.userreview_timelist_path=f'{prefix}/userreview_timelist.npy'
        self.itemreview_timelist_path=f'{prefix}/itemreview_timelist.npy'

        self.vocab_size_path= f'{self.data_root}/vocab_size.npy'
        self.timestamp_size_path= f'{self.data_root}/timestamp_size.npy'
        self.r_max_len_path= f'{self.data_root}/r_max_len.npy'
        self.u_max_r_path= f'{self.data_root}/u_max_r.npy'
        self.i_max_r_path= f'{self.data_root}/i_max_r.npy'
        self.train_data_size_path= f'{self.data_root}/train_data_size.npy'
        self.test_data_size_path= f'{self.data_root}/test_data_size.npy'
        self.val_data_size_path= f'{self.data_root}/val_data_size.npy'
        self.user_num_path= f'{self.data_root}/user_num.npy'
        self.item_num_path= f'{self.data_root}/item_num.npy'

        self.w2v_path = f'{prefix}/w2v.npy'

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        print("load npy from dist...")
        self.users_review_list = np.load(self.user_list_path, encoding='bytes')
        self.items_review_list = np.load(self.item_list_path, encoding='bytes')
        self.user2itemid_list = np.load(self.user2itemid_path, encoding='bytes')
        self.item2userid_list = np.load(self.item2userid_path, encoding='bytes')
        self.user_doc = np.load(self.user_doc_path, encoding='bytes')
        self.item_doc = np.load(self.item_doc_path, encoding='bytes')

        #wukui
        self.userreview_timelist = np.load(self.userreview_timelist_path, encoding='bytes')
        self.itemreview_timelist = np.load(self.itemreview_timelist_path, encoding='bytes')

        self.vocab_size = int(np.load(self.vocab_size_path, encoding='bytes'))
        self.timestamp_size = int(np.load(self.timestamp_size_path, encoding='bytes'))+1
        self.r_max_len = int(np.load(self.r_max_len_path, encoding='bytes'))
        self.u_max_r = int(np.load(self.u_max_r_path, encoding='bytes'))

        self.i_max_r = int(np.load(self.i_max_r_path, encoding='bytes'))
        self.train_data_size = int(np.load(self.train_data_size_path, encoding='bytes'))
        self.test_data_size = int(np.load(self.test_data_size_path, encoding='bytes'))
        self.val_data_size = int(np.load(self.val_data_size_path, encoding='bytes'))
        self.user_num = int(np.load(self.user_num_path, encoding='bytes'))+2
        self.item_num = int(np.load(self.item_num_path, encoding='bytes'))+2
        #

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'user_list' and k != 'item_list':
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


class Digital_Music_data_Config(DefaultConfig):
    dataset = 'Digital_Music_data'

    def __init__(self):
        self.set_path('Digital_Music_data')
    #原始格式
    # vocab_size = 50002
    # timestamp_size=196+1
    # u_max_r = 13
    # i_max_r = 23

    # train_data_size = 48268
    # test_data_size = 8990
    # val_data_size = 8990
    #
    # user_num = 5541 + 2
    # item_num = 3568 + 2
    word_dim = 300
    vocab_size = 0
    timestamp_size=0
    r_max_len=0
    u_max_r = 0
    i_max_r = 0

    train_data_size = 0
    test_data_size = 0
    val_data_size = 0

    user_num = 0
    item_num = 0

    batch_size = 128
    print_step = 100

class Toys_and_Games_data_Config(DefaultConfig):
    dataset = 'Toys_and_Games_data'

    def __init__(self):
        self.set_path('Toys_and_Games_data')

    word_dim = 300
    vocab_size = 0
    timestamp_size=0
    r_max_len=0

    u_max_r = 0
    i_max_r = 0

    train_data_size = 0
    test_data_size = 0
    val_data_size = 0

    user_num = 0
    item_num = 0
    batch_size = 128
    print_step = 100

class Office_Products_data_Config(DefaultConfig):
    dataset = 'Office_Products_data'

    def __init__(self):
        self.set_path('Office_Products_data')

    word_dim = 300
    vocab_size = 0
    timestamp_size=0
    r_max_len=0

    u_max_r = 0
    i_max_r = 0

    train_data_size = 0
    test_data_size = 0
    val_data_size = 0

    user_num = 0
    item_num = 0
    batch_size = 128
    print_step = 100

class Baby_data_Config(DefaultConfig):
    dataset = 'Baby_data'

    def __init__(self):
        self.set_path('Baby_data')
    word_dim = 300
    vocab_size = 0
    timestamp_size=0
    r_max_len=0

    u_max_r = 0
    i_max_r = 0

    train_data_size = 0
    test_data_size = 0
    val_data_size = 0

    user_num = 0
    item_num = 0

    batch_size = 64
    print_step = 100
class Health_and_Personal_Care_data_Config(DefaultConfig):
    dataset = 'Health_and_Personal_Care_data'

    def __init__(self):
        self.set_path('Health_and_Personal_Care_data')
    word_dim = 300
    vocab_size = 0
    timestamp_size=0
    r_max_len=0

    u_max_r = 0
    i_max_r = 0

    train_data_size = 0
    test_data_size = 0
    val_data_size = 0

    user_num = 0
    item_num = 0
    batch_size = 128
    print_step = 100
