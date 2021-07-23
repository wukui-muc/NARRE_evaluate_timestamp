# -*- coding: utf-8 -*-

import numpy as np


class DefaultConfig:

    model = 'NARRE'

    # -------------base config-----------------------#
    use_gpu = True
    gpu_id = 0
    multi_gpu = False
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

    vocab_size = 50002
    word_dim = 300
    timestamp_size=778+1
    r_max_len = 202

    u_max_r = 13
    i_max_r = 24

    train_data_size = 51764
    test_data_size = 6471
    val_data_size = 6471

    user_num = 5541 + 2
    item_num = 3568 + 2

    batch_size = 128
    print_step = 100

class Toys_and_Games_data_Config(DefaultConfig):
    dataset = 'Toys_and_Games_data'

    def __init__(self):
        self.set_path('Toys_and_Games_data')

    vocab_size = 50002
    word_dim = 300
    timestamp_size=647+1
    r_max_len = 113

    u_max_r = 9
    i_max_r = 18

    train_data_size = 134087
    test_data_size = 16755
    val_data_size = 16755

    user_num = 19412 + 2
    item_num = 11924 + 2

    batch_size = 128
    print_step = 100

class Office_Products_data_Config(DefaultConfig):
    dataset = 'Office_Products_data'

    def __init__(self):
        self.set_path('Office_Products_data')

    vocab_size = 47808
    word_dim = 300
    timestamp_size=504+1
    r_max_len = 134

    u_max_r = 14
    i_max_r = 35

    train_data_size = 42611
    test_data_size = 5323
    val_data_size = 5324

    user_num = 4905 + 2
    item_num = 2420 + 2

    batch_size = 128
    print_step = 100

class Baby_data_Config(DefaultConfig):
    dataset = 'Baby_data'

    def __init__(self):
        self.set_path('Baby_data')

    vocab_size = 50002
    word_dim = 300
    timestamp_size=599+1
    r_max_len = 94

    u_max_r = 9
    i_max_r = 29

    train_data_size = 128644
    test_data_size = 16074
    val_data_size = 16074

    user_num = 19445 + 2
    item_num = 7050 + 2

    batch_size = 128
    print_step = 100
class Health_and_Personal_Care_data_Config(DefaultConfig):
    dataset = 'Health_and_Personal_Care_data'

    def __init__(self):
        self.set_path('Health_and_Personal_Care_data')

    vocab_size = 50002
    word_dim = 300
    timestamp_size=552+1
    r_max_len = 103

    u_max_r = 10
    i_max_r = 23

    train_data_size = 277109
    test_data_size = 34623
    val_data_size = 34623

    user_num = 38609 + 2
    item_num = 18534 + 2

    batch_size = 128
    print_step = 100
