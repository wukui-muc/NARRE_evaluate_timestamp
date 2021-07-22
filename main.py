# -*- encoding: utf-8 -*-
import time
import random
import math
import fire
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ReviewData
from framework import Model
import models
import config
import logging
import sys

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a",encoding='utf-8')     #文件权限为'a'，追加模式

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def train(**kwargs):
    logging.basicConfig(filename='logger.log', format='%(asctime)s - %(levelname)s - %(message)s',level=logging.WARNING)
    logger = logging.getLogger(__name__)   	#定义一次就可以，其他地方需要调用logger,只需要直接使用logger就行了
    logger.setLevel(level=logging.INFO)  	#定义过滤级别
    filehandler = logging.FileHandler("log.txt")  	# Handler用于将日志记录发送至合适的目的地，如文件、终端等
    filehandler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    filehandler.setFormatter(formatter)

    console = logging.StreamHandler()  		#日志信息显示在终端terminal
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    logger.addHandler(filehandler)
    logger.addHandler(console)

    logger.info("Start log")
    log_save_folder='dataset'
    if not os.path.exists(log_save_folder + '/log'):
        os.makedirs(log_save_folder + '/log')

    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if ~opt.multi_gpu and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if opt.multi_gpu:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)
    if opt.multi_gpu:
        if model.module.net.num_fea != opt.num_fea:
            raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")
    else:
        if model.net.num_fea != opt.num_fea:
            raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")
    if opt.continue_train:
        # model.load(opt.pth_path)
        model.load('checkpoints/NARRE_Digital_Music_data_addtime_True_addcnn_True_default.pth')
        print("load checkpoints model")
    # 3 data
    train_data = ReviewData(opt.data_root, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    val_data = ReviewData(opt.data_root, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f'train data: {len(train_data)}; test data: {len(val_data)}')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # training
    print("start training....")
    min_loss = 1e+10
    best_res = 1e+10
    mse_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    smooth_mae_func = nn.SmoothL1Loss()
    Train_rmse_log=[]
    Val_rmse_log=[]
    message='_addtime_'+str(opt.addtime)+'_addcnn_'+str(opt.addcnn)+'_bestres_'+str(best_res)
    logger.info('evaluate_time_'+opt.dataset+message)
    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        total_maeloss = 0.0
        model.train()
        print(f"{now()}  Epoch {epoch}...")
        for idx, (train_datas, scores) in enumerate(train_data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            train_datas = unpack_input(opt, train_datas)

            optimizer.zero_grad()
            output = model(train_datas)
            mse_loss = mse_func(output, scores)
            total_loss += mse_loss.item() * len(scores)

            mae_loss = mae_func(output, scores)
            total_maeloss += mae_loss.item()
            smooth_mae_loss = smooth_mae_func(output, scores)
            if opt.loss_method == 'mse':
                loss = mse_loss
            if opt.loss_method == 'rmse':
                loss = torch.sqrt(mse_loss) / 2.0
            if opt.loss_method == 'mae':
                loss = mae_loss
            if opt.loss_method == 'smooth_mae':
                loss = smooth_mae_loss
            loss.backward()
            optimizer.step()
            if opt.fine_step:
                if idx % opt.print_step == 0 and idx > 0:
                    print("\t{}, {} step finised;".format(now(), idx))
                    val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
                    fine_val_rmse=np.sqrt(val_mse)
                    if val_loss < min_loss:
                        if opt.multi_gpu:
                            model.module.save(name=opt.dataset, opt=opt.print_opt)
                            min_loss = val_loss
                            print("\tmodel save")
                        else:
                            model.save(name=opt.dataset, opt=opt.print_opt)
                            min_loss = val_loss
                            print("\tmodel save")
                    if val_loss > min_loss:
                        best_res = min_loss

        scheduler.step()
        mse = total_loss * 1.0 / len(train_data)
        train_rmse=np.sqrt(mse)
        Train_rmse_log.append(train_rmse)
        print(f"\ttrain data: loss:{total_loss:.4f}, train_rmse: {train_rmse:.4f};")

        val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
        val_rmse=np.sqrt(val_mse)
        Val_rmse_log.append(val_rmse)
        if val_loss < min_loss:
            model_name=opt.dataset+'_addtime_'+str(opt.addtime)+'_addcnn_'+str(opt.addcnn)
            if opt.multi_gpu:
                model.module.save(name=model_name, opt=opt.print_opt)
                min_loss = val_loss
                print("model save")
            else:
                model.save(name=model_name, opt=opt.print_opt)
                min_loss = val_loss
                print("model save")
        if val_rmse < best_res:
            best_res = val_rmse
        print(f"\t best_rmse:{best_res:.4f};")
        print("*"*30)

        trainfilename=str(opt.dataset)+'train_'+str(opt.loss_method)+'loss_'+str(opt.r_id_merge)+str(opt.ui_merge)\
                      +"_addtime_"+str(opt.addtime)+"_addcnn_"+str(opt.addcnn)+"_"+'.npy'
        valfilename=str(opt.dataset)+'val_'+str(opt.loss_method)+'loss_'+str(opt.r_id_merge)+str(opt.ui_merge) \
                    +"_addtime_"+str(opt.addtime)+"_addcnn_"+str(opt.addcnn)+"_"'.npy'

        np.save(f"{log_save_folder}/log/{trainfilename}", Train_rmse_log)
        np.save(f"{log_save_folder}/log/{valfilename}", Val_rmse_log)
    print("----"*20)
    print(f"{now()} {opt.dataset} {opt.print_opt} best_rmse:  {best_res}")
    message='_addtime_'+str(opt.addtime)+'_addtcnn_'+str(opt.addcnn)+'_bestres_'+str(best_res)
    logger.info('evaluate_time_'+opt.dataset+message)
    print("----"*20)


def test(**kwargs):

    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    assert(len(opt.pth_path) > 0)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)
    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    model.load(opt.pth_path)
    print(f"load model: {opt.pth_path}")
    test_data = ReviewData(opt.data_root, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"{now()}: test in the test datset")
    predict_loss, test_mse, test_mae = predict(model, test_data_loader, opt)


def predict(model, data_loader, opt):
    total_loss = 0.0
    total_maeloss = 0.0
    model.eval()
    with torch.no_grad():
        for idx, (test_data, scores) in enumerate(data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            test_data = unpack_input(opt, test_data)

            output = model(test_data)
            mse_loss = torch.sum((output-scores)**2)
            total_loss += mse_loss.item()

            mae_loss = torch.sum(abs(output-scores))
            total_maeloss += mae_loss.item()

    data_len = len(data_loader.dataset)
    mse = total_loss * 1.0 / data_len
    mae = total_maeloss * 1.0 / data_len
    print(f"\tevaluation reslut: mse: {mse:.4f}; rmse: {math.sqrt(mse):.4f};")
    model.train()
    return total_loss, mse, mae


def unpack_input(opt, x):

    uids, iids,time= list(zip(*x))
    uids = list(uids)
    iids = list(iids)

    user_reviews = opt.users_review_list[uids]
    user_item2id = opt.user2itemid_list[uids]  # 检索出该user对应的item id

    user_doc = opt.user_doc[uids]

    item_reviews = opt.items_review_list[iids]
    item_user2id = opt.item2userid_list[iids]  # 检索出该item对应的user id
    item_doc = opt.item_doc[iids]

    userreview_time=opt.userreview_timelist[uids]
    itemreview_time=opt.itemreview_timelist[iids]
    data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc,userreview_time,itemreview_time]
    data = list(map(lambda x: torch.LongTensor(x).cuda(), data))

    return data


if __name__ == "__main__":
    fire.Fire()
