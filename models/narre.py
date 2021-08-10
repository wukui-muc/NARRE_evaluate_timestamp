# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import pickle
import pandas as pd
class NARRE(nn.Module):
    '''
    NARRE: WWW 2018
    '''
    def __init__(self, opt):
        super(NARRE, self).__init__()
        self.opt = opt
        self.num_fea = 2  # ID + Review

        self.user_net = Net(opt, 'user')
        self.item_net = Net(opt, 'item')

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc,userreview_time,itemreview_time = datas
        u_fea = self.user_net(user_reviews, uids, user_item2id,userreview_time)
        # if not self.training:
            # i_fea,data_att = self.item_net(item_reviews, iids, item_user2id,itemreview_time)
            # return u_fea, i_fea,data_att

        i_fea = self.item_net(item_reviews, iids, item_user2id,itemreview_time)
        return u_fea, i_fea


class Net(nn.Module):
    def __init__(self, opt, uori='user'):
        super(Net, self).__init__()
        self.opt = opt

        if uori == 'user':
            id_num = self.opt.user_num
            ui_id_num = self.opt.item_num
        else:
            id_num = self.opt.item_num
            ui_id_num = self.opt.user_num

        self.id_embedding = nn.Embedding(id_num, self.opt.id_emb_size)  # user/item num * 32
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # vocab_size * 300
        self.u_i_id_embedding = nn.Embedding(ui_id_num, self.opt.id_emb_size)

        self.review_linear = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)
        self.id_linear = nn.Linear(self.opt.id_emb_size, self.opt.id_emb_size, bias=False)
        self.attention_linear = nn.Linear(self.opt.id_emb_size, 1)
        self.fc_layer = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)

        if self.opt.addcnn:
            if self.opt.addtime:
                self.time_linear=nn.Linear(self.opt.id_emb_size, self.opt.id_emb_size,bias=False)
                self.u_i_time_embedding=nn.Embedding(self.opt.timestamp_size,self.opt.id_emb_size)
                self.blannce_cn=nn.Conv2d(3,1,(1,1))#timestamp
            else:
                self.blannce_cn=nn.Conv2d(2,1,(1,1))#non-timestamp

        self.cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))
        self.rs_drop=nn.Dropout(self.opt.drop_out)

        self.dropout = nn.Dropout(self.opt.drop_out)
        self.reset_para()

    def forward(self, reviews, ids, ids_list,time_list):
        # --------------- word embedding ----------------------------------

        # if not self.training:
        #     bs, r_num, r_len = reviews.size()
        #     if r_num==self.opt.i_max_r: #判断是否为item 网络
        #         file = open(os.path.join(self.opt.data_root, 'wordindex'), 'rb')
        #         file=pickle.load(file)
        #         word=list(file.keys())
        #         word_index=list(file.values())
        #         reviews_idx=reviews.tolist()
        #         review_text= np.array([word[z] for x in reviews_idx for y in x for z in y ])
        #
        #         review_text=review_text.reshape(bs,r_num,r_len)
        #         file = open(os.path.join(self.opt.data_root, 'userindex'), 'rb')
        #         file = pickle.load(file)
        #         userID=list(file.keys())
        #         file = open(os.path.join(self.opt.data_root, 'itemindex'), 'rb')
        #         file = pickle.load(file)
        #         itemID = list(file.keys())

        reviews = self.word_embs(reviews)  # size * 300

        bs, r_num, r_len, wd = reviews.size()
        reviews = reviews.view(-1, r_len, wd)

        id_emb = self.id_embedding(ids)

        u_i_id_emb = self.u_i_id_embedding(ids_list)
        if self.opt.addtime:
            u_i_time_emb=self.u_i_time_embedding(time_list)


        # --------cnn for review--------------------
        fea = F.relu(self.cnn(reviews.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        fea = F.max_pool1d(fea, fea.size(2)).squeeze(2)
        fea = fea.view(-1, r_num, fea.size(1))

        # ------------------linear attention-------------------------------
        # rs_mix = F.relu(self.review_linear(fea) + self.id_linear(F.relu(u_i_id_emb)))
        if self.opt.addcnn:
            if self.opt.addtime:
                rs_mix=torch.cat([self.review_linear(fea).unsqueeze(2),self.id_linear(u_i_id_emb).unsqueeze(2),self.time_linear(u_i_time_emb).unsqueeze(2)],dim=2)
                bs,r_num,channel,emb_dim=rs_mix.size()
                rs_mix=rs_mix.view(-1,channel,emb_dim)
                rs_mix = F.relu(self.blannce_cn(rs_mix.unsqueeze(2)).squeeze(1).squeeze(1))
                rs_mix=rs_mix.view(-1,r_num,rs_mix.size(1))
                if self.opt.rs_drop:
                    rs_mix=self.rs_drop(rs_mix)

            else:
                rs_mix=torch.cat([self.review_linear(fea).unsqueeze(2),self.id_linear(u_i_id_emb).unsqueeze(2)],dim=2)
                bs,r_num,channel,emb_dim=rs_mix.size()
                rs_mix=rs_mix.view(-1,channel,emb_dim)
                rs_mix = F.relu(self.blannce_cn(rs_mix.unsqueeze(2)).squeeze(1).squeeze(1))
                rs_mix=rs_mix.view(-1,r_num,rs_mix.size(1))
        else:
            rs_mix = F.relu(self.review_linear(fea) + self.id_linear(u_i_id_emb))

        att_score = self.attention_linear(rs_mix)
        att_weight = F.softmax(att_score, 1)
        r_fea = fea * att_weight
        r_fea = r_fea.sum(1)
        r_fea = self.dropout(r_fea)

        # if self.training and r_num==self.opt.i_max_r:
        #     att=np.array(att_score.cpu())
        #     max_att=[np.where(x==max(x))[0] for x in att]
        #     min_att=[]
        #     for x in att:
        #         for i in range(0,len(x)):
        #             if x[i]<0:
        #                 x[i]=1000
        #         min_att.append(np.where(x==min(x))[0])
        #     itemid =[itemID[x] for x in ids.cpu()]
        #     max_att_review=[review_text[i][max_att[i]] for i in range(0,128)]
        #     min_att_review=[review_text[i][min_att[i]] for i in range(0,128)]
        #     data_frame = {'item_id': pd.Series(itemid),'maxatt_id':pd.Series(max_att),'minatt_id':pd.Series(min_att),'max_att_review':pd.Series(max_att_review),'min_att_review':pd.Series(min_att_review)}
        #     return torch.stack([id_emb, self.fc_layer(r_fea)], 1),data_frame

        return torch.stack([id_emb, self.fc_layer(r_fea)], 1)

    def reset_para(self):
        nn.init.xavier_normal_(self.cnn.weight)
        nn.init.constant_(self.cnn.bias, 0.1)

        nn.init.uniform_(self.id_linear.weight, -0.1, 0.1)

        nn.init.uniform_(self.review_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.review_linear.bias, 0.1)

        nn.init.uniform_(self.attention_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.attention_linear.bias, 0.1)

        nn.init.uniform_(self.fc_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer.bias, 0.1)
        if self.opt.addtime:
            nn.init.uniform_(self.time_linear.weight,-0.1,0.1)


        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)
        nn.init.uniform_(self.id_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.u_i_id_embedding.weight, a=-0.1, b=0.1)
        if self.opt.addtime:
            nn.init.uniform_(self.u_i_time_embedding.weight,a=-0.1,b=0.1)
