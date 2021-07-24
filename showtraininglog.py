import os
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == "__main__":

    path = 'dataset/log'
    train_loss = np.load(path + '/Baby_datatrain_rmseloss_sumdot_addtime_False_addcnn_False_.npy')
    val_loss=np.load(path + '/Baby_dataval_rmseloss_sumdot_addtime_False_addcnn_False_.npy')
    x_train = np.linspace(0, np.shape(train_loss)[0], np.shape(train_loss)[0])
    x_val=np.linspace(0, np.shape(val_loss)[0], np.shape(val_loss)[0])
    plt.figure()
    l1,=plt.plot(x_train,train_loss,color='red',label='train_loss')
    l2,=plt.plot(x_val,val_loss,color='blue',label='val_loss')
    plt.legend(loc = 'upper right')
    plt.title('toy_mseloss_sumcatproduct_bestres_'+str(min(val_loss)))
    plt.show()
    print('loading train data')
