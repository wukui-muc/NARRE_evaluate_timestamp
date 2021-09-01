# NARRE_evaluate_timestamp
This project is a attempt to embed timestamp into NARRE model.

#process dataset   
python data_pro.py Automotive_5.json month False False 0.9 0.7 50000  

Command line parameter:  
Automotive_5.json (dataset name)  
month (timestamp format:month/week)  
False (remove stopwords: True/False)  
False (splid dataset in the order of time: True/Flase)  
0.9 (P_REVIEW)  
0.7 (MAX_DF)  
50000 (MAX_VOCAB)  

#train     
python main.py train --dataset=Automotive_data --addtime=True --addcnn=True --num_epochs=70 --r_id_merge=cat --ui_merge=cat  

Thanks for ShomyLiu's work, I use the toolbox from here:  
  https://github.com/ShomyLiu/Neu-Review-Rec  

@inproceedings{liu2019nrpa,
  title={NRPA: Neural Recommendation with Personalized Attention},
  author={Liu, Hongtao and Wu, Fangzhao and Wang, Wenjun and Wang, Xianchen and Jiao, Pengfei and Wu, Chuhan and Xie, Xing},
  booktitle={Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1233--1236},
  year={2019}
}
