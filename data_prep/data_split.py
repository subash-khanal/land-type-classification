#This script creates train/val/test split csv files for each dataset type that we plan to experiment on.

from ..config import dataconfig
import os
import pandas as pd
from collections import Counter
import random

datasets = ["NWPU_RESISC45", "RSSCN7","UCMerced_LandUse"]

def get_train_val_test(ids):
    random.shuffle(ids)
    train_ids = ids[:int(0.8*len(ids))]
    val_ids = ids[int(0.8*len(ids)):int(0.9*len(ids))]
    test_ids = ids[int(0.9*len(ids)):]

    return train_ids, val_ids, test_ids

for dataset in datasets:
    data_path = dataconfig[dataset]
    data_classes = dataconfig[dataset+"_classes"]
    train_ids_final = []
    train_classes_final = []

    val_ids_final = []
    val_classes_final = []

    test_ids_final = []
    test_classes_final = []

    for c in data_classes:
        class_img_ids = os.listdir(os.path.join(data_path,c))
        train_ids, val_ids, test_ids = get_train_val_test(ids=class_img_ids)
        
        train_ids_final = train_ids_final + train_ids
        val_ids_final = val_ids_final + val_ids
        test_ids_final = test_ids_final + test_ids

        train_classes_final = train_classes_final + [c]*len(train_ids)
        val_classes_final = val_classes_final + [c]*len(val_ids)
        test_classes_final = test_classes_final + [c]*len(test_ids)

        print("Sanity check for ",(dataset, c))
        print(set(train_ids).intersection(set(val_ids)))
        print(set(train_ids).intersection(set(test_ids)))
        print(set(val_ids).intersection(set(test_ids)))

    train_df = pd.DataFrame(columns=["id","class"])
    val_df = pd.DataFrame(columns=["id","class"])
    test_df = pd.DataFrame(columns=["id","class"])

    train_df['id'] = train_ids_final
    train_df['class'] = train_classes_final

    val_df['id'] = val_ids_final
    val_df['class'] = val_classes_final

    test_df['id'] = test_ids_final
    test_df['class'] = test_classes_final

    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = val_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    print("train/val/test splits size:",(len(train_df),len(val_df),len(test_df)))
    print("class distribution of train",Counter(train_df['class']))
    print("class distribution of val",Counter(val_df['class']))
    print("class distribution of test",Counter(test_df['class']))
    
    train_df.to_csv(os.path.join(data_path,dataset+"_train.csv"))
    val_df.to_csv(os.path.join(data_path,dataset+"_val.csv"))
    test_df.to_csv(os.path.join(data_path,dataset+"_test.csv"))


    




