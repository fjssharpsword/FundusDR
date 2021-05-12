import os

config = {
            'CKPT_PATH': '/data/pycode/Thesis/ckpt/',
            'log_path':  '/data/pycode/Thesis/log/',
            'img_path': '/data/pycode/Thesis/imgs/',
            'CUDA_VISIBLE_DEVICES': "7",
            'MAX_EPOCHS': 50,
            'BATCH_SIZE': 16, 
            'TRAN_SIZE': 256
         } 

#config for dataset
CLASS_NAMES = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
N_CLASSES = len(CLASS_NAMES)

