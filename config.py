import os

config = {
            'CKPT_PATH': '/data/pycode/FundusDR/ckpt/',
            'log_path':  '/data/pycode/FundusDR/log/',
            'img_path': '/data/pycode/FundusDR/imgs/',
            'CUDA_VISIBLE_DEVICES': '6,7', #"0,1,2,3,4,5,6,7",
            'MAX_EPOCHS': 50, #20
            'BATCH_SIZE': 8, #128
            'TRAN_SIZE': 256
         } 

#config for dataset
CLASS_NAMES = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
N_CLASSES = len(CLASS_NAMES)

