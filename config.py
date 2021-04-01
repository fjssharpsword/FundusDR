import os

config = {
            'CKPT_PATH': '/data/pycode/FundusDR/ckpt/',
            'log_path':  '/data/pycode/FundusDR/log/',
            'img_path': '/data/pycode/FundusDR/imgs/',
            'CUDA_VISIBLE_DEVICES': "0,1,2,3,4,5,6,7", #'6,7'
            'MAX_EPOCHS': 10, #50
            'BATCH_SIZE': 32, #128
            'TRAN_SIZE': 256
         } 

#config for dataset
CLASS_NAMES = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
N_CLASSES = len(CLASS_NAMES)

