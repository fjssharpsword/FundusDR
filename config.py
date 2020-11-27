import os

config = {
            'CKPT_PATH': '/data/pycode/FundusDR/models/',
            'log_path':  '/data/pycode/FundusDR/log/',
            'img_path': '/data/pycode/FundusDR/imgs/',
            'CUDA_VISIBLE_DEVICES': "0,1,2,3,4,5,6,7",
            'MAX_EPOCHS': 20,
            'BATCH_SIZE': 512,#128
            'TRAN_SIZE': 256,
            'TRAN_CROP': 224,
            'sizeX': 20, 
            'sizeY': 20
         } 

#config for dataset
CLASS_NAMES = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
N_CLASSES = len(CLASS_NAMES)
CLASS_PROB= {  0: 0.6975, #'No DR',
               1: 0.0578, #'Mild DR',
               2: 0.1104, #'Moderate DR', 
               3: 0.0260, #'Severe DR',
               4: 0.0104  #'Proliferative DR'
            }

