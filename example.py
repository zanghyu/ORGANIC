# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from model.organic import ORGANIC
# import model.mol_methods as mm
# from collections import OrderedDict

# params = {
#     'MAX_LENGTH':     16,
#     'GEN_ITERATIONS':  1,
#     'DIS_EPOCHS':      1,
#     'DIS_BATCH_SIZE': 30,           # DISCRIMINATOR Batch Size
#     'GEN_BATCH_SIZE': 30,           # Generator Batch Size
#     'GEN_EMB_DIM':    32,
#     'DIS_EMB_DIM':    32,
#     'DIS_FILTER_SIZES': [5,  10,  15],
#     'DIS_NUM_FILTERS': [100, 100, 100]
# }

# model = ORGANIC('Sample', params=params)

# data = './data/trainingsets/toy.csv'
# ckpt = 'checkpoints/Sample.ckpt'
# model.load_training_set(data)
# # model.load_prev_training(ckpt)
# model.set_training_program(['logP'], [5])
# model.load_metrics()

# results = OrderedDict({'exp_name': 'Sample'})
# results['Batch'] = 30
# train_samples = mm.load_train_data(data)
# char_dict, ord_dict = mm.build_vocab(train_samples)
# gen_samples = model.generate_samples(30)
# mm.exa_compute_results(gen_samples, train_samples, ord_dict, results)
from model.organic import ORGANIC

organ_params = {
    'PRETRAIN_GEN_EPOCHS': 250, 'PRETRAIN_DIS_EPOCHS': 20, 'MAX_LENGTH': 60, 'LAMBDA': 0.5, "DIS_EPOCHS": 2, 'SAMPLE_NUM': 6400, 'WGAN': True, 'TBOARD_LOG': True}

# hyper-optimized parameters
disc_params = {"DIS_L2REG": 0.2, "DIS_EMB_DIM": 32, "DIS_FILTER_SIZES": [
    1, 2, 3, 4, 5, 8, 10, 15], "DIS_NUM_FILTERS": [50, 50, 50, 50, 50, 50, 50, 75], "DIS_DROPOUT": 0.75}

organ_params.update(disc_params)

model = ORGANIC('test', params=organ_params)
model.load_training_set('data/smallmols_sub.smi')
# model.load_prev_pretraining('pretrain_ckpt/qm9-5k_pretrain_ckpt')
model.set_training_program(
    ['novelty'], [100])
model.load_metrics()
# model.load_prev_training(ckpt='qm9-5k_20.ckpt')
model.train()
