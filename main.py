from trainer import *
from data_loader import *
import json
import torch
import argparse

data_dir = {
    'train_tasks_in_train': '/train_tasks_in_train.json',
    'train_tasks': '/train_tasks.json',
    'test_tasks': '/test_tasks.json',
    'dev_tasks': '/dev_tasks.json',

    'rel2candidates_in_train': '/rel2candidates_in_train.json',
    'rel2candidates': '/rel2candidates_all.json',

    'e1rel_e2_in_train': '/e1rel_e2_in_train.json',
    'e1rel_e2': '/e1rel_e2.json',

    'ent2ids': '/ent2ids',
}


def get_params():
    args = argparse.ArgumentParser()
    args.add_argument("-data", "--dataset", default="NL27K", type=str)
    args.add_argument("-prefix", "--prefix", default="test", type=str, help='output folder name')
    args.add_argument("-path", "--data_path", default="NL27K/NL27K-N1", type=str)
    args.add_argument("-seed", "--seed", default=3407, type=int)
    args.add_argument("-few", "--few", default=3, type=int)
    args.add_argument("-nq", "--num_query", default=10, type=int)
    args.add_argument("-metric", "--metric", default="MRR", choices=["MRR", "Hits@1", "Hits@5", "Hits@10", "MSE", "MAE"])

    args.add_argument("-dim", "--embed_dim", default=500, type=int)
    args.add_argument("-bs", "--batch_size", default=128, type=int)
    args.add_argument("-lr", "--learning_rate", default=0.01, type=float)
    args.add_argument("-es_p", "--early_stopping_patience", default=30, type=int)

    args.add_argument("-epo", "--epoch", default=15000, type=int)
    args.add_argument("-prt_epo", "--print_epoch", default=100, type=int)
    args.add_argument("-eval_epo", "--eval_epoch", default=100, type=int)
    args.add_argument("-ckpt_epo", "--checkpoint_epoch", default=1000, type=int)

    args.add_argument("-p", "--dropout_p", default=0.5, type=float)

    args.add_argument("-d", "--delta", default=5, type=float)

    args.add_argument("-m", "--margin", default=1, type=float)
    args.add_argument("-g", "--gamma", default=1, type=int)
    
    args.add_argument("-a", "--alpha", default=1.0, type=float)
    args.add_argument("-b", "--beta", default=1.0, type=float)

    args.add_argument("-step", "--step", default="train", type=str, choices=['train', 'test', 'dev'])
    args.add_argument("-log_dir", "--log_dir", default="log", type=str)
    args.add_argument("-state_dir", "--state_dir", default="state", type=str)
    args.add_argument("-eval_ckpt", "--eval_ckpt", default=None, type=str)

    args.add_argument("-gpu", "--device", default=6, type=int)

    # ----------------------------
    args = args.parse_args()
    params = {}
    for k, v in vars(args).items():
        params[k] = v
    params['device'] = torch.device('cuda:'+str(args.device))

    return params


if __name__ == '__main__':
    params = get_params()
    print("---------Parameters---------")
    for k, v in params.items():
        print(k + ': ' + str(v))
    print("----------------------------")
    # control random seed
    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    # select the dataset
    for k, v in data_dir.items():
        data_dir[k] = params['data_path'] + v

    dataset = dict()
    dataset['train_tasks'] = json.load(open(data_dir['train_tasks']))
    print("loading test_tasks ... ...")
    dataset['test_tasks'] = json.load(open(data_dir['test_tasks']))
    print("loading dev_tasks ... ...")
    dataset['dev_tasks'] = json.load(open(data_dir['dev_tasks']))
    print("loading rel2candidates ... ...")
    dataset['rel2candidates'] = json.load(open(data_dir['rel2candidates']))
    print("loading e1rel_e2 ... ...")
    dataset['e1rel_e2'] = json.load(open(data_dir['e1rel_e2']))
    print("loading ent2id ... ...")
    dataset['ent2id'] = json.load(open(data_dir['ent2ids']))
    print("----------------------------")

    # data_loader
    train_data_loader = DataLoader(dataset, params, step='train')
    dev_data_loader = DataLoader(dataset, params, step='dev')
    test_data_loader = DataLoader(dataset, params, step='test')
    data_loaders = [train_data_loader, dev_data_loader, test_data_loader]

    # trainer
    trainer = Trainer(data_loaders, dataset, params)

    if params['step'] == 'train':
        trainer.train()
        print("test")
        print(params['prefix'])
        trainer.reload()
        trainer.eval(istest=True)

    elif params['step'] == 'test':
        print(params['prefix'])
        trainer.eval(istest=True)

    elif params['step'] == 'dev':
        print(params['prefix'])
        trainer.eval(istest=False)

