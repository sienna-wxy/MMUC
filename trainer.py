import time

from models import *
from tensorboardX import SummaryWriter
import os
import sys
import torch
import shutil
import logging


class Trainer:
    def __init__(self, data_loaders, dataset, parameter):
        self.parameter = parameter
        # data loader
        self.train_data_loader = data_loaders[0]
        self.dev_data_loader = data_loaders[1]
        self.test_data_loader = data_loaders[2]
        # parameters
        self.few = parameter['few']
        self.num_query = parameter['num_query']
        self.batch_size = parameter['batch_size']
        self.learning_rate = parameter['learning_rate']
        self.early_stopping_patience = parameter['early_stopping_patience']
        # epoch
        self.epoch = parameter['epoch']
        self.print_epoch = parameter['print_epoch']
        self.eval_epoch = parameter['eval_epoch']
        self.checkpoint_epoch = parameter['checkpoint_epoch']
        # device
        self.device = parameter['device']

        self.MMUC = MMUC(dataset, parameter)
        self.MMUC.to(self.device)
        # optimizer
        self.optimizer = torch.optim.Adam(self.MMUC.parameters(), self.learning_rate)

        self.alpha = parameter['alpha']
        self.beta = parameter['beta']

        # tensorboard log writer
        if parameter['step'] == 'train':
            self.writer = SummaryWriter(os.path.join(parameter['log_dir'], parameter['prefix']))

        self.state_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'])
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        self.ckpt_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'], 'checkpoint')
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.state_dict_file = ''

        # logging
        logging_dir = os.path.join(self.parameter['log_dir'], self.parameter['prefix'], 'res.log')
        logging.basicConfig(filename=logging_dir, level=logging.INFO, format="%(asctime)s - %(message)s")

        # load state_dict and params
        if parameter['step'] in ['test', 'dev']:
            self.reload()

    def reload(self):
        if self.parameter['eval_ckpt'] is not None:
            state_dict_file = os.path.join(self.ckpt_dir, 'state_dict_' + self.parameter['eval_ckpt'] + '.ckpt')
        else:
            state_dict_file = os.path.join(self.state_dir, 'state_dict')
        self.state_dict_file = state_dict_file
        logging.info('Reload state_dict from {}'.format(state_dict_file))
        print('reload state_dict from {}'.format(state_dict_file))
        state = torch.load(state_dict_file, map_location=self.device)
        if os.path.isfile(state_dict_file):
            self.MMUC.load_state_dict(state)
        else:
            raise RuntimeError('No state dict in {}!'.format(state_dict_file))

    def save_checkpoint(self, epoch):
        torch.save(self.MMUC.state_dict(), os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt'))

    def del_checkpoint(self, epoch):
        path = os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt')
        if os.path.exists(path):
            os.remove(path)
        else:
            raise RuntimeError('No such checkpoint to delete: {}'.format(path))

    def save_best_state_dict(self, best_epoch):
        shutil.copy(os.path.join(self.ckpt_dir, 'state_dict_' + str(best_epoch) + '.ckpt'),
                    os.path.join(self.state_dir, 'state_dict'))

    def write_training_log(self, data, epoch):
        self.writer.add_scalar('Training_Loss', data['Loss'], epoch)

    def write_validating_log(self, data, epoch):
        self.writer.add_scalar('Validating_MRR', data['MRR'], epoch)
        self.writer.add_scalar('Validating_Hits_1', data['Hits@1'], epoch)
        self.writer.add_scalar('Validating_Hits_5', data['Hits@5'], epoch)
        self.writer.add_scalar('Validating_Hits_10', data['Hits@10'], epoch)
        self.writer.add_scalar('Validating_MSE', data['MSE'], epoch)
        self.writer.add_scalar('Validating_MAE', data['MAE'], epoch)

    def logging_training_data(self, data, epoch):
        logging.info("Epoch: {}\tMRR: {:.5f}\tHits@1: {:.5f}\tHits@5: {:.5f}\tHits@10: {:.5f}\t"
                     "MSE: {:.5f}\tMAE: {:.5f}\r".format(epoch, data['MRR'],
                                                         data['Hits@1'], data['Hits@5'], data['Hits@10'],
                                                         data['MSE'], data['MAE']))

    def logging_eval_data(self, data, state_path, istest=False):
        setname = 'dev set'
        if istest:
            setname = 'test set'
        logging.info("Eval {} on {}".format(state_path, setname))
        logging.info("MRR: {:.5f}\tHits@1: {:.5f}\tHits@5: {:.5f}\tHits@10: {:.5f}\tMSE: {:.5f}\tMAE: {:.5f}\r".format(
            data['MRR'], data['Hits@1'], data['Hits@5'], data['Hits@10'], data['MSE'], data['MAE']))

    def rank_predict(self, data, x, ranks):
        query_idx = x.shape[0] - 1
        _, idx = torch.sort(x, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        ranks.append(rank)
        if rank <= 10:
            data['Hits@10'] += 1
        if rank <= 5:
            data['Hits@5'] += 1
        if rank == 1:
            data['Hits@1'] += 1
        data['MRR'] += 1.0 / rank

    def confidence_predict(self, data, predict_score, query_confidence):
        data['MSE'] += (predict_score.item() - query_confidence.item()) ** 2
        data['MAE'] += abs(predict_score.item() - query_confidence.item())

    def do_one_step(self, task, iseval=False, curr_rel=''):
        rank_loss, mse_loss, loss, p_score, n_score, predict_cs, query_cs = 0, 0, 0, 0, 0, 0, 0
        if not iseval:
            self.optimizer.zero_grad()
            p_score, n_score, predict_cs, query_cs = self.MMUC(task, iseval, curr_rel)
            rank_loss = self.MMUC.un_ge_loss(p_score, n_score, query_cs)
            mse_loss = self.MMUC.mse_loss(predict_cs, query_cs)
            loss = self.alpha * rank_loss + self.beta * mse_loss
            loss.backward()
            self.optimizer.step()

        elif curr_rel != '':
            p_score, n_score, predict_cs, query_cs = self.MMUC(task, iseval, curr_rel)
            rank_loss = self.MMUC.un_ge_loss(p_score, n_score, query_cs)
            mse_loss = self.MMUC.mse_loss(predict_cs, query_cs)
            loss = self.alpha * rank_loss + self.beta * mse_loss
        return rank_loss, mse_loss, loss, p_score, n_score, predict_cs, query_cs

    def train(self):
        best_epoch = 0
        best_value = 0
        bad_counts = 0

        start_time = time.time()
        for e in range(self.epoch):
            train_task, curr_rel = self.train_data_loader.next_batch()  # batch_size tasks
            rank_loss, mse_loss, loss, _, _, _, _ = self.do_one_step(train_task, iseval=False, curr_rel=curr_rel)
            if e % self.print_epoch == 0:
                rank_value = rank_loss.item()
                mse_value = mse_loss.item()
                loss_value = loss.item()
                self.write_training_log({'Loss': loss_value}, e)
                end_time = time.time()
                print("Epoch: {}\trank_loss: {:.5f}\tmse_loss: {:.5f}\tLoss: {:.5f}\tcost: {:.2f}s".format(e, rank_value, mse_value, loss_value, (end_time - start_time)))
            if e % self.checkpoint_epoch == 0 and e != 0:
                print('Epoch  {} has finished, saving...'.format(e))
                self.save_checkpoint(e)

            if e % self.eval_epoch == 0 and e != 0:
                print('Epoch  {} has finished, validating...'.format(e))
                valid_data = self.eval(istest=False, epoch=e)
                self.write_validating_log(valid_data, e)

                metric = self.parameter['metric']
                if valid_data[metric] > best_value:
                    best_value = valid_data[metric]
                    best_epoch = e
                    print('\tBest model | {0} of valid set is {1:.5f}'.format(metric, best_value))
                    bad_counts = 0
                    self.save_checkpoint(best_epoch)    # save the best
                else:
                    print('\tBest {0} of valid set is {1:.5f} at {2} | bad count is {3}'.format(metric, best_value,
                                                                                                best_epoch, bad_counts))
                    bad_counts += 1
                if bad_counts >= self.early_stopping_patience:
                    print('\tEarly stopping at epoch %d' % e)
                    break

        print('Training has finished')
        print('\tBest epoch is {0} | {1} of valid set is {2:.5f}'.format(best_epoch, metric, best_value))
        end_time = time.time()
        print('Training cost {:.2f}s'.format(end_time - start_time))
        self.save_best_state_dict(best_epoch)
        print('Finish')

    def eval(self, istest=False, epoch=None):
        self.MMUC.eval()
        self.MMUC.rel_q_sharing = dict()  # clear sharing rel_q
        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0, 'MSE': 0, 'MAE': 0}
        ranks = []
        t = 0
        temp = dict()
        while True:
            eval_task, curr_rel = data_loader.next_one_on_eval()
            if eval_task == 'EOT':
                break
            t += 1
            _, _, _, p_score, n_score, predict_cs, query_cs = self.do_one_step(eval_task, iseval=True, curr_rel=curr_rel)
            x = torch.cat([n_score, p_score], 1).squeeze()
            self.rank_predict(data, x, ranks)
            self.confidence_predict(data, predict_cs.data, query_cs)

        for k in data.keys():
            data[k] = round(data[k] / t, 5)

        if self.parameter['step'] == 'train':
            self.logging_training_data(data, epoch)
        else:
            self.logging_eval_data(data, self.state_dict_file, istest)

        print("{}\tMRR: {:.5f}\tHits@1: {:.5f}\tHits@5: {:.5f}\tHits@10: {:.5f}\tMSE: {:.5f}\tMAE: {:.5f}\r".format(
            t, data['MRR'], data['Hits@1'], data['Hits@5'], data['Hits@10'], data['MSE'], data['MAE']))

        return data
