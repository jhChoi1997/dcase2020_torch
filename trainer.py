import os
import numpy as np
import copy

import utils
import dataset

from sklearn import metrics
from tqdm import tqdm
from sklearn.manifold import TSNE

import torch
from torch import nn


class WaveNetTrainer(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.machine_type = kwargs['machine_type']
        self.visualizer = kwargs['visualizer']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.csv_lines = []

    def loss_function(self, true, pred, is_eval=False):
        receptive_field = true.shape[-1] - pred.shape[-1]

        mse = nn.MSELoss()
        loss = mse(true[..., receptive_field:], pred)

        return loss

    def train(self, train_dataset, val_dataset):
        os.makedirs(os.path.join(self.args.model_dir, self.args.version, self.machine_type), exist_ok=True)
        print(f'Start Training - {self.machine_type}, {self.args.epochs} epochs')

        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0
        final_mean, final_inv_cov = 0, 0

        for epoch in range(self.args.epochs):
            mean_loss = 0
            pbar = tqdm(train_dataset, total=len(train_dataset), ncols=200)
            for data in pbar:
                data = data.float().to(self.args.device)
                self.model.train()
                output = self.model(data)
                loss = self.loss_function(data, output)
                pbar.set_description(f'Epoch: {epoch + 1}\tLoss: {loss.item():.4f}')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                mean_loss += loss
            mean_loss /= len(train_dataset)
            self.visualizer.add_train_loss1(mean_loss.detach().cpu().numpy())

            if epoch % 1 == 0:
                mean, inv_cov = self.eval_preprocess(val_dataset)
                auc, pauc = self.eval(mean, inv_cov)
                if auc + pauc > best_auc:
                    no_better = 0
                    best_auc = auc + pauc
                    a, p, e = auc, pauc, epoch
                    checkpoint_path = 'checkpoint_best_model.pth.tar'
                    utils.save_checkpoint(args=self.args,
                                          model=self.model,
                                          epoch=epoch,
                                          machine_type=self.machine_type,
                                          path=checkpoint_path,
                                          visualizer=self.visualizer)
                    print(f'Model saved! \t mean AUC: {a}, mean pAUC: {p}')
                    final_mean, final_inv_cov = mean, inv_cov
                else:
                    no_better += 1
                if no_better > self.args.early_stop:
                    break

        print(f'Training {self.machine_type} completed! \t Best Epoch: {e:4d}\tBest AUC: {a:3.3f}\tpAUC: {p:3.3f}')
        return final_mean, final_inv_cov

    def eval(self, mean, inv_cov):
        os.makedirs(os.path.join(self.args.result_dir, self.args.version), exist_ok=True)
        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))

        for id_str in machine_id_list:
            y_pred = []
            batch_data, y_true = dataset.get_wavenet_eval_test_dataset(self.args, self.machine_type, id_str, is_eval=True)
            for batch, name in batch_data:
                batch = batch.to(self.args.device)
                self.model.eval()
                output = self.model(batch)
                error_vector = self.get_error_vector(batch, output)
                score = self.anomaly_score(error_vector[0], mean, inv_cov)
                y_pred.append(score.detach().cpu().numpy())

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        return mean_auc, mean_p_auc

    def test(self, mean, inv_cov):
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)

        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        eval_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))
        test_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.test_dir, self.machine_type, 'test'))
        machine_id_list = eval_machine_id_list + test_machine_id_list
        machine_id_list.sort()
        for id_str in machine_id_list:
            y_pred = []
            anomaly_score_csv = f'{self.args.result_dir}/{self.args.version}/anomaly_score_{self.machine_type}_{id_str}.csv'
            anomaly_score_list = []
            anomaly_score_list.append(['Name', 'Anomaly Score'])
            if id_str in eval_machine_id_list:
                batch_data, y_true = dataset.get_wavenet_eval_test_dataset(self.args, self.machine_type, id_str, is_eval=True)
            else:
                batch_data, y_true = dataset.get_wavenet_eval_test_dataset(self.args, self.machine_type, id_str, is_eval=False)
            for batch, name in batch_data:
                batch = batch.to(self.args.device)
                self.model.eval()
                output = self.model(batch)
                error_vector = self.get_error_vector(batch, output)
                score = self.anomaly_score(error_vector[0], mean, inv_cov)
                y_pred.append(score.detach().cpu().numpy())
                anomaly_score_list.append([os.path.split(name[0])[1], score.detach().cpu().numpy()])

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
            id_AUC.append(auc)
            print(f'{id_str} AUC: {auc:3.3f} \t pAUC: {p_auc:3.3f}')
            utils.save_csv(anomaly_score_csv, anomaly_score_list)
            self.csv_lines.append([id_str, auc, p_auc])

        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        print(self.machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)

        self.csv_lines.append(['Average'] + [mean_auc, mean_p_auc])
        self.csv_lines.append([])

        result_path = os.path.join(result_dir, f'{self.machine_type}_{self.args.result_file}')
        utils.save_csv(result_path, self.csv_lines)

    def eval_preprocess(self, val_dataset):
        loss_list = []
        for data in val_dataset:
            data = data.to(self.args.device)
            with torch.no_grad():
                self.model.eval()
                output = self.model(data)
            error_vector = self.get_error_vector(data, output)
            loss_list.extend(error_vector)

        loss_array = torch.stack(loss_list)

        mean = torch.mean(loss_array, dim=0)
        cov = torch.cov(torch.transpose(loss_array, 0, 1))
        inv_cov = torch.linalg.inv(cov)
        return mean, inv_cov

    def get_error_vector(self, data, output):
        receptive_field = data.shape[-1] - output.shape[-1]

        error = data[..., receptive_field:] - output
        error_vector = torch.mean(error, dim=-1)
        return error_vector

    def anomaly_score(self, error_vector, mean, inv_cov):
        x = error_vector - mean
        tmp = torch.matmul(x, inv_cov)
        tmp1 = torch.matmul(tmp, x)
        dist = torch.sqrt(tmp1)
        return dist


class ResNetTrainer(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.machine_type = kwargs['machine_type']
        self.visualizer = kwargs['visualizer']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.n_class = kwargs['n_class']
        self.csv_lines = []

    def loss_function(self, true, pred):
        cce = nn.CrossEntropyLoss(label_smoothing=0.1)
        loss = cce(pred, true)
        return loss

    def train(self, train_dataset, val_dataset):
        os.makedirs(os.path.join(self.args.model_dir, self.args.version, self.machine_type), exist_ok=True)
        print(f'Start Training - {self.machine_type}, {self.args.epochs} epochs')

        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0

        for epoch in range(self.args.epochs):
            mean_loss = 0
            pbar = tqdm(train_dataset, total=len(train_dataset), ncols=200)
            for data, label in pbar:
                data = data.float().to(self.args.device)
                label = label.float().to(self.args.device)
                self.model.train()
                output = self.model(data)
                loss = self.loss_function(label, output)
                pbar.set_description(f'Epoch: {epoch + 1}\tLoss: {loss.item():.4f}')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                mean_loss += loss

            mean_loss /= len(train_dataset)
            self.visualizer.add_train_loss1(mean_loss.detach().cpu().numpy())

            if epoch % 1 == 0:
                auc, pauc = self.eval()
                if auc + pauc > best_auc:
                    no_better = 0
                    best_auc = auc + pauc
                    a, p, e = auc, pauc, epoch
                    checkpoint_path = 'checkpoint_best_model.pth.tar'
                    utils.save_checkpoint(args=self.args,
                                          model=self.model,
                                          epoch=epoch,
                                          machine_type=self.machine_type,
                                          path=checkpoint_path,
                                          visualizer=self.visualizer)
                    print(f'Model saved! \t mean AUC: {a}, mean pAUC: {p}')
                else:
                    no_better += 1
                if no_better > self.args.early_stop:
                    break
        print(f'Training {self.machine_type} completed! \t Best Epoch: {e:4d}\tBest AUC: {a:3.3f}\tpAUC: {p:3.3f}')

    def eval(self):
        os.makedirs(os.path.join(self.args.result_dir, self.args.version), exist_ok=True)
        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))

        for id_str in machine_id_list:
            y_pred = []
            batch_data, y_true = dataset.get_resnet_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=True)
            for batch, label, name in batch_data:
                batch = batch.float().to(self.args.device)
                label = label.to(self.args.device)
                self.model.eval()
                output = self.model(batch)
                score = self.anomaly_score(label, output)
                y_pred.append(score.detach().cpu().numpy())

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        return mean_auc, mean_p_auc

    def test(self):
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)

        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        eval_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))
        test_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.test_dir, self.machine_type, 'test'))
        machine_id_list = eval_machine_id_list + test_machine_id_list
        machine_id_list.sort()
        feature_maps, true_idx, true_y = [], [], []
        for id_str in machine_id_list:
            y_pred = []
            anomaly_score_csv = f'{self.args.result_dir}/{self.args.version}/anomaly_score_{self.machine_type}_{id_str}.csv'
            anomaly_score_list = []
            anomaly_score_list.append(['Name', 'Anomaly Score'])
            if id_str in eval_machine_id_list:
                batch_data, y_true = dataset.get_resnet_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=True)
            else:
                batch_data, y_true = dataset.get_resnet_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=False)
            for batch, label, name in batch_data:
                batch = batch.float().to(self.args.device)
                label = label.to(self.args.device)
                self.model.eval()
                feature_model = copy.deepcopy(self.model)
                feature_model.eval()
                feature_model.fc = nn.Identity()
                output = self.model(batch)
                feature = feature_model(batch)
                score = self.anomaly_score(label, output)
                y_pred.append(score.detach().cpu().numpy())
                anomaly_score_list.append([os.path.split(name[0])[1], score.detach().cpu().numpy()])

                feature_maps += feature.detach().cpu().numpy().tolist()
                true_idx.append(torch.argmax(label).detach().cpu().numpy())

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
            id_AUC.append(auc)
            print(f'{id_str} AUC: {auc:3.3f} \t pAUC: {p_auc:3.3f}')
            utils.save_csv(anomaly_score_csv, anomaly_score_list)
            self.csv_lines.append([id_str, auc, p_auc])

            true_y.extend(y_true)
        tsne = TSNE(n_components=2)
        cluster = np.array(tsne.fit_transform(np.array(feature_maps)))
        normal_cluster = np.array([v for i, v in enumerate(cluster) if true_y[i] == 0])
        anomaly_cluster = np.array([v for i, v in enumerate(cluster) if true_y[i] == 1])
        normal_idx = np.array([v for i, v in enumerate(true_idx) if true_y[i] == 0])
        anomaly_idx = np.array([v for i, v in enumerate(true_idx) if true_y[i] == 1])
        utils.plot_tsne(self.args, self.machine_type, normal_cluster, anomaly_cluster, normal_idx, anomaly_idx)

        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        print(self.machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)

        self.csv_lines.append(['Average'] + [mean_auc, mean_p_auc])
        self.csv_lines.append([])

        result_path = os.path.join(result_dir, f'{self.machine_type}_{self.args.result_file}')
        utils.save_csv(result_path, self.csv_lines)

    def anomaly_score(self, label, output):
        output_softmax = nn.functional.softmax(output, dim=-1)
        loss = torch.sum(label * output_softmax)
        return 1 - loss


class MTLClassTrainer(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.machine_type = kwargs['machine_type']
        self.visualizer = kwargs['visualizer']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.n_class = kwargs['n_class']
        self.csv_lines = []
        self.loss_value = []
        self.weight_value = []

    def loss_function(self, true_mel, output_mel, true_label, pred_label, weight):
        receptive_field = true_mel.shape[-1] - output_mel.shape[-1]

        mse = nn.MSELoss()
        loss1 = mse(true_mel[..., receptive_field:], output_mel)

        cce = nn.CrossEntropyLoss(label_smoothing=0.0)
        loss2 = cce(pred_label, true_label)

        w1, w2 = weight
        return w1 * loss1 + w2 * loss2, loss1, loss2

    def train(self, train_dataset, val_dataset):
        os.makedirs(os.path.join(self.args.model_dir, self.args.version, self.machine_type), exist_ok=True)
        print(f'Start Training - {self.machine_type}, {self.args.epochs} epochs')

        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0
        final_mean, final_inv_cov, final_score_mean, final_score_inv_cov = 0, 0, 0, 0

        for epoch in range(self.args.epochs):
            mean_loss = 0
            mean_loss1, mean_loss2 = 0, 0
            pbar = tqdm(train_dataset, total=len(train_dataset), ncols=200)

            self.weight_value = self.loss_weighting(self.loss_value, self.weight_value)

            for data, label in pbar:
                data = data.float().to(self.args.device)
                label = label.to(self.args.device)
                self.model.train()
                output1, output2 = self.model(data)
                loss, loss1, loss2 = self.loss_function(data, output1, label, output2, self.weight_value[-1])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                mean_loss += loss
                mean_loss1 += loss1
                mean_loss2 += loss2
                pbar.set_description(f'Epoch: {epoch + 1}\tLoss: {loss.item():.4f}\tLoss1: {loss1.item():.4f}\tLoss2: {loss2.item():.4f}')

            mean_loss /= len(train_dataset)
            mean_loss1 /= len(train_dataset)
            mean_loss2 /= len(train_dataset)

            self.visualizer.add_train_loss1(mean_loss1.detach().cpu().numpy())
            self.visualizer.add_train_loss2(mean_loss2.detach().cpu().numpy())
            self.loss_value.append([mean_loss1.detach().cpu().numpy(), mean_loss2.detach().cpu().numpy()])

            if epoch % 1 == 0:
                mean, inv_cov, score_mean, score_inv_cov = self.eval_preprocess(val_dataset)
                auc, pauc = self.eval(mean, inv_cov, score_mean, score_inv_cov)
                if auc + pauc > best_auc:
                    no_better = 0
                    best_auc = auc + pauc
                    a, p, e = auc, pauc, epoch
                    checkpoint_path = 'checkpoint_best_model.pth.tar'
                    utils.save_checkpoint(args=self.args,
                                          model=self.model,
                                          epoch=epoch,
                                          machine_type=self.machine_type,
                                          path=checkpoint_path,
                                          visualizer=self.visualizer)
                    print(f'Model saved! \t mean AUC: {a}, mean pAUC: {p}')
                    final_mean, final_inv_cov, final_score_mean, final_score_inv_cov = mean, inv_cov, score_mean, score_inv_cov
                else:
                    no_better += 1
                if no_better > self.args.early_stop:
                    break

        print(f'Training {self.machine_type} completed! \t Best Epoch: {e:4d}\tBest AUC: {a:3.3f}\tpAUC: {p:3.3f}')
        return final_mean, final_inv_cov, final_score_mean, final_score_inv_cov

    def eval(self, mean, inv_cov, score_mean, score_inv_cov):
        os.makedirs(os.path.join(self.args.result_dir, self.args.version), exist_ok=True)
        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))

        for id_str in machine_id_list:
            y_pred = []
            batch_data, y_true = dataset.get_mtl_class_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=True)
            for batch, label, name in batch_data:
                batch = batch.float().to(self.args.device)
                label = label.to(self.args.device)
                self.model.eval()
                output1, output2 = self.model(batch)
                error_vector = self.get_error_vector(batch, output1)
                score1 = self.anomaly_score1(error_vector, mean, inv_cov)
                score2 = self.anomaly_score2(label, output2)
                score = self.anomaly_score(score1, score2, score_mean, score_inv_cov)
                y_pred.append(score.detach().cpu().numpy())

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        return mean_auc, mean_p_auc

    def test(self, mean, inv_cov, score_mean, score_inv_cov):
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)

        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        feature_model = copy.deepcopy(self.model)
        feature_model.class_layer.fc = nn.Identity()

        eval_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))
        test_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.test_dir, self.machine_type, 'test'))
        machine_id_list = eval_machine_id_list + test_machine_id_list
        machine_id_list.sort()
        feature_maps, true_idx, true_y = [], [], []
        for id_str in machine_id_list:
            y_pred = []
            anomaly_score_csv = f'{self.args.result_dir}/{self.args.version}/anomaly_score_{self.machine_type}_{id_str}.csv'
            anomaly_score_list = []
            anomaly_score_list.append(['Name', 'Anomaly Score', 'Score 1', 'Score 2'])
            if id_str in eval_machine_id_list:
                batch_data, y_true = dataset.get_mtl_class_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=True)
            else:
                batch_data, y_true = dataset.get_mtl_class_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=False)
            for batch, label, name in batch_data:
                batch = batch.float().to(self.args.device)
                label = label.to(self.args.device)
                self.model.eval()
                feature_model.eval()
                output1, output2 = self.model(batch)
                _, feature = feature_model(batch)
                feature_maps += feature.detach().cpu().numpy().tolist()
                true_idx.append(torch.argmax(label).detach().cpu().numpy())

                error_vector = self.get_error_vector(batch, output1)
                score1 = self.anomaly_score1(error_vector, mean, inv_cov)
                score2 = self.anomaly_score2(label, output2)
                score = self.anomaly_score(score1, score2, score_mean, score_inv_cov)

                y_pred.append(score.detach().cpu().numpy())
                anomaly_score_list.append([os.path.split(name[0])[1], score.detach().cpu().numpy(), score1[0].detach().cpu().numpy(), score2[0].detach().cpu().numpy()])

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
            id_AUC.append(auc)
            print(f'{id_str} AUC: {auc:3.3f} \t pAUC: {p_auc:3.3f}')
            utils.save_csv(anomaly_score_csv, anomaly_score_list)
            self.csv_lines.append([id_str, auc, p_auc])
            true_y.extend(y_true)

        tsne = TSNE(n_components=2)
        cluster = np.array(tsne.fit_transform(np.array(feature_maps)))
        normal_cluster = np.array([v for i, v in enumerate(cluster) if true_y[i] == 0])
        anomaly_cluster = np.array([v for i, v in enumerate(cluster) if true_y[i] == 1])
        normal_idx = np.array([v for i, v in enumerate(true_idx) if true_y[i] == 0])
        anomaly_idx = np.array([v for i, v in enumerate(true_idx) if true_y[i] == 1])

        utils.plot_tsne(self.args, self.machine_type, normal_cluster, anomaly_cluster, normal_idx, anomaly_idx)

        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        print(self.machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)

        self.csv_lines.append(['Average'] + [mean_auc, mean_p_auc])
        self.csv_lines.append([])

        result_path = os.path.join(result_dir, f'{self.machine_type}_{self.args.result_file}')
        utils.save_csv(result_path, self.csv_lines)

    def loss_weighting(self, loss_list, weight_list):
        if len(weight_list) < 2:
            weight_list.append([1, 1])
        else:
            T = 2
            r1 = loss_list[-1][0] / loss_list[-2][0]
            r2 = loss_list[-1][1] / loss_list[-2][1]
            w1 = 2 * np.exp(r1 / T) / (np.exp(r1 / T) + np.exp(r2 / T))
            w2 = 2 * np.exp(r2 / T) / (np.exp(r1 / T) + np.exp(r2 / T))
            weight_list.append([w1, w2])
        return weight_list

    def eval_preprocess(self, val_dataset):
        error_vector_list = []
        score_list1 = []
        score_list2 = []
        for data, label in val_dataset:
            data = data.float().to(self.args.device)
            label = label.to(self.args.device)
            self.model.eval()
            with torch.no_grad():
                output1, output2 = self.model(data)
            error_vector = self.get_error_vector(data, output1)
            score2 = self.anomaly_score2(label, output2)
            error_vector_list.extend(error_vector)
            score_list2.extend(score2)

        error_vector_array = torch.stack(error_vector_list)
        mean = torch.mean(error_vector_array, dim=0)
        cov = torch.cov(torch.transpose(error_vector_array, 0, 1))
        inv_cov = torch.linalg.inv(cov)

        score_list1.extend(self.anomaly_score1(error_vector_array, mean, inv_cov))

        score_list1 = torch.stack(score_list1)
        score_list2 = torch.stack(score_list2)

        score_array = torch.cat((score_list1.unsqueeze(0), score_list2.unsqueeze(0)), dim=0)

        score_mean = torch.mean(score_array, dim=1)
        score_cov = torch.cov(score_array)
        score_inv_cov = torch.linalg.inv(score_cov)

        return mean, inv_cov, score_mean, score_inv_cov

    def get_error_vector(self, data, output):
        receptive_field = data.shape[-1] - output.shape[-1]
        error = data[..., receptive_field:] - output
        error_vector = torch.mean(error, dim=-1)
        return error_vector


    def anomaly_score1(self, error_vector, mean, inv_cov):
        x = error_vector - mean

        tmp = torch.matmul(x, inv_cov)
        tmp1 = torch.matmul(tmp, torch.transpose(x, 0, 1))
        dist = torch.sqrt(torch.diag(tmp1))
        return dist

    def anomaly_score2(self, true, pred):
        output_softmax = nn.functional.softmax(pred, dim=-1)
        loss = torch.sum(true * output_softmax, dim=-1)
        return loss

    def anomaly_score(self, score1, score2, score_mean, score_inv_cov):
        score = torch.cat((score1, score2), dim=0)
        x = score - score_mean
        tmp = torch.matmul(x, score_inv_cov)
        tmp1 = torch.matmul(tmp, x)
        dist = torch.sqrt(tmp1)
        return dist

        # score = np.array([(u, v) for u, v in zip(score1, score2)])
        # x = score - score_mean
        # tmp = np.matmul(x, score_inv_cov)
        # tmp1 = np.matmul(tmp, np.transpose(x))
        # dist = np.sqrt(np.diag(tmp1))
        # return dist


class MultiResolutionWaveNetTrainer(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.machine_type = kwargs['machine_type']
        self.visualizer = kwargs['visualizer']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.csv_lines = []

    def loss_function(self, true_mel, output_mel):
        n_blocks = output_mel.shape[1]
        receptive_field = true_mel.shape[-1] - output_mel.shape[-1]

        true_mel_blocks = torch.stack([true_mel for _ in range(n_blocks)], dim=1)

        mse = nn.MSELoss()
        loss = mse(true_mel_blocks[..., receptive_field:], output_mel)

        return loss

    def train(self, train_dataset, val_dataset):
        os.makedirs(os.path.join(self.args.model_dir, self.args.version, self.machine_type), exist_ok=True)
        print(f'Start Training - {self.machine_type}, {self.args.epochs} epochs')

        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0
        final_mean, final_inv_cov, final_block_mean, final_block_inv_cov = 0, 0, 0, 0

        for epoch in range(self.args.epochs):
            mean_loss = 0
            pbar = tqdm(train_dataset, total=len(train_dataset), ncols=200)
            for data in pbar:
                data = data.float().to(self.args.device)
                self.model.train()
                output = self.model(data)
                loss = self.loss_function(data, output)
                pbar.set_description(f'Epoch: {epoch + 1}\tLoss: {loss.item():.4f}')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                mean_loss += loss
            mean_loss /= len(train_dataset)
            self.visualizer.add_train_loss1(mean_loss.detach().cpu().numpy())

            if epoch % 1 == 0:
                mean, inv_cov, block_mean, block_inv_cov = self.eval_preprocess(val_dataset)
                auc, pauc = self.eval(mean, inv_cov, block_mean, block_inv_cov)
                if auc + pauc > best_auc:
                    no_better = 0
                    best_auc = auc + pauc
                    a, p, e = auc, pauc, epoch
                    checkpoint_path = 'checkpoint_best_model.pth.tar'
                    utils.save_checkpoint(args=self.args,
                                          model=self.model,
                                          epoch=epoch,
                                          machine_type=self.machine_type,
                                          path=checkpoint_path,
                                          visualizer=self.visualizer)
                    print(f'Model saved! \t mean AUC: {a}, mean pAUC: {p}')
                    final_mean, final_inv_cov, final_block_mean, final_block_inv_cov = mean, inv_cov, block_mean, block_inv_cov
                else:
                    no_better += 1
                if no_better > self.args.early_stop:
                    break

        print(f'Training {self.machine_type} completed! \t Best Epoch: {e:4d}\tBest AUC: {a:3.3f}\tpAUC: {p:3.3f}')
        return final_mean, final_inv_cov, final_block_mean, final_block_inv_cov

    def eval(self, mean, inv_cov, block_mean, block_inv_cov):
        os.makedirs(os.path.join(self.args.result_dir, self.args.version), exist_ok=True)
        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))

        for id_str in machine_id_list:
            y_pred = []
            batch_data, y_true = dataset.get_wavenet_eval_test_dataset(self.args, self.machine_type, id_str, is_eval=True)
            for batch, name in batch_data:
                batch = batch.to(self.args.device)
                self.model.eval()
                output = self.model(batch)

                error_vector = self.get_error_vectors(batch, output)
                n_block = error_vector.shape[1]
                score_vector = []

                for i in range(n_block):
                    score_vector.append(self.block_score(error_vector[:, i], mean[i], inv_cov[i]))
                score_vector = torch.transpose(torch.stack(score_vector), 0, 1)
                score = self.anomaly_score(score_vector, block_mean, block_inv_cov)
                y_pred.extend(score.detach().cpu().numpy())

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        return mean_auc, mean_p_auc

    def test(self, mean, inv_cov, block_mean, block_inv_cov):
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)

        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        eval_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))
        test_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.test_dir, self.machine_type, 'test'))
        machine_id_list = eval_machine_id_list + test_machine_id_list
        machine_id_list.sort()
        for id_str in machine_id_list:
            y_pred = []
            anomaly_score_csv = f'{self.args.result_dir}/{self.args.version}/anomaly_score_{self.machine_type}_{id_str}.csv'
            anomaly_score_list = []
            anomaly_score_list.append(['Name', 'Anomaly Score', 'Score'])
            if id_str in eval_machine_id_list:
                batch_data, y_true = dataset.get_wavenet_eval_test_dataset(self.args, self.machine_type, id_str, is_eval=True)
            else:
                batch_data, y_true = dataset.get_wavenet_eval_test_dataset(self.args, self.machine_type, id_str, is_eval=False)
            for batch, name in batch_data:
                batch = batch.to(self.args.device)
                self.model.eval()
                output = self.model(batch)

                error_vector = self.get_error_vectors(batch, output)
                n_block = error_vector.shape[1]
                score_vector = []
                for i in range(n_block):
                    score_vector.append(self.block_score(error_vector[:, i], mean[i], inv_cov[i]))
                score_vector = torch.transpose(torch.stack(score_vector), 0, 1)
                score = self.anomaly_score(score_vector, block_mean, block_inv_cov)
                y_pred.extend(score.detach().cpu().numpy())

                anomaly_score_list.append([os.path.split(name[0])[1], score.detach().cpu().numpy()])

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
            id_AUC.append(auc)
            print(f'{id_str} AUC: {auc:3.3f} \t pAUC: {p_auc:3.3f}')
            utils.save_csv(anomaly_score_csv, anomaly_score_list)
            self.csv_lines.append([id_str, auc, p_auc])

        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        print(self.machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)

        self.csv_lines.append(['Average'] + [mean_auc, mean_p_auc])
        self.csv_lines.append([])

        result_path = os.path.join(result_dir, f'{self.machine_type}_{self.args.result_file}')
        utils.save_csv(result_path, self.csv_lines)

    def eval_preprocess(self, val_dataset):
        error_vector_list = []

        for data in val_dataset:
            data = data.to(self.args.device)
            with torch.no_grad():
                self.model.eval()
                output = self.model(data)
            error_vectors = self.get_error_vectors(data, output)
            error_vector_list.extend(error_vectors)


        error_vector_array = torch.stack(error_vector_list)
        n_blocks = error_vector_array.shape[1]
        mean_list, inv_cov_list = [], []
        score_list = []

        for block_ids in range(n_blocks):
            mean = torch.mean(error_vector_array[:, block_ids], dim=0)
            cov = torch.cov(torch.transpose(error_vector_array[:, block_ids], 0, 1))
            inv_cov = torch.linalg.inv(cov)
            mean_list.append(mean)
            inv_cov_list.append(inv_cov)

            score = self.anomaly_score(error_vector_array[:, block_ids], mean, inv_cov)
            score_list.append(score)

        score_array = torch.stack(score_list)
        score_mean = torch.mean(score_array, dim=-1)
        score_cov = torch.cov(score_array)
        score_inv_cov = torch.linalg.inv(score_cov)

        mean_list = torch.stack(mean_list)
        inv_cov_list = torch.stack(inv_cov_list)
        return mean_list, inv_cov_list, score_mean, score_inv_cov


    def get_error_vectors(self, data, output):
        n_blocks = output.shape[1]
        receptive_field = data.shape[-1] - output.shape[-1]

        data_copy = torch.stack([data for _ in range(n_blocks)], dim=1)
        spec_diff = data_copy[..., receptive_field:] - output
        freq_mean = torch.mean(spec_diff, dim=-1)
        return freq_mean

    def block_score(self, spec, mean, inv_cov):
        x = spec - mean
        tmp = torch.matmul(x, inv_cov)
        tmp1 = torch.matmul(tmp, torch.transpose(x, 0, 1))
        dist = torch.sqrt(torch.diag(tmp1))
        return dist

    def anomaly_score(self, vector, mean, inv_cov):
        x = vector - mean
        tmp = torch.matmul(x, inv_cov)
        tmp1 = torch.matmul(tmp, torch.transpose(x, 0, 1))
        dist = torch.sqrt(torch.diag(tmp1))
        return dist


class MTLSegmentationTrainer(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.machine_type = kwargs['machine_type']
        self.visualizer = kwargs['visualizer']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.n_class = kwargs['n_class']
        self.csv_lines = []
        self.loss_value = []
        self.weight_value = []

    def loss_function(self, true_mel, output_mel, true_label, pred_label, weight):
        receptive_field = true_mel.shape[-1] - output_mel.shape[-1]

        mse = nn.MSELoss()
        loss1 = mse(true_mel[..., receptive_field:], output_mel)

        cce = nn.CrossEntropyLoss(label_smoothing=0.1)
        loss2 = cce(pred_label, true_label[..., receptive_field:])

        w1, w2 = weight
        return w1 * loss1 + w2 * loss2, loss1, loss2

    def train(self, train_dataset, val_dataset):
        os.makedirs(os.path.join(self.args.model_dir, self.args.version, self.machine_type), exist_ok=True)
        print(f'Start Training - {self.machine_type}, {self.args.epochs} epochs')

        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0
        final_mean, final_inv_cov, final_score_mean, final_score_inv_cov = 0, 0, 0, 0


        for epoch in range(self.args.epochs):
            mean_loss = 0
            mean_loss1, mean_loss2 = 0, 0
            pbar = tqdm(train_dataset, total=len(train_dataset), ncols=200)

            self.weight_value = self.loss_weighting(self.loss_value, self.weight_value)

            for data, label in pbar:
                data = data.float().to(self.args.device)
                label = label.to(self.args.device)
                self.model.train()
                output1, output2 = self.model(data)
                loss, loss1, loss2 = self.loss_function(data, output1, label, output2, self.weight_value[-1])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                mean_loss += loss
                mean_loss1 += loss1
                mean_loss2 += loss2
                pbar.set_description(f'Epoch: {epoch + 1}\tLoss: {loss.item():.4f}\tLoss1: {loss1.item():.4f}\tLoss2: {loss2.item():.4f}')

            mean_loss /= len(train_dataset)
            mean_loss1 /= len(train_dataset)
            mean_loss2 /= len(train_dataset)

            self.visualizer.add_train_loss1(mean_loss1.detach().cpu().numpy())
            self.visualizer.add_train_loss2(mean_loss2.detach().cpu().numpy())
            self.loss_value.append([mean_loss1.detach().cpu().numpy(), mean_loss2.detach().cpu().numpy()])

            if epoch % 1 == 0:
                mean, inv_cov, score_mean, score_inv_cov = self.eval_preprocess(val_dataset)
                auc, pauc = self.eval(mean, inv_cov, score_mean, score_inv_cov)
                if auc + pauc > best_auc:
                    no_better = 0
                    best_auc = auc + pauc
                    a, p, e = auc, pauc, epoch
                    checkpoint_path = 'checkpoint_best_model.pth.tar'
                    utils.save_checkpoint(args=self.args,
                                          model=self.model,
                                          epoch=epoch,
                                          machine_type=self.machine_type,
                                          path=checkpoint_path,
                                          visualizer=self.visualizer)
                    print(f'Model saved! \t mean AUC: {a}, mean pAUC: {p}')
                    final_mean, final_inv_cov, final_score_mean, final_score_inv_cov = mean, inv_cov, score_mean, score_inv_cov
                else:
                    no_better += 1
                if no_better > self.args.early_stop:
                    break

        print(f'Training {self.machine_type} completed! \t Best Epoch: {e:4d}\tBest AUC: {a:3.3f}\tpAUC: {p:3.3f}')
        return final_mean, final_inv_cov, final_score_mean, final_score_inv_cov

    def eval(self, mean, inv_cov, score_mean, score_inv_cov):
        os.makedirs(os.path.join(self.args.result_dir, self.args.version), exist_ok=True)
        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))

        for id_str in machine_id_list:
            y_pred = []
            batch_data, y_true = dataset.get_mtl_class_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=True, is_seg=True)
            for batch, label, name in batch_data:
                batch = batch.float().to(self.args.device)
                label = label.to(self.args.device)
                self.model.eval()
                output1, output2 = self.model(batch)
                error_vector = self.get_error_vector(batch, output1)
                score1 = self.anomaly_score1(error_vector, mean, inv_cov)
                score2 = self.anomaly_score2(label, output2)
                score = self.anomaly_score(score1, score2, score_mean, score_inv_cov)
                y_pred.append(score.detach().cpu().numpy())

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        return mean_auc, mean_p_auc

    def test(self, mean, inv_cov, score_mean, score_inv_cov):
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)

        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        feature_model = copy.deepcopy(self.model)
        feature_model.segment[-1] = nn.Identity()

        eval_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))
        test_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.test_dir, self.machine_type, 'test'))
        machine_id_list = eval_machine_id_list + test_machine_id_list
        machine_id_list.sort()

        feature_maps, true_idx, true_y = [], [], []
        for id_str in machine_id_list:
            y_pred = []
            anomaly_score_csv = f'{self.args.result_dir}/{self.args.version}/anomaly_score_{self.machine_type}_{id_str}.csv'
            anomaly_score_list = []
            anomaly_score_list.append(['Name', 'Anomaly Score', 'Score 1', 'Score 2'])
            if id_str in eval_machine_id_list:
                batch_data, y_true = dataset.get_mtl_class_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=True, is_seg=True)
            else:
                batch_data, y_true = dataset.get_mtl_class_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=False, is_seg=True)
            for batch, label, name in batch_data:
                batch = batch.float().to(self.args.device)
                label = label.to(self.args.device)

                self.model.eval()
                feature_model.eval()
                output1, output2 = self.model(batch)
                _, feature = feature_model(batch)

                error_vector = self.get_error_vector(batch, output1)
                score1 = self.anomaly_score1(error_vector, mean, inv_cov)
                score2 = self.anomaly_score2(label, output2)
                score = self.anomaly_score(score1, score2, score_mean, score_inv_cov)

                y_pred.append(score.detach().cpu().numpy())
                anomaly_score_list.append([os.path.split(name[0])[1], score.detach().cpu().numpy(), score1[0].detach().cpu().numpy(), score2[0].detach().cpu().numpy()])

                feature_maps += torch.mean(feature, dim=-1).detach().cpu().numpy().tolist()
                true_idx.append(torch.argmax(torch.mean(label, dim=-1)).detach().cpu().numpy())

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
            id_AUC.append(auc)
            print(f'{id_str} AUC: {auc:3.3f} \t pAUC: {p_auc:3.3f}')
            utils.save_csv(anomaly_score_csv, anomaly_score_list)
            self.csv_lines.append([id_str, auc, p_auc])
            true_y.extend(y_true)

        tsne = TSNE(n_components=2)
        cluster = np.array(tsne.fit_transform(np.array(feature_maps)))
        normal_cluster = np.array([v for i, v in enumerate(cluster) if true_y[i] == 0])
        anomaly_cluster = np.array([v for i, v in enumerate(cluster) if true_y[i] == 1])
        normal_idx = np.array([v for i, v in enumerate(true_idx) if true_y[i] == 0])
        anomaly_idx = np.array([v for i, v in enumerate(true_idx) if true_y[i] == 1])
        utils.plot_tsne(self.args, self.machine_type, normal_cluster, anomaly_cluster, normal_idx, anomaly_idx)

        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        print(self.machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)

        self.csv_lines.append(['Average'] + [mean_auc, mean_p_auc])
        self.csv_lines.append([])

        result_path = os.path.join(result_dir, f'{self.machine_type}_{self.args.result_file}')
        utils.save_csv(result_path, self.csv_lines)

    def loss_weighting(self, loss_list, weight_list):
        if len(weight_list) < 2:
            weight_list.append([1, 1])
        else:
            T = 2
            r1 = loss_list[-1][0] / loss_list[-2][0]
            r2 = loss_list[-1][1] / loss_list[-2][1]
            w1 = 2 * np.exp(r1 / T) / (np.exp(r1 / T) + np.exp(r2 / T))
            w2 = 2 * np.exp(r2 / T) / (np.exp(r1 / T) + np.exp(r2 / T))
            weight_list.append([w1, w2])
        return weight_list

    def eval_preprocess(self, val_dataset):
        error_vector_list = []
        score_list1 = []
        score_list2 = []
        for data, label in val_dataset:
            data = data.float().to(self.args.device)
            label = label.to(self.args.device)
            self.model.eval()
            with torch.no_grad():
                output1, output2 = self.model(data)
            error_vector = self.get_error_vector(data, output1)
            score2 = self.anomaly_score2(label, output2)
            error_vector_list.extend(error_vector)
            score_list2.extend(score2)

        error_vector_array = torch.stack(error_vector_list)
        mean = torch.mean(error_vector_array, dim=0)
        cov = torch.cov(torch.transpose(error_vector_array, 0, 1))
        inv_cov = torch.linalg.inv(cov)

        score_list1.extend(self.anomaly_score1(error_vector_array, mean, inv_cov))

        score_list1 = torch.stack(score_list1)
        score_list2 = torch.stack(score_list2)
        score_array = torch.cat((score_list1.unsqueeze(0), score_list2.unsqueeze(0)), dim=0)

        score_mean = torch.mean(score_array, dim=1)
        score_cov = torch.cov(score_array)
        score_inv_cov = torch.linalg.inv(score_cov)

        return mean, inv_cov, score_mean, score_inv_cov

    def get_error_vector(self, data, output):
        receptive_field = data.shape[-1] - output.shape[-1]
        error = data[..., receptive_field:] - output
        error_vector = torch.mean(error, dim=-1)
        return error_vector

    def anomaly_score1(self, error_vector, mean, inv_cov):
        x = error_vector - mean
        tmp = torch.matmul(x, inv_cov)
        tmp1 = torch.matmul(tmp, torch.transpose(x, 0, 1))
        dist = torch.sqrt(torch.diag(tmp1))
        return dist

    def anomaly_score2(self, true, pred):
        receptive_field = true.shape[-1] - pred.shape[-1]
        output_softmax = nn.functional.softmax(pred, dim=1)
        loss = torch.sum(true[..., receptive_field:] * output_softmax, dim=1)
        score = torch.mean(loss, dim=-1)
        return score

    def anomaly_score(self, score1, score2, score_mean, score_inv_cov):
        score = torch.cat((score1, score2), dim=0)
        x = score - score_mean
        tmp = torch.matmul(x, score_inv_cov)
        tmp1 = torch.matmul(tmp, x)
        dist = torch.sqrt(tmp1)
        return dist


class MTLClassSegTrainer(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.machine_type = kwargs['machine_type']
        self.visualizer = kwargs['visualizer']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.n_class = kwargs['n_class']
        self.csv_lines = []
        self.loss_value = []
        self.weight_value = []

    def loss_function(self, true_mel, output_mel, true_label1, pred_label1, true_label2, pred_label2, weight):
        receptive_field = true_mel.shape[-1] - output_mel.shape[-1]

        mse = nn.MSELoss()
        loss1 = mse(true_mel[..., receptive_field:], output_mel)

        cce2 = nn.CrossEntropyLoss(label_smoothing=0.1)
        loss2 = cce2(pred_label1, true_label1)

        cce3 = nn.CrossEntropyLoss(label_smoothing=0.1)
        loss3 = cce3(pred_label2, true_label2[..., receptive_field:])

        w1, w2, w3 = weight
        return w1 * loss1 + w2 * loss2 + w3 * loss3, loss1, loss2, loss3

    def train(self, train_dataset, val_dataset):
        os.makedirs(os.path.join(self.args.model_dir, self.args.version, self.machine_type), exist_ok=True)
        print(f'Start Training - {self.machine_type}, {self.args.epochs} epochs')

        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0
        final_mean, final_inv_cov, final_score_mean, final_score_inv_cov = 0, 0, 0, 0

        for epoch in range(self.args.epochs):
            mean_loss = 0
            mean_loss1, mean_loss2, mean_loss3 = 0, 0, 0
            pbar = tqdm(train_dataset, total=len(train_dataset), ncols=200)
            self.weight_value = self.loss_weighting(self.loss_value, self.weight_value)

            for data, label1, label2 in pbar:
                data = data.float().to(self.args.device)
                label1 = label1.to(self.args.device)
                label2 = label2.to(self.args.device)
                self.model.train()
                output1, output2, output3 = self.model(data)
                loss, loss1, loss2, loss3 = self.loss_function(data, output1, label1, output2, label2, output3, self.weight_value[-1])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                mean_loss += loss
                mean_loss1 += loss1
                mean_loss2 += loss2
                mean_loss3 += loss3
                pbar.set_description(f'Epoch: {epoch + 1}\tLoss: {loss.item():.4f}\tLoss1: {loss1.item():.4f}\tLoss2: {loss2.item():.4f}\tLoss3: {loss3.item():.4f}')

            mean_loss /= len(train_dataset)
            mean_loss1 /= len(train_dataset)
            mean_loss2 /= len(train_dataset)

            self.visualizer.add_train_loss1(mean_loss1.detach().cpu().numpy())
            self.visualizer.add_train_loss2(mean_loss2.detach().cpu().numpy())
            self.visualizer.add_train_loss3(mean_loss3.detach().cpu().numpy())
            self.loss_value.append([mean_loss1.detach().cpu().numpy(), mean_loss2.detach().cpu().numpy(), mean_loss3.detach().cpu().numpy()])

            if epoch % 1 == 0:
                mean, inv_cov, score_mean, score_inv_cov = self.eval_preprocess(val_dataset)
                auc, pauc = self.eval(mean, inv_cov, score_mean, score_inv_cov)
                if auc + pauc > best_auc:
                    no_better = 0
                    best_auc = auc + pauc
                    a, p, e = auc, pauc, epoch
                    checkpoint_path = 'checkpoint_best_model.pth.tar'
                    utils.save_checkpoint(args=self.args,
                                          model=self.model,
                                          epoch=epoch,
                                          machine_type=self.machine_type,
                                          path=checkpoint_path,
                                          visualizer=self.visualizer)
                    print(f'Model saved! \t mean AUC: {a}, mean pAUC: {p}')
                    final_mean, final_inv_cov, final_score_mean, final_score_inv_cov = mean, inv_cov, score_mean, score_inv_cov
                else:
                    no_better += 1
                if no_better > self.args.early_stop:
                    break

        print(f'Training {self.machine_type} completed! \t Best Epoch: {e:4d}\tBest AUC: {a:3.3f}\tpAUC: {p:3.3f}')
        return final_mean, final_inv_cov, final_score_mean, final_score_inv_cov

    def eval(self, mean, inv_cov, score_mean, score_inv_cov):
        os.makedirs(os.path.join(self.args.result_dir, self.args.version), exist_ok=True)
        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))

        for id_str in machine_id_list:
            y_pred = []
            batch_data, y_true = dataset.get_mtl_class_seg_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=True)
            for batch, label1, label2, name in batch_data:
                batch = batch.float().to(self.args.device)
                label1 = label1.to(self.args.device)
                label2 = label2.to(self.args.device)
                self.model.eval()
                output1, output2, output3 = self.model(batch)
                error_vector = self.get_error_vector(batch, output1)
                score1 = self.anomaly_score1(error_vector, mean, inv_cov)
                score2 = self.anomaly_score2(label1, output2)
                score3 = self.anomaly_score3(label2, output3)
                score = self.anomaly_score(score1, score2, score3, score_mean, score_inv_cov)
                y_pred.append(score.detach().cpu().numpy())

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        return mean_auc, mean_p_auc

    def test(self, mean, inv_cov, score_mean, score_inv_cov):
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)

        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        eval_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))
        test_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.test_dir, self.machine_type, 'test'))
        machine_id_list = eval_machine_id_list + test_machine_id_list
        machine_id_list.sort()
        for id_str in machine_id_list:
            y_pred = []
            anomaly_score_csv = f'{self.args.result_dir}/{self.args.version}/anomaly_score_{self.machine_type}_{id_str}.csv'
            anomaly_score_list = []
            anomaly_score_list.append(['Name', 'Anomaly Score', 'Score 1', 'Score 2', 'Score 3'])
            if id_str in eval_machine_id_list:
                batch_data, y_true = dataset.get_mtl_class_seg_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=True)
            else:
                batch_data, y_true = dataset.get_mtl_class_seg_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=False)
            for batch, label1, label2, name in batch_data:
                batch = batch.float().to(self.args.device)
                label1 = label1.to(self.args.device)
                label2 = label2.to(self.args.device)
                output1, output2, output3 = self.model(batch)

                error_vector = self.get_error_vector(batch, output1)
                score1 = self.anomaly_score1(error_vector, mean, inv_cov)
                score2 = self.anomaly_score2(label1, output2)
                score3 = self.anomaly_score3(label2, output3)
                score = self.anomaly_score(score1, score2, score3, score_mean, score_inv_cov)
                y_pred.append(score.detach().cpu().numpy())
                anomaly_score_list.append([os.path.split(name[0])[1], score.detach().cpu().numpy(), score1[0].detach().cpu().numpy(), score2[0].detach().cpu().numpy(), score3[0].detach().cpu().numpy()])

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
            id_AUC.append(auc)
            print(f'{id_str} AUC: {auc:3.3f} \t pAUC: {p_auc:3.3f}')
            utils.save_csv(anomaly_score_csv, anomaly_score_list)
            self.csv_lines.append([id_str, auc, p_auc])

        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        print(self.machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)

        self.csv_lines.append(['Average'] + [mean_auc, mean_p_auc])
        self.csv_lines.append([])

        result_path = os.path.join(result_dir, f'{self.machine_type}_{self.args.result_file}')
        utils.save_csv(result_path, self.csv_lines)

    def loss_weighting(self, loss_list, weight_list):
        if len(weight_list) < 2:
            weight_list.append([1, 1, 1])
        else:
            T = 2
            r1 = loss_list[-1][0] / loss_list[-2][0]
            r2 = loss_list[-1][1] / loss_list[-2][1]
            r3 = loss_list[-1][2] / loss_list[-2][2]
            w1 = 2 * np.exp(r1 / T) / (np.exp(r1 / T) + np.exp(r2 / T) + np.exp(r3 / T))
            w2 = 2 * np.exp(r2 / T) / (np.exp(r1 / T) + np.exp(r2 / T) + np.exp(r3 / T))
            w3 = 2 * np.exp(r3 / T) / (np.exp(r1 / T) + np.exp(r2 / T) + np.exp(r3 / T))
            weight_list.append([w1, w2, w3])
        return weight_list

    def eval_preprocess(self, val_dataset):
        error_vector_list = []
        score_list1 = []
        score_list2 = []
        score_list3 = []
        for data, label1, label2 in val_dataset:
            data = data.float().to(self.args.device)
            label1 = label1.to(self.args.device)
            label2 = label2.to(self.args.device)
            self.model.eval()
            with torch.no_grad():
                output1, output2, output3 = self.model(data)
            error_vector = self.get_error_vector(data, output1)
            score2 = self.anomaly_score2(label1, output2)
            score3 = self.anomaly_score3(label2, output3)
            error_vector_list.extend(error_vector)
            score_list2.extend(score2)
            score_list3.extend(score3)

        error_vector_array = torch.stack(error_vector_list)
        mean = torch.mean(error_vector_array, dim=0)
        cov = torch.cov(torch.transpose(error_vector_array, 0, 1))
        inv_cov = torch.linalg.inv(cov)

        score_list1.extend(self.anomaly_score1(error_vector_array, mean, inv_cov))

        score_list1 = torch.stack(score_list1)
        score_list2 = torch.stack(score_list2)
        score_list3 = torch.stack(score_list3)

        score_array = torch.cat((score_list1.unsqueeze(0), score_list2.unsqueeze(0), score_list3.unsqueeze(0)), dim=0)

        score_mean = torch.mean(score_array, dim=1)
        score_cov = torch.cov(score_array)
        score_inv_cov = torch.linalg.inv(score_cov)

        return mean, inv_cov, score_mean, score_inv_cov

    def get_error_vector(self, data, output):
        receptive_field = data.shape[-1] - output.shape[-1]
        error = data[..., receptive_field:] - output
        error_vector = torch.mean(error, dim=-1)
        return error_vector

    def anomaly_score1(self, error_vector, mean, inv_cov):
        x = error_vector - mean
        tmp = torch.matmul(x, inv_cov)
        tmp1 = torch.matmul(tmp, torch.transpose(x, 0, 1))
        dist = torch.sqrt(torch.diag(tmp1))
        return dist

    def anomaly_score2(self, true, pred):
        output_softmax = nn.functional.softmax(pred, dim=-1)
        loss = torch.sum(true * output_softmax, dim=-1)
        return loss

    def anomaly_score3(self, true, pred):
        receptive_field = true.shape[-1] - pred.shape[-1]
        output_softmax = nn.functional.softmax(pred, dim=1)
        loss = torch.sum(true[..., receptive_field:] * output_softmax, dim=1)
        score = torch.mean(loss, dim=-1)
        return score

    def anomaly_score(self, score1, score2, score3, score_mean, score_inv_cov):
        score = torch.cat((score1, score2, score3), dim=0)
        x = score - score_mean
        tmp = torch.matmul(x, score_inv_cov)
        tmp1 = torch.matmul(tmp, x)
        dist = torch.sqrt(tmp1)
        return dist