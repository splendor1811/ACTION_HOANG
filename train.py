import argparse

import numpy as np
import torch.cuda
import wandb
import yaml

from utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

gc.collect()
torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description='Train Action Recognition Model')
    parser.add_argument('--config',
                        default='/home/tuantran/AI_TEAM/ACT_HOANG/posec3d_v2/configs/slowonly_r50_ntu60_xsub/limb_xview.yaml',
                        help='config file path')

    args = parser.parse_args()

    return args


class TrainPipline():
    def __init__(self, args):
        self.args = args
        self.reweighted = True
        self.args['print_log'] = True
        self.args['save_interval'] = 1
        self.args['model_saved_name'] = os.path.join(self.args['work_dir'], 'runs')
        self.args['save_epoch'] = 15
        self.args['step'] = [15, 25, 35]
        # Setting device
        self.output_device = self.args['device']
        self.best_acc = 0

    def load_data(self):
        # load data and preprocessing
        train_dataset = PoseC3DDataset(ann_file=self.args['data']['train']['ann_file'],
                                       pipeline=self.args['train_pipeline'],
                                       split=self.args['data']['train']['split'],
                                       multi_class=True,
                                       test_mode=False,
                                       num_classes=self.args['model']['cls_head']['args']['num_classes']
                                       )
        val_dataset = PoseC3DDataset(ann_file=self.args['data']['val']['ann_file'],
                                     pipeline=self.args['val_pipeline'],
                                     split=self.args['data']['val']['split'],
                                     multi_class=True,
                                     test_mode=True,
                                     num_classes=self.args['model']['cls_head']['args']['num_classes']
                                     )
        data_loader = dict()
        if self.reweighted:
            class_count_file = open(self.args['data']['train']['class_count'], 'r')
            class_count_data = json.load(class_count_file)
            if self.args['data']['train']['split'] == 'xsub_train':
                class_count = class_count_data['class_count_CS']
            else:
                class_count = class_count_data['class_count_CV']

            self.reweighted_idx = []
            self.ratio = np.array(class_count) / np.sum(np.array(class_count))
            for clc in class_count:
                self.reweighted_idx.append(min(class_count) / (clc + 1))

        data_loader['train'] = torch.utils.data.DataLoader(dataset=train_dataset,
                                                           batch_size=self.args['data']['videos_per_gpu'],
                                                           shuffle=True,
                                                           num_workers=self.args['data']['workers_per_gpu'],
                                                           drop_last=True,
                                                           worker_init_fn=init_seed)

        data_loader['val'] = torch.utils.data.DataLoader(dataset=val_dataset,
                                                         batch_size=self.args['data']['videos_per_gpu'],
                                                         shuffle=True,
                                                         num_workers=self.args['data']['workers_per_gpu'],
                                                         drop_last=False,
                                                         worker_init_fn=init_seed)

        return data_loader

    def make_model(self):
        model = Recognizer3D(backbone=self.args['model']['backbone']['args'],
                             cls_head=self.args['model']['cls_head']['args'])

        model = model.cuda(self.output_device)

        if type(self.args['device']) is list:
            if len(self.args['device']) > 1:
                model = nn.DataParallel(
                    model,
                    device_ids=self.args['device'],
                    output_device=self.output_device
                )
        return model

    def make_loss(self, loss_type):
        if loss_type == 'train':
            loss = binary_cross_entrophy_with_logits(self.ratio)
        elif loss_type == 'eval':
            if self.reweighted:
                loss = CrossEntrophyLoss(args=self.args, reweighted=self.reweighted,
                                         reweighted_idx=self.reweighted_idx)
            else:
                loss = CrossEntrophyLoss()
        return loss

    def make_optimizer(self, model):
        optimizer = Optimizer(self.args, model).get_optim()
        return optimizer

    def make_scheduler(self, optimizer, epoch=None):
        self.args['warm_up_epoch'] = 5
        scheduler = Schedulers(self.args, optimizer).get_schedulers(epoch=epoch)
        return scheduler

    def make_work_dirs_path(self):
        if os.path.isdir(self.args['model_saved_name']):
            print('log_dir: ', self.args['model_saved_name'], 'already exist')
            answer = input('delete it? y/n:')
            if answer == 'y':
                shutil.rmtree(self.args['model_saved_name'])
                print('Dir removed: ', self.args['model_saved_name'])
                input('Refresh the website of tensorboard by pressing any keys')
            else:
                print('Dir not removed: ', self.args['model_saved_name'])
        self.train_writer = SummaryWriter(os.path.join(self.args['model_saved_name'], 'train'), 'train')
        self.val_writer = SummaryWriter(os.path.join(self.args['model_saved_name'], 'val'), 'val')



    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.args['print_log']:
            with open('{}/log.txt'.format(self.args['work_dir']), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def train_step(self, model, optimizer: optim.Optimizer, epoch, scheduler, save_model=False):
        model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        train_loader = self.load_data()['train']

        lr = self.make_scheduler(optimizer=optimizer, epoch=epoch)

        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        loss = self.make_loss(loss_type='train')
        loss_per_batch = 0
        acc_value = []
        for batch_idx, (results, index) in enumerate(tqdm(train_loader, ncols=40)):

            data = torch.squeeze(results['imgs'], 1)
            label = torch.squeeze(results['label'], 1)

            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            #forward
            output = model(data)
            targets = one_hot(label).cuda(self.output_device)
            loss_batch = loss(output, targets)
            optimizer.zero_grad()
            loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=self.args['optimizer_config']['grad_clip']['max_norm'],
                                           norm_type=self.args['optimizer_config']['grad_clip']['norm_type'])
            optimizer.step()
            loss_per_batch += loss_batch.data.item()

            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())

            timer['statistics'] += self.split_time()

        scheduler.step()
        loss_per_batch /= len(train_loader.dataset)
        train_accuracy = np.mean(acc_value) * 100

        lr = optimizer.param_groups[0]['lr']
        print('Learning rate', lr)

        self.train_writer.add_scalar('acc', train_accuracy, self.global_step)
        self.train_writer.add_scalar('loss_train', loss_per_batch, self.global_step)

        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(loss_per_batch,
                                                                                train_accuracy))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights,
                       self.args['model_saved_name'] + '-' + str(epoch + 1) + '-' + str(int(self.global_step)) + '.pt')

        return loss_per_batch, train_accuracy

    def eval_step(self, model, epoch, loader_name=['val'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            test_loader = self.load_data()[ln]
            loss = self.make_loss(loss_type='eval')

            for batch_idx, (results, index) in enumerate(tqdm(test_loader, ncols=40)):
                data = torch.squeeze(results['imgs'], 1)
                label = torch.squeeze(results['label'], 1)

                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output = model(data)
                    loss_batch = loss(output, label)
                    loss_value.append(loss_batch.data.item())
                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
            loss_result = np.mean(loss_value)

            self.print_log('\tMean {} loss eval of {} batches: {}.'.format(
                ln, len(test_loader), np.mean(loss_value)))

            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)

            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum

            accuracy = np.mean(each_acc)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            self.print_log(
                '\nMean Accuracy: {:.4f}%.'.format(accuracy * 100))
            self.print_log('\nconfusion matrix: \n{}'.format(confusion))
            self.val_writer.add_scalar('loss', loss_result, self.global_step)
            self.val_writer.add_scalar('acc', accuracy, self.global_step)
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.args['work_dir'], epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)
            f.close()

        return loss_result, accuracy

    def epoch_step(self):
        self.make_work_dirs_path()
        model = self.make_model()
        self.args['start_epoch'] = 0
        self.global_step = self.args['start_epoch'] * len(self.load_data()['train']) / self.args['data'][
            'videos_per_gpu']
        
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        
        optimizer = self.make_optimizer(model=model)
        scheduler = self.make_scheduler(optimizer=optimizer)
        self.print_log(f'# Parameters: {count_parameters(model)}')
        for epoch in range(self.args['start_epoch'], self.args['num_epoch']):
            # save_model = (((epoch + 1) % self.args['save_interval'] == 0) or (
            #         epoch + 1 == self.args['num_epoch'])) and (epoch + 1) > self.args['save_epoch']
            save_model = True
            train_loss, train_accuracy = self.train_step(model=model, optimizer=optimizer, epoch=epoch,
                                                         scheduler=scheduler, save_model=save_model)
            test_loss, test_accuracy = self.eval_step(model=model, epoch=epoch, loader_name=['val'])
            metrics = {
                "Train/Train loss": train_loss,
                "Train/Train accuracy": train_accuracy
            }
            val_metrics = {
                "Val/Val loss": test_loss,
                "Val/Val accuracy": test_accuracy
            }
            wandb.log({**metrics, **val_metrics})



def main(args):
    config = args.config
    f = open(config, 'r')
    default_args = yaml.safe_load(f)
    train = TrainPipline(default_args)
    train.epoch_step()


if __name__ == '__main__':
    args = parse_args()
    wandb.login(relogin=True)
    wandb.init(project='ACTION_HOANG', name='experiment_train', entity='ai-iot', config=args)
    main(args)































