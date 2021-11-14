import torch
import torch.utils.data.dataloader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm


class BasePredictor():
    def __init__(self,
                 batch_size=64,
                 max_epochs=10,
                 valid=None,
                 device=None,
                 metrics={},
                 learning_rate=1e-3,
                 max_iters_in_epoch=1e20,
                 grad_accumulate_steps=1):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.valid = valid
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.max_iters_in_epoch = max_iters_in_epoch
        self.grad_accumulate_steps = grad_accumulate_steps

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                       else 'cpu')

        self.epoch = 0

    def fit_dataset(self, data,  callbacks=[]):
        # Start the training loop.
        while self.epoch < self.max_epochs:
            if (self.epoch+1)%10 == 0:
                self.batch_size+=4
            self.batch_size = min(self.batch_size, 32)
            if self.epoch == 30:
                self.learning_rate=0.0001
            if self.epoch == 40:
                self.learning_rate=0.00001
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

            print('training epoch: ', self.epoch, "/", self.max_epochs)
            
            dataloader = torch.utils.data.DataLoader(
                                dataset=data,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=8
                            )

            # train epoch
            log_train = self._run_epoch(dataloader, True)

            # evaluate valid score
            if self.valid is not None:
                print('evaluating epoch: ', self.epoch, "/", self.max_epochs)
                
                dataloader = torch.utils.data.DataLoader(
                                dataset=self.valid,
                                batch_size=32,
                                shuffle=False,
                                num_workers=4
                            )
                log_valid = self._run_epoch(dataloader, False)
            else:
                log_valid = None

            print(log_train)
            print(log_valid)
            #print(callbacks)
            for callback in callbacks:
                #print("In")
                callback.on_epoch_end(log_train, log_valid, self)

            self.epoch += 1

    def predict_dataset(self, data,
                        collate_fn=default_collate,
                        batch_size=None,
                        predict_fn=None):
        if batch_size is None:
            batch_size = self.batch_size
        if predict_fn is None:
            predict_fn = self._predict_batch

        # set model to eval mode
        self.model.eval()

        dataloader = torch.utils.data.DataLoader(
                                dataset=data,
                                batch_size=100,
                                collate_fn=data.collate_fn,
                                shuffle=False
                            )

        ys_ = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch_y_ = predict_fn(batch)
                ys_.append(batch_y_)

        ys_ = torch.cat(ys_, 0)

        return ys_

    def save(self, path):
        torch.save({
            'epoch': self.epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        # TODO: Load saved model from path here.
        
        print("### Load Model ###")
        ckp = torch.load(path)
        
        self.model.load_state_dict(ckp['model'])

    def _run_epoch(self, dataloader, training):
        # set model training/evaluation mode
        if training:
            self.model.train()
        else:
            self.model.eval()


        # run batches for train
        loss = 0

        # reset metric accumulators
        for metric in self.metrics:
            metric.reset()

        if training:
            iter_in_epoch = min(len(dataloader), self.max_iters_in_epoch)
            description = 'training'
        else:
            iter_in_epoch = len(dataloader)
            description = 'evaluating'

        # run batches
        trange = tqdm(enumerate(dataloader),
                      total=iter_in_epoch,
                      desc=description)
        for i, batch in trange:
            if training and i >= iter_in_epoch:
                break

            if training:
                output, batch_loss = \
                    self._run_iter(batch, training)
                #print(batch_loss)
                batch_loss /= self.grad_accumulate_steps
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    output, batch_loss = \
                        self._run_iter(batch, training)

            # accumulate loss and metric scores
            loss += batch_loss.item()
            #for metric in self.metrics:
            #    metric.update(output, batch, at)
            
            #trange.set_postfix(
            #    loss=loss / (i + 1),
            #    **{m.name: m.print_score() for m in self.metrics})
            trange.set_postfix(
                loss=loss / (i + 1))

        # calculate averate loss and metrics
        loss /= iter_in_epoch

        epoch_log = {}
        epoch_log['loss'] = float(loss)
        #for metric in self.metrics:
            #score = metric.get_score()
            #print('{}: {} '.format(metric.name, score))
            #epoch_log[metric.name] = score
        print('loss=%f\n' % loss)
        
        return epoch_log

    def _run_iter(self, batch, training):
        """ Run iteration for training.

        Args:
            batch (dict)
            training (bool)

        Returns:
            predicts: Prediction of the batch.
            loss (FloatTensor): Loss of the batch.
        """
        pass

    def _predict_batch(self, batch):
        """ Run iteration for predicting.

        Args:
            batch (dict)

        Returns:
            predicts: Prediction of the batch.
        """
        pass
