
import keras
import numpy as np
import os

SAVE_MODEL_EPOCHS = 5


class Experiment(keras.callbacks.Callback):
    def __init__(self, run_id, epoch_count=0, tl=[], dl=[]):
        self.run_id = run_id
        self.train_losses = tl
        self.dev_losses = dl
        self.epoch_count = epoch_count
        self.last_model_file = None
        self.report_fn = None

    def on_epoch_end(self, epoch, logs):
        # print logs
        self.train_losses.append(logs['loss'])
        self.dev_losses.append(logs['val_loss'])

        self.epoch_count += 1
        if self.epoch_count % SAVE_MODEL_EPOCHS == 0:
            self.last_model_file = self.model_filename(self.run_id, self.epoch_count)
            self.model.save(self.last_model_file)
            self.save()

            if self.report_fn is not None:
                self.report_fn()


    def save(self):
        np.savez(
            self.exp_filename(self.run_id),
            run_id=self.run_id,
            epoch_count=self.epoch_count,
            tl=np.array(self.train_losses),
            dl=np.array(self.dev_losses),
            last_model_file=self.last_model_file)

    @staticmethod
    def exp_filename(run_id):
        return "experiment_" + run_id + ".npz"

    @staticmethod
    def model_filename(run_id, epoch):
        return 'model-{}-{}.h5'.format(run_id, epoch)

    @classmethod
    def load(cls, run_id):
        fname = cls.exp_filename(run_id)
        print "loading from file:", fname
        npz = np.load(fname)
        run_id = str(npz['run_id'])
        epoch_count = int(npz['epoch_count'])
        tl = list(npz['tl'])
        dl = list(npz['dl'])
        last_model_file = str(npz['last_model_file'])
        experiment = Experiment(run_id, epoch_count, tl, dl)
        experiment.last_model_file = last_model_file
        return experiment

    @classmethod
    def create_or_load(cls, run_id):
        fname = cls.exp_filename(run_id)
        if os.path.exists(fname):
            prog = cls.load(run_id)
            print "loaded progress: ", prog.run_id, "training epochs:", prog.epoch_count
            return prog
        else:
            print "creating new progress for run id:", run_id
            return Experiment(run_id)



def train_model(run_id, model, X_train, y_train, epochs, batch_size=1000, report_fn=None):
    progress = Experiment.create_or_load(run_id)
    progress.report_fn = report_fn

    hist = model.fit(
        X_train.as_matrix(), y_train,
        batch_size=batch_size, validation_split=0.01, verbose=2,
        nb_epoch=epochs,
        callbacks=[progress],
        initial_epoch=progress.epoch_count
    )
    return progress
