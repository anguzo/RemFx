import pytorch_lightning as pl
import torch
from einops import rearrange
from pytorch_lightning.callbacks import Callback
from torch import Tensor

from remfx import effects

ALL_EFFECTS = effects.Pedalboard_Effects


class AudioCallback(Callback):
    def __init__(self, sample_rate, log_audio, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_audio = log_audio
        self.log_train_audio = True
        self.sample_rate = sample_rate
        if not self.log_audio:
            self.log_train_audio = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Log initial audio
        if self.log_train_audio:
            x, y, _, _ = batch
            # Concat samples together for easier viewing in dashboard
            input_samples = rearrange(x, "b c t -> c (b t)").unsqueeze(0)
            target_samples = rearrange(y, "b c t -> c (b t)").unsqueeze(0)

            self.log_train_audio = False

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        x, target, _, rem_fx_labels = batch
        # Only run on first batch
        if batch_idx == 0 and self.log_audio:
            with torch.no_grad():
                # Avoids circular import
                from remfx.models import RemFXChainInference

                if isinstance(pl_module, RemFXChainInference):
                    y = pl_module.sample(batch)
                    effects_present_name = [
                        [
                            ALL_EFFECTS[i].__name__.replace("RandomPedalboard", "")
                            for i, effect in enumerate(effect_label)
                            if effect == 1.0
                        ]
                        for effect_label in rem_fx_labels
                    ]
                    for i, label in enumerate(effects_present_name):
                        self.log(f"{'_'.join(label)}", 0.0)
                else:
                    y = pl_module.model.sample(x)

            log_tensorboard_audio_batch(
                logger=trainer.logger,
                x=x,
                y=target,
                y_pred=y,
                sample_rate=self.sample_rate,
                max_items=10,
                global_step=trainer.current_epoch,
            )

    def on_test_batch_start(self, *args):
        self.on_validation_batch_start(*args)


def log_tensorboard_audio_batch(
    logger: pl.loggers.TensorBoardLogger,
    x: Tensor,
    y: Tensor,
    y_pred: Tensor,
    sample_rate: int,
    max_items: int = 10,
    global_step: int = 0,
):
    if not isinstance(logger, pl.loggers.TensorBoardLogger):
        return

    max_items = min(max_items, x.shape[0])

    for i in range(max_items):
        wet = x[i]

        logger.experiment.add_audio(
            tag=f"valid_example_{i}/x_(wet)_audio",
            snd_tensor=wet.cpu(),
            global_step=global_step,
            sample_rate=sample_rate,
        )

        dry = y[i]

        logger.experiment.add_audio(
            tag=f"valid_example_{i}/y_(dry)_audio",
            snd_tensor=dry.cpu(),
            global_step=global_step,
            sample_rate=sample_rate,
        )

        recovered = y_pred[i]

        logger.experiment.add_audio(
            tag=f"valid_example_{i}/y_pred_(recovered)_audio",
            snd_tensor=recovered.cpu(),
            global_step=global_step,
            sample_rate=sample_rate,
        )
