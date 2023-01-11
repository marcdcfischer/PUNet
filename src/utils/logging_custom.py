from datetime import datetime
import pytorch_lightning as pl
import pathlib as plb


def setup_loggers(hparams, path_root):

    timestamp = datetime.now().strftime('%m%d%H%M%S')
    run_name = f'{plb.Path(path_root).stem}_{timestamp}'
    hparams.__dict__['run_name'] = run_name
    log_dir = plb.Path(path_root) / 'logs'
    log_dir.mkdir(exist_ok=True, parents=True)
    hparams.default_root_dir = str(log_dir / 'lightning')

    loggers = list()
    loggers.append(pl.loggers.TensorBoardLogger(name=run_name,
                                                save_dir=str(log_dir / 'tensorboard')))

    if hparams.online_logger.lower() == 'wandb':
        loggers.append(pl.loggers.wandb.WandbLogger(name=run_name,
                                                    save_dir=log_dir,
                                                    project=hparams.online_project,
                                                    tags=hparams.tags,
                                                    entity=hparams.online_entity,
                                                    offline=not hparams.online_on))
    else:
        raise NotImplementedError()

    return loggers
