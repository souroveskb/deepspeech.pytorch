from dataclasses import dataclass, field
from typing import Any
from typing import Optional


@dataclass
class ModelCheckpointConf:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    dirpath: Optional[str] = '../../snapshots'
    # filepath: Optional[str] = None
    monitor: Optional[str] = None
    verbose: bool = False
    save_last: Optional[bool] = True
    save_top_k: Optional[int] = 1
    save_weights_only: bool = False
    mode: str = "min"
    # dirpath: Any = None  # Union[str, Path, NoneType]
    filename: Optional[str] = "deepspeech_checkpoint_{epoch}"
    auto_insert_metric_name: bool = True
    every_n_train_steps: Optional[int] = None
    train_time_interval: Optional[str] = None
    every_n_epochs: Optional[int] = 1
    save_on_train_epoch_end: Optional[bool] = None


@dataclass
class TrainerConf:
    _target_: str = "pytorch_lightning.trainer.Trainer"
    logger: Any = (
        True  # Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]
    )
    enable_checkpointing: bool = True
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0
    callbacks: Any = None
    num_nodes: int = 1
    max_epochs: int = 16
    min_epochs: int = 1
    limit_train_batches: Any = 1.0  # Union[int, float]
    limit_val_batches: Any = 1.0  # Union[int, float]
    limit_test_batches: Any = 1.0  # Union[int, float]
    val_check_interval: Any = 1.0  # Union[int, float]
    log_every_n_steps: int = 50
    accelerator: Any = None  # Union[str, Accelerator, NoneType]
    sync_batchnorm: bool = False
    precision: int = 32
    num_sanity_val_steps: int = 2
    profiler: Any = None  # Union[BaseProfiler, bool, str, NoneType]
    benchmark: bool = False
    deterministic: bool = False
    detect_anomaly: bool = False
    plugins: Any = None  # Union[str, list, NoneType]
    devices: Any = None
    enable_progress_bar: bool = True
    max_time: Optional[str] = None
    limit_predict_batches: float = 1.0
    strategy: Optional[str] = "ddp"
    enable_model_summary: bool = True
    reload_dataloaders_every_n_epochs: int = 0
    use_distributed_sampler: bool = False
    
    # accelerator: Optional[str] = None
    # overfit_batches: Any = 0.0  # Union[int, float]
    # track_grad_norm: Any = -1  # Union[int, float, str]
    # check_val_every_n_epoch: int = 1  
    # fast_dev_run: Any = False  # Union[int, bool]
    # accumulate_grad_batches: Any = 1  # Union[int, Dict[int, int], List[list]] 
    # weights_save_path: Optional[str] = None
    # resume_from_checkpoint: Any = None  # Union[str, Path, NoneType]
    # auto_lr_find: Any = False  # Union[bool, str]
    # auto_scale_batch_size: Any = False  # Union[str, bool]
    # amp_backend: str = "native"
    # amp_level: Any = None
    # gradient_clip_algorithm: Optional[str] = None
    # ipus: Optional[int] = None
    # multiple_trainloader_mode: str = "max_size_cycle"
