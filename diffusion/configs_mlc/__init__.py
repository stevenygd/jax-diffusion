import ml_collections
from ml_collections import ConfigDict
from ml_collections.config_dict.config_dict import ConfigDict as ConfigDictType
from ml_collections.config_dict.config_dict import FieldReference as FieldReferenceType


def CFG(**data):
    return ConfigDict(initial_dictionary=data)


def update(cfg, **kwargs):
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def cfg2dict(cfg: ConfigDict):
    return cfg.to_dict()


def basic_config():
    return CFG(
        # Path configurations
        data_dir="./data",
        work_dir="./results",
        
        # Experiment details
        expr_name="test",

        # Train configurations 
        total_iters=400_000, 
        global_batch_size=256,
        grad_acc=1,
        global_seed=0,
        log_every=100,
        ckpt_every=25_000,
        ema_decay=0.9999,
        max_ckpt_keep=16,
        return_aux=True,
        
        dp_dim=-1,
        fsdp_dim=1,
        tp_dim=1,
    )


def ref_join(sep, lst):
    if len(lst) == 0:
        return ""
    ret = lst[0]
    for elt in lst[1:]:
        ret = ret + sep + elt
    return ret


def concretize(value):
    if isinstance(value, ConfigDictType) or isinstance(value, dict):
        out = {}
        for k in value.keys():
            out[k] = concretize(value[k])
        return CFG(**out) 

    if isinstance(value, list):
        out = []
        for v in value:
            out.append(concretize(v))
        return out 
    
    if isinstance(value, FieldReferenceType):
        return value.get()
    return value