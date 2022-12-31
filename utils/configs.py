import os
from easydict import EasyDict as edict


def generate_path(env_name, method):
    return os.path.join(env_name, method)

def generate_paths(env_name, method):
    return os.path.join(env_name, method, 'checkpoints'), \
           os.path.join(env_name, method, 'videos'), \
           os.path.join(env_name, method, 'records')

def generate_paths_in_cfgs(cfgs: edict, env_name, method):
    if "logs" not in cfgs.keys():
        cfgs.logs = edict()
    if "checkpoint" not in cfgs.logs.keys():
        cfgs.logs.checkpoint = edict()
    if "video" not in cfgs.logs.keys():
        cfgs.logs.video = edict()
    if "print" not in cfgs.logs.keys():
        cfgs.logs.print = edict()
    checkpoint_path, video_path, print_path = generate_paths(env_name, method)
    cfgs.logs.checkpoint.suffix = checkpoint_path
    cfgs.logs.video.suffix = video_path
    cfgs.logs.print.suffix = print_path
    return cfgs


def recursive_update(a, b):
    for key in b.keys():
        if key in a.keys():
            dict_a = type(a[key]) is edict or type(a[key]) is dict
            dict_b = type(b[key]) is edict or type(b[key]) is dict
            if dict_a and dict_b:
                a[key] = recursive_update(a[key], b[key])
            elif not dict_a and not dict_b:
                a[key] = b[key]
            else:
                raise AttributeError('Invalid update.')
        else:
            a[key] = b[key]
    return a

def merge_cfgs(default, custom_cfgs):
    cfgs = recursive_update(edict(default.copy()), edict(custom_cfgs.copy()))
    if "path" not in cfgs.logs.checkpoint:
        assert "suffix" in cfgs.logs.checkpoint
        cfgs.logs.checkpoint.path = os.path.join(cfgs.logs.prefix, cfgs.logs.checkpoint.suffix)
    if "path" not in cfgs.logs.video:
        assert "suffix" in cfgs.logs.video
        cfgs.logs.video.path = os.path.join(cfgs.logs.prefix, cfgs.logs.video.suffix)
    if "path" not in cfgs.logs.print:
        assert "suffix" in cfgs.logs.print
        cfgs.logs.print.path = os.path.join(cfgs.logs.prefix, cfgs.logs.print.suffix)
    return cfgs
