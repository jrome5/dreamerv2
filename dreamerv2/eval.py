import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings

try:
  import rich.traceback
  rich.traceback.install()
except ImportError:
  pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
# import ruamel.yaml as yaml
from ruamel.yaml import YAML

import agent
import common

from os.path import join, dirname, abspath

mypath = join(dirname(abspath(__file__)), "../../pyrfuniverse/")
# mypath = join(dirname(abspath(__file__)), "../../pyrfuniverse/")

sys.path.append(mypath)

from pyrfuniverse.envs.robotics import FrankaClothHangEnv as ClothEnv

from train import make_env


def main():
  yaml = YAML(typ='safe', pure=True)
  configs = yaml.load((
      pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
  config = common.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = common.Flags(config).parse(remaining)

  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  import tensorflow as tf
  tf.config.experimental_run_functions_eagerly(not config.jit)
  message = 'No GPU found. To actually train on CPU remove this assert.'
  assert tf.config.experimental.list_physical_devices('GPU'), message
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    # from tensorflow.keras.mixed_precision import experimental as prec
    from tensorflow.python.keras.mixed_precision.policy import set_global_policy
    set_global_policy('mixed_float16')

  eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
      capacity=config.replay.capacity // 10,
      datadir=config.datadir,
      minlen=config.dataset.length,
      maxlen=config.dataset.length,
      max_loaded_data=0))
  step = common.Counter(eval_replay.stats['total_steps'])

  outputs = [
      common.TerminalOutput(),
      common.JSONLOutput(logdir),
      common.TensorBoardOutput(logdir),
  ]

  logger = common.Logger(step, outputs, multiplier=config.action_repeat)

  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')

  print('Create envs.')
  num_eval_envs = min(config.envs, config.eval_eps)
  if config.envs_parallel == 'none':
    eval_envs = [make_env('eval', config, logdir) for _ in range(num_eval_envs)]
  else:
    make_async_env = lambda mode: common.Async(
        functools.partial(make_env, mode), config.envs_parallel)
    eval_envs = [make_async_env('eval') for _ in range(eval_envs)]
  act_space = eval_envs[0].act_space
  obs_space = eval_envs[0].obs_space
  eval_driver = common.Driver(eval_envs)
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
  eval_driver.on_episode(eval_replay.add_episode)

  print('Create agent.')
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  agnt = agent.Agent(config, obs_space, act_space, step)
  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(eval_dataset))
  agnt.load(logdir / 'variables.pkl')
  agnt.wm.load(logdir / 'wm_variables.pkl')
  eval_policy = lambda *args: agnt.policy(*args, mode='eval')

  while step < config.steps:
    logger.write()
    print('Start evaluation.')
    logger.add(agnt.report(next(eval_dataset)), prefix='eval')
    eval_driver(eval_policy, episodes=config.eval_eps)
  for env in eval_envs:
    try:
      env.close()
    except Exception:
      pass


if __name__ == '__main__':
  main()
