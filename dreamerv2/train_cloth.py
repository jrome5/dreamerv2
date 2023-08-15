from os.path import join, dirname, abspath
import sys
import api as dv2

mypath = join(dirname(abspath(__file__)), "../../pyrfuniverse/")
sys.path.append(mypath)

from pyrfuniverse.envs.robotics import FrankaClothHangEnv

if __name__ == "__main__":

    resolution = 64

    env = FrankaClothHangEnv(
        executable_file="@editor",
        )

    config = dv2.defaults.update({
        'logdir': '~/logdir/dreamercloth/replay_franka_gripper_rgb5',
        'log_every': 1e4,
        'prefill': 1e5,
        'pretrain': 100,
        'render_size': [resolution,resolution],
    }).parse_flags()

    dv2.train(env, config)