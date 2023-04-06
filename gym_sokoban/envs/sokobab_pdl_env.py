import gym
from gym.utils import seeding
from gym.spaces.discrete import Discrete
from gym.spaces import Box
from .room_utils import generate_room
from .render_utils import room_to_rgb, room_to_tiny_world_rgb
import numpy as np

ACTION_LOOKUP = {
    # 0: 'no operation',
    # 1: 'push up',
    # 2: 'push down',
    # 3: 'push left',
    # 4: 'push right',
    # 5: 'move up',
    # 6: 'move down',
    # 7: 'move left',
    # 8: 'move right',
    'hold': 'no operation',
    'up': 'push up',
    'down': 'move down',
    'left': 'move left',
    'right': 'move right',
}

CHANGE_COORDINATES = {
'up': (-1, 0),
'down': (1, 0),
'left': (0, -1),
'right': (0, 1)
}

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw']

class SokobanPDLEnv(gym.env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw']
    }
    

    
    def __init__(self,
                dim_room=(10, 10),
                max_steps=120,
                num_boxes=4,
                num_gen_steps=None,
                reset=True):
        self.dim_room = dim_room
        if num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.boxes_on_target = 0

        # Penalties and Rewards
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1
        self.reward_box_on_target = 1
        self.reward_finished = 10
        self.reward_last = 0

        # Other Settings
        self.viewer = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        
        if reset:
            self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]