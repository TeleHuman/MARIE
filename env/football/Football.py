import copy
import numpy as np
import gfootball.env as football_env
from gym.spaces import Discrete, Box

env_num_agents = {
    "academy_pass_and_shoot_with_keeper": 2,
    "academy_run_pass_and_shoot_with_keeper": 2,
    "academy_3_vs_1_with_keeper": 3,
    "academy_counterattack_easy": 4,
    "academy_counterattack_hard": 4,
    "academy_corner": 11,
    "academy_single_goal_versus_lazy": 11,

    # additional env_names: 
    # https://github.com/google-research/football/blob/9a9e35bcd1929a82c2b91eeed777e5e571f29d38/gfootball/doc/scenarios.md?plain=1#L20
    "academy_empty_goal_close": 3,
    "academy_empty_goal": 3,
    "academy_run_to_score": 3,
    "academy_run_to_score_with_keeper": 3,
}

class Football:
    def __init__(self, env_name, **kwargs):
        """args
        env_name='',
        stacked=False,
        representation='extracted',
        rewards='scoring',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
        write_video=False,
        dump_frequency=1,
        logdir='',
        extra_players=None,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
        channel_dimensions=(
            observation_preprocessing.SMM_WIDTH,
            observation_preprocessing.SMM_HEIGHT))
        """

        kwargs['dump_frequency'] = 1
        kwargs['extra_players'] = None
        kwargs['logdir'] = None
        kwargs['channel_dimensions'] = (96, 72)
        kwargs['representation'] = "simple115v2"
        kwargs['rewards'] = "scoring,checkpoints"
        kwargs["stacked"] = False

        kwargs['write_full_episode_dumps'] = False
        kwargs['write_goal_dumps'] = False
        kwargs['write_video'] = False
        kwargs['render'] = False
        kwargs['other_config_options'] = None

        kwargs['env_name'] = env_name
        
        if env_name in ["academy_run_to_score", "academy_run_to_score_with_keeper"]:
            kwargs['number_of_left_players_agent_controls'] = 0 # env_num_agents[env_name]
            kwargs['number_of_right_players_agent_controls'] = env_num_agents[env_name]
        else:
            kwargs['number_of_left_players_agent_controls'] = env_num_agents[env_name]
            kwargs['number_of_right_players_agent_controls'] = 0

        self.args = copy.deepcopy(kwargs)
        self.process_args(self.args)
        self.env = football_env.create_environment(**self.args)
        self.n_agents = env_num_agents[env_name]
        self.share_observation_space = self.repeat(self.get_state_space())
        self.observation_space = self.get_obs_shape()
        self.action_space = [Discrete(i) for i in self.env.action_space.nvec]
        self.avail_actions = self.get_avail_actions()

        # compatiable with MARIE
        self.n_actions = self.action_space[0].n
        self.n_obs = self.observation_space[0].shape[0]
        self.individual_action_space = self.action_space[0]
        self.discrete = True

    def wrap(self, l):
        d = {}
        for i in range(self.n_agents):
            d[i] = l[i]
        return d

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        if isinstance(actions, np.ndarray):
            obs, rew, done, info = self.env.step(actions.flatten())
        elif isinstance(actions, list):
            obs, rew, done, info = self.env.step(actions)
        else:
            raise NotImplementedError(f"Unsupported actions: {actions}, {type(actions)}")
        # obs, rew, done, info = self.env.step(actions.flatten())
        
        rewards = [[rew[0]]] * self.n_agents
        if self.img:
            obs = obs.transpose(0, 3, 1, 2)
        return (
            self.wrap(self.split(obs)),
            self.wrap(self.repeat(self.get_state())),
            self.wrap(rewards),
            self.wrap(self.repeat(done)),
            self.wrap(self.repeat(info)),
            self.avail_actions,
        )

    def reset(self):
        """Returns initial observations and states"""
        obs = self.env.reset()
        if self.img:
            obs = obs.transpose(0, 3, 1, 2)
        return self.wrap(self.split(obs)), self.wrap(self.repeat(self.get_state())), self.avail_actions

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)

    def process_args(self, args):
        if "channel_dimensions" in args:
            args["channel_dimensions"] = tuple(args["channel_dimensions"])
        if "logdir" in args and args["logdir"] is None:
            args["logdir"] = ""
        if "other_config_options" in args and args["other_config_options"] is None:
            args["other_config_options"] = {}
        if self.args["representation"] in ("pixels", "pixels_gray", "extracted"):
            self.img = True
        else:
            self.img = False

    def get_state_space(self):
        # state space is designed following Simple115StateWrapper.convert_observation
        # global states are included once, and the active one-hot encodings for all players are included.
        total_length = 115 + (self.n_agents - 1) * 11
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_length,),
            dtype=self.env.observation_space.dtype,
        )

    def get_state(self):
        # adapted from imple115StateWrapper.convert_observation
        raw_state = self.env.unwrapped.observation()

        def do_flatten(obj):
            """Run flatten on either python list or numpy array."""
            if type(obj) == list:
                return np.array(obj).flatten()
            return obj.flatten()

        s = []
        for i, name in enumerate(
            ["left_team", "left_team_direction", "right_team", "right_team_direction"]
        ):
            s.extend(do_flatten(raw_state[0][name]))
            # If there were less than 11vs11 players we backfill missing values
            # with -1.
            if len(s) < (i + 1) * 22:
                s.extend([-1] * ((i + 1) * 22 - len(s)))
        # ball position
        s.extend(raw_state[0]["ball"])
        # ball direction
        s.extend(raw_state[0]["ball_direction"])
        # one hot encoding of which team owns the ball
        if raw_state[0]["ball_owned_team"] == -1:
            s.extend([1, 0, 0])
        if raw_state[0]["ball_owned_team"] == 0:
            s.extend([0, 1, 0])
        if raw_state[0]["ball_owned_team"] == 1:
            s.extend([0, 0, 1])
        game_mode = [0] * 7
        game_mode[raw_state[0]["game_mode"]] = 1
        s.extend(game_mode)
        for obs in raw_state:
            active = [0] * 11
            if obs["active"] != -1:
                active[obs["active"]] = 1
            s.extend(active)
        return np.array(s, dtype=np.float32)

    def get_obs_shape(self):
        obs_sp = self.env.observation_space
        if self.img:
            w, h, c = self.env.observation_space.shape[1:]
            # w, h, c = self.env.observation_space.shape
            print(f"shape: {self.env.observation_space.shape}")
            return [
                Box(
                    low=obs_sp.low[idx].transpose(2, 0, 1),
                    high=obs_sp.high[idx].transpose(2, 0, 1),
                    shape=(c, w, h),
                    dtype=obs_sp.dtype,
                )
                for idx in range(self.n_agents)
            ]
        else:
            return [
                Box(
                    low=-np.inf,#obs_sp.low[idx],
                    high=np.inf,#obs_sp.high[idx],
                    shape=obs_sp.shape[1:],
                    dtype=obs_sp.dtype,
                )
                for idx in range(self.n_agents)
            ]

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]

    def split(self, a):
        return [a[i] for i in range(self.n_agents)]


if __name__ == "__main__":
    args = {
        "env_name": "academy_pass_and_shoot_with_keeper",
        "stacked": False,
        "representation": "simple115v2",
        "rewards": "scoring,checkpoints",
        "write_goal_dumps": False,
        "write_full_episode_dumps": False,
        "render": False,
        "write_video": False,
        "dump_frequency": 1,
        "number_of_left_players_agent_controls": 2,
        "number_of_right_players_agent_controls": 0,
        "channel_dimensions": [96, 72],
        }
    env = Football(env_name="academy_pass_and_shoot_with_keeper")
    env.reset()

    actions = [i.sample() for i in env.action_space]
    actions[0] = 10
    print(actions)
    env.step(actions)
    
    print(env.get_state_space())

# import gfootball.env as football_env
# import random
# class Football:
#     def __init__(self, env_name="simple_tag_v3", n_agents=3, height=42, width=42, seed=42):
#         random.seed(seed)
#         self.env = football_env.create_environment(
#                 env_name=env_name, stacked=False,
#                 logdir="./tmp/football",
#                 write_goal_dumps=False, write_full_episode_dumps=False, render=True,
#                 dump_frequency=0,
#                 number_of_left_players_agent_controls=n_agents,
#                 channel_dimensions=(height, width))
#         self.env.reset()
#         # env_info = self.env.get_env_info()
#         # self.n_obs = env_info["obs_shape"]
#         # self.n_actions = env_info["n_actions"]
#         self.n_agents = n_agents

#     def to_dict(self, l):
#         return {i: e for i, e in enumerate(l)}

#     def step(self, action_dict):
#         observation, reward, done, info = self.env.step(action_dict)
#         return observation, {i: reward for i in range(self.n_agents)}, \
#                {i: done for i in range(self.n_agents)}, info

#     def reset(self, seed=42):
#         self.env.reset()
#         return {i: obs for i, obs in enumerate(self.env.observation())}

#     def render(self, mode="rgb_array"):
#         self.env.render(mode=mode)

#     def close(self):
#         self.env.close()
