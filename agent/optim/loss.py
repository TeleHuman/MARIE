import numpy as np
import torch
import wandb
import torch.nn.functional as F

from agent.optim.utils import rec_loss, compute_return, state_divergence_loss, calculate_ppo_loss, \
    batch_multi_agent, log_prob_loss, info_loss, compute_lambda_returns
from agent.utils.params import FreezeParameters
from networks.dreamer.rnns import rollout_representation, rollout_policy
from agent.models.world_model import rollout_policy_trans
from agent.models.world_model_env import MAWorldModelEnv
from utils import symexp

import ipdb


def model_loss(config, model, obs, action, av_action, reward, done, fake, last):
    time_steps = obs.shape[0]
    batch_size = obs.shape[1]
    n_agents = obs.shape[2]

    embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
    embed = embed.reshape(time_steps, batch_size, n_agents, -1)

    prev_state = model.representation.initial_state(batch_size, n_agents, device=obs.device)
    prior, post, deters = rollout_representation(model.representation, time_steps, embed, action, prev_state, last)
    feat = torch.cat([post.stoch, deters], -1)
    feat_dec = post.get_features()

    reconstruction_loss, i_feat = rec_loss(model.observation_decoder,
                                           feat_dec.reshape(-1, n_agents, feat_dec.shape[-1]),
                                           obs[:-1].reshape(-1, n_agents, obs.shape[-1]),
                                           1. - fake[:-1].reshape(-1, n_agents, 1))
    reward_loss = F.smooth_l1_loss(model.reward_model(feat), reward[1:]) # 这里用feat很奇怪
    pcont_loss = log_prob_loss(model.pcont, feat, (1. - done[1:]))
    av_action_loss = log_prob_loss(model.av_action, feat_dec, av_action[:-1]) if av_action is not None else 0.
    i_feat = i_feat.reshape(time_steps - 1, batch_size, n_agents, -1)

    dis_loss = info_loss(i_feat[1:], model, action[1:-1], 1. - fake[1:-1].reshape(-1))
    div = state_divergence_loss(prior, post, config)

    model_loss = div + reward_loss + dis_loss + reconstruction_loss + pcont_loss + av_action_loss
    if np.random.randint(20) == 4:
        wandb.log({'Model/reward_loss': reward_loss, 'Model/div': div, 'Model/av_action_loss': av_action_loss,
                   'Model/reconstruction_loss': reconstruction_loss, 'Model/info_loss': dis_loss,
                   'Model/pcont_loss': pcont_loss})

    return model_loss

# transformer imagination
import sys
from tqdm import tqdm
def trans_actor_rollout(obs, av_actions, filled, tokenizer, world_model, actor, critic, config, env_type, **kwargs):
    # newly added
    rew_model = kwargs.get("external_rew_model", None)
    use_external_rew_model = (rew_model is not None)

    use_stack = kwargs.get("use_stack", False)
    stack_obs_num = kwargs.get("stack_obs_num", None)

    use_valuenorm = kwargs.get("use_valuenorm", False)
    value_normalizer = kwargs.get("value_normalizer", None)
    # ----------

    n_agents = obs.shape[2]
    sequence_length = obs.shape[1]
    items_list = []
    with FreezeParameters([tokenizer, world_model, critic, actor]): # [tokenizer, world_model]
        wm_env = MAWorldModelEnv(tokenizer=tokenizer, world_model=world_model, device=config.DEVICE, env_type=env_type)       
        items = rollout_policy_trans(wm_env, actor, critic,
                                     config.HORIZON,
                                     obs,
                                     av_actions,
                                     filled,
                                     use_stack = use_stack,
                                     stack_obs_num = stack_obs_num,
                                     )

    if use_external_rew_model:
        with torch.no_grad():
            new_rewards = rew_model(items["actor_feats"], items["actions"]).unsqueeze(-1)  # this is the no-repeating version
            items['rewards'] = symexp(new_rewards)

    returns = trans_critic_rollout(critic, items['critic_feats'], items['rewards'], items['discounts'], config,
                                   use_valuenorm = use_valuenorm, value_normalizer = value_normalizer)
    output = [items["actions"][:-1].detach(),
              items["av_actions"][:-1].detach() if items["av_actions"] is not None else None,
              items["old_policy"][:-1].detach(),
              items["actor_feats"][:-1].detach(),
              items["critic_feats"][:-1].detach(),
              returns.detach(),
              items["old_values"][:-1].detach()]
    return [batch_multi_agent(v, n_agents) for v in output]


def trans_critic_rollout(critic, critic_feats, imag_rewards, discounts, config,
                         use_valuenorm = False, value_normalizer = None):
    with FreezeParameters([critic]):
        if config.critic_average_r:
            imag_rewards = imag_rewards.mean(-2, keepdim=True)[:-1]
        else:
            imag_rewards = imag_rewards[:-1]

        discounts = discounts[:-1]
        value = critic(critic_feats)
        wandb.log({'Value/Max reward': imag_rewards.max(), 'Value/Min reward': imag_rewards.min(),
                   'Value/Reward': imag_rewards.mean(), 'Value/Discount': discounts.mean(),
                   'Value/Value': value.mean()})
        
        print(f"Value/Max reward: {imag_rewards.max():.4f}, Value/Min reward: {imag_rewards.min():.4f}, Value/Value: {value.mean():.4f}")
    
    returns = compute_return(imag_rewards, value[:-1], discounts, bootstrap=value[-1], lmbda=config.DISCOUNT_LAMBDA,
                             gamma=config.GAMMA, use_vn=use_valuenorm, vn=value_normalizer)
    
    return returns
# -----------------------


def actor_rollout(obs, action, last, model, actor, critic, config):
    n_agents = obs.shape[2]
    with FreezeParameters([model]):
        embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
        embed = embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)
        prev_state = model.representation.initial_state(obs.shape[1], obs.shape[2], device=obs.device)
        prior, post, _ = rollout_representation(model.representation, obs.shape[0], embed, action,
                                                prev_state, last)
        post = post.map(lambda x: x.reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1))
        items = rollout_policy(model.transition, model.av_action, config.HORIZON, actor, post) ### imagination rollouts
    imag_feat = items["imag_states"].get_features()
    imag_rew_feat = torch.cat([items["imag_states"].stoch[:-1], items["imag_states"].deter[1:]], -1)
    
    returns = critic_rollout(model, critic, imag_feat, imag_rew_feat, items["actions"],
                             items["imag_states"].map(lambda x: x.reshape(-1, n_agents, x.shape[-1])), config)
    output = [items["actions"][:-1].detach(),
              items["av_actions"][:-1].detach() if items["av_actions"] is not None else None,
              items["old_policy"][:-1].detach(), imag_feat[:-1].detach(), returns.detach()]
    return [batch_multi_agent(v, n_agents) for v in output]


def critic_rollout(model, critic, states, rew_states, actions, raw_states, config):
    with FreezeParameters([model, critic]):
        imag_reward = calculate_next_reward(model, actions, raw_states)
        imag_reward = imag_reward.reshape(actions.shape[:-1]).unsqueeze(-1).mean(-2, keepdim=True)[:-1]
        value = critic(states)
        discount_arr = model.pcont(rew_states).mean
        wandb.log({'Value/Max reward': imag_reward.max(), 'Value/Min reward': imag_reward.min(),
                   'Value/Reward': imag_reward.mean(), 'Value/Discount': discount_arr.mean(),
                   'Value/Value': value.mean()})
    returns = compute_return(imag_reward, value[:-1], discount_arr, bootstrap=value[-1], lmbda=config.DISCOUNT_LAMBDA,
                             gamma=config.GAMMA)
    return returns


def calculate_reward(model, states, mask=None):
    imag_reward = model.reward_model(states)
    if mask is not None:
        imag_reward *= mask
    return imag_reward


def calculate_next_reward(model, actions, states):
    actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    next_state = model.transition(actions, states)
    imag_rew_feat = torch.cat([states.stoch, next_state.deter], -1)
    return calculate_reward(model, imag_rew_feat)


def actor_loss(imag_states, actions, av_actions, old_policy, advantage, actor, ent_weight):
    _, new_policy = actor(imag_states)
    if av_actions is not None:
        new_policy[av_actions == 0] = -1e10
    actions = actions.argmax(-1, keepdim=True)
    rho = (F.log_softmax(new_policy, dim=-1).gather(2, actions) -
           F.log_softmax(old_policy, dim=-1).gather(2, actions)).exp()
    ppo_loss, ent_loss = calculate_ppo_loss(new_policy, rho, advantage)
    if np.random.randint(10) == 9:
        wandb.log({'Policy/Entropy': ent_loss.mean(), 'Policy/Mean action': actions.float().mean()})
        # print(f'in function actor loss, entropy is {ent_loss.detach().mean().item()}')
    return (ppo_loss + ent_loss.unsqueeze(-1) * ent_weight).mean()

## update
def continuous_actor_loss(imag_states, actions, av_actions, old_log_probs, advantage, actor, ent_weight, clip_param):
    action_log_probs, dist_entropy, _ = actor.evaluate_actions(imag_states, actions)

    imp_weights = torch.prod(
        torch.exp(action_log_probs - old_log_probs),
        dim=-1,
        keepdim=True,
    )
    surr1 = imp_weights * advantage
    surr2 = torch.clamp(imp_weights, 1.0 - clip_param, 1.0 + clip_param) * advantage

    policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
    actor_loss = policy_loss - dist_entropy * ent_weight
    # (policy_loss - dist_entropy * ent_weight).backward()
    if np.random.randint(10) == 9:
        wandb.log({'Policy/Entropy': dist_entropy.detach().item(), 'Policy/Mean action': actions.detach().float().mean().item()})
        # print(f'in function continuous actor loss, entropy is {dist_entropy.detach().item()}')
    return actor_loss

def value_loss(critic, imag_feat, targets):
    value_pred = critic(imag_feat)
    mse_loss = (targets - value_pred) ** 2 / 2.0
    return torch.mean(mse_loss)


# pylint: disable-next=invalid-name
def huber_loss(e, d):
    """Huber loss."""
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


# pylint: disable-next=invalid-name
def mse_loss(e):
    """MSE loss."""
    return e**2 / 2