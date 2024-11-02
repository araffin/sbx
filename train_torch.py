import gymnasium as gym
from sbx.bro.bro import BRO
import numpy as np
from make_dmc import make_env_dmc
import wandb

from absl import app, flags

flags.DEFINE_string('env_name', 'cheetah-run', 'Environment name.')
flags.DEFINE_string('benchmark', 'dmc', 'Environment name.')
flags.DEFINE_integer('learning_starts', 2000, 'Number of training steps to start training.')
flags.DEFINE_integer('training_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('gradient_steps', 2, 'Number of updates per step.')
flags.DEFINE_integer('n_quantiles', 100, 'Number of training steps.')
flags.DEFINE_integer('eval_freq', 25000, 'Eval interval.')
flags.DEFINE_integer('num_episodes', 5, 'Number of episodes used for evaluation.')
FLAGS = flags.FLAGS

'''
class flags:
    env_name: str = "cheetah-run"
    learning_starts: int = 5000
    training_steps: int = 10_000
    seed: int = 0
    batch_size: int = 128
    gradient_steps: int = 2
    use_wandb: bool = False
    n_quantiles: int = 100
    eval_freq: int = 5000
    num_episodes: int = 5
FLAGS = flags()
'''
    
def evaluate(env, model, num_episodes):
    returns = np.zeros(num_episodes)
    for episode in range(num_episodes):
        not_done = True
        obs, _ = env.reset(seed=np.random.randint(1e7))
        obs = np.expand_dims(obs, axis=0)
        ret = 0
        while not_done:
            action = model.policy.forward(obs, deterministic=True)[0]
            next_obs, reward, term, trun, info = env.step(action)
            next_obs = np.expand_dims(next_obs, axis=0)
            obs = next_obs
            ret += reward
            if term or trun :
                not_done = False
                returns[episode] = ret
    return {'return_eval': returns.mean()}

def log_to_wandb(step, infos):
    dict_to_log = {'timestep': step}
    for info_key in infos:
        dict_to_log[f'{info_key}'] = infos[info_key]
    wandb.log(dict_to_log, step=step)
    
def get_env(benchmark, env_name):
    if benchmark == 'gym':
        return gym.make(FLAGS.env_name)
    else:
        return make_env_dmc(env_name=FLAGS.env_name, action_repeat=1)

def main(_):
    SEED = np.random.randint(1e7)
    wandb.init(
        config=FLAGS,
        entity='naumix',
        project='BRO_SBX',
        group=f'{FLAGS.env_name}',
        name=f'BRO_Quantile:{FLAGS.n_quantiles}_BS:{FLAGS.batch_size}_{SEED}'
    )
    
    env = get_env(FLAGS.benchmark, FLAGS.env_name)
    eval_env = get_env(FLAGS.benchmark, FLAGS.env_name)
    model = BRO("MlpPolicy", env, learning_starts=FLAGS.learning_starts, verbose=0, n_quantiles=FLAGS.n_quantiles, seed=SEED, batch_size=FLAGS.batch_size, learning_starts=FLAGS.learning_starts, gradient_steps=FLAGS.gradient_steps)
    np.random.seed(SEED)
    
    reset_list = [20000]
    obs, _ = env.reset(seed=np.random.randint(1e7))
    obs = np.expand_dims(obs, axis=0)
    
    for i in range(1, FLAGS.training_steps+1):
        if i <= FLAGS.learning_starts:
            action = env.action_space.sample()
        else:
            action = model.policy.forward(obs, deterministic=False)[0]
        next_obs, reward, term, trun, info = env.step(action)
        next_obs = np.expand_dims(next_obs, axis=0)
    
        done = 1.0 if (term and not trun) else 0.0
            
        model.replay_buffer.add(obs, next_obs, action, reward, done, info)
        if term or trun:
            obs, _ = env.reset(seed=np.random.randint(1e7))
            obs = np.expand_dims(obs, axis=0)
        else:
            obs = next_obs
            
        if i in reset_list:
            model.reset()
            
        if i >= FLAGS.learning_starts:
            train_info = model.train(FLAGS.gradient_steps, FLAGS.batch_size)
            
        if i % FLAGS.eval_freq == 0:
            eval_info = evaluate(eval_env, model, FLAGS.num_episodes)
            stat_info = model.get_stats(FLAGS.batch_size)
            info = {**eval_info, **train_info, **stat_info}
            #print(eval_info)
            log_to_wandb(i, info)
        
if __name__ == '__main__':
    app.run(main)
