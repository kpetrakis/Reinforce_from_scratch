from neural_net import *
from itertools import count
import argparse


env = gym.make('CartPole-v1')
act_dim = env.action_space.n
obs_dim = env.observation_space.shape[0]

# reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
env.seed(0)


def sum_of_discounted_rewards(gamma,rewards):
    gamma_array = np.array([gamma**i for i in range(rewards.size)])
    sum_of_disc_rews = (rewards*gamma_array).sum()
    return sum_of_disc_rews

def rewards_to_go(rews):
    rews_to_go = [sum(rews[i:len(rews)]) for i in range(len(rews))] # undiscounted rewards to go!
    # rews_to_go = [sum_of_discounted_rewards(gamma,rews[i:len(rews)]) for i in range(len(rews))] # discounted rewards to go!
    return rews_to_go

# Policy Net instatiation
policy = PolicyNet(obs_dim, act_dim)

# command line 
parser = argparse.ArgumentParser("Reinforce from scratch")
parser.add_argument('-g','--gamma', type=float,default=0.99, help="Gamma parameter for sum of discountet rewards")
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('-e','--epochs', type=int, default=2000, help='Epochs to train on')
args = parser.parse_args()
argv = vars(args)

# Hyperparameters
gamma = argv['gamma']
lr = argv['learning_rate']
epochs = argv['epochs'] 

optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

def pick_action(probs):
    '''
        returns (action, log_prob[action])
        note: if called seperately 2nd time for log_prob[action] it might
        be erronous since it return always the same value 0.67..
    '''
    # the easy way:
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    action_log_prob = dist.log_prob(action)
    # the not so easy way :)
    # action = np.random.choice(valid_actions, p=torch.detach(probs).numpy()[0])
    # action_log_prob = torch.log(probs)[0][action].unsqueeze(0)
    return action.item(), action_log_prob

log_interval = 100 # log kathe 100 epochs!
ep_lens = []
for epoch in range(epochs):
    obs, done = env.reset(), False
    
    ep_rews = []
    action_log_probs = []

    ep_len = 0

    # collect a set of trajectories D_k by running policy ??_?? in the enviroment
    for t in count():
        obs = torch.tensor(obs,dtype=torch.float32).unsqueeze(0)

        probs = policy(obs) # probs = PolicyNet(obs) -> Softmax(logits) = probabilities of actions! -> p.x. [0.43 0.57]
        action, action_log_prob = pick_action(probs)

        action_log_probs.append(action_log_prob) # pusharw tensors

        obs, reward, done, info = env.step(action)

        ep_rews.append(reward)

        if done:
            obs = env.reset()

            rews_to_go = rewards_to_go(ep_rews)
            rews_to_go = torch.tensor(rews_to_go.copy(),dtype=torch.float32)
            action_log_probs_tensor = torch.cat(action_log_probs) # na tsekarw mipws prepei na orisw allo tensor gia tin anathesi p.x action_log_probs_tensor = ..

            ep_len = len(ep_rews) # i diarkeia tou epeisodiou isoutai me to reward!
            ep_lens.append(ep_len)
            if len(ep_lens)>=100:
                ep_lens_t = torch.tensor(ep_lens,dtype=torch.float32)
                means = ep_lens_t.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                # Log
                if epoch % log_interval==0:
                    print(f"Epoch = {epoch} | 100-mean Episode length = {means[-1].item()}")
            ep_rews = [] 
            action_log_probs = []
            break
    
    # print(f"Episode {epoch} | reward = {ep_len}")
    optimizer.zero_grad()
    loss = (-action_log_probs_tensor * rews_to_go).mean() # PROSOXI sto attr is_leaf kai grad
    loss.backward()
    # for param in policy.parameters():
        # param.grad.data.clamp_(-1,1)
    optimizer.step()


#plt.figure(figsize=(12,8))
#plt.xlabel("Episode")
#plt.ylabel("100-mean Episode Length")
#plt.plot(means)
#plt.show()
