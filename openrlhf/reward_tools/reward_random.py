import torch
import gc
import random


# @ray.remote(max_retries=3)
def reward_func(queries, prompts, labels):
    # queries is prompts + responses
    # labels is answers
    # print(queries)
    # return torch.randn(len(queries))

    responses = [q[len(p):] for q,p in zip(queries, prompts)]

    rewards = []
    for r, gt in zip(responses, labels):
        reward = random.uniform(0, 1)
        rewards.append(reward)

    results = torch.tensor(rewards)
    del rewards
    gc.collect()
    return results