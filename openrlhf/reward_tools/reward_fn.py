import torch
import re
import json
import gc
import ast
import numpy as np


def find_matched_keep_order_num(keywords, s):
    pattern = r'(?<!\w)(?:' + '|'.join(re.escape(phrase) for phrase in keywords) + r')(?!\w)'
    matches = re.findall(pattern, s, flags=re.IGNORECASE)
    return [m.lower() for m in matches]


def lcs_length(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if seq1[i].lower() == seq2[j].lower():
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def keyword_reward_func(ai_answer, keypoint_keywords, reference_answer):
    try:
        keypoint_keywords = ast.literal_eval(keypoint_keywords) if isinstance(keypoint_keywords, str) else keypoint_keywords
        if not keypoint_keywords:
            return 0.0
        
        max_lcs_scores = []
        for multi_ref_keywords in keypoint_keywords:
            max_lcs = 0.0

            for keywords in multi_ref_keywords:
                reference_keywords = find_matched_keep_order_num(keywords, reference_answer)
                ai_keywords = find_matched_keep_order_num(keywords, ai_answer)
                total_len = max(len(ai_keywords), len(reference_keywords))
                if total_len > 0:
                    lcs_score = lcs_length(ai_keywords, reference_keywords) / total_len
                else:
                    lcs_score = 0
                max_lcs = max(lcs_score, max_lcs)

            max_lcs_scores.append(max_lcs)
        avg_lcs_score = np.mean(max_lcs_scores)

        gen_len = len(ai_answer.split())
        ref_len = len(reference_answer.split())
        length_ratio = gen_len / ref_len
        if 0.9 <= length_ratio <= 1.2:
            length_penalty = 0
        else:
            length_penalty = abs(length_ratio - 1)

        reward = avg_lcs_score - 0.0 * length_penalty
        print(f'lcs_score: {avg_lcs_score}, length_penalty: {length_penalty}')
        return max(reward, 0.0)

    except Exception as e:
        print(f"Error: {e}")
        return 0.0


def style_reward_func(answer, json_str):
    def check_style(response, validation_code):
        try:
            local_scope = {}  
            exec(validation_code, {"re": re, "json": json}, local_scope)
            func_name = next(iter(local_scope.keys()))  
            validation_func = local_scope[func_name]  
            return validation_func(response)  
        except Exception as e:
            print(f"Validation error: {e}")
            return False
    
    res_dict = ast.literal_eval(json_str)
    score = 0.0
    for k in res_dict['key_points']:
        if check_style(answer, k['verification_code']):
            score += k['weight']
    return score


# @ray.remote(max_retries=3)
def reward_func(queries, prompts, labels):
    # queries is prompts + responses
    # labels is answers
    # print(queries)
    # return torch.randn(len(queries))

    responses = [q[len(p):] for q,p in zip(queries, prompts)]

    rewards = []
    for r, gt in zip(responses, labels):
        reward = 0

        reference_r = keyword_reward_func(r, gt.split("[[[SPLIT]]]")[1], gt.split("[[[SPLIT]]]")[0])

        style_r = style_reward_func(r, gt.split("[[[SPLIT]]]")[-1])

        reward += reference_r + style_r

        print(f'reference_r: {reference_r}, style_r: {style_r}, total_r: {reward}')

        del reference_r, style_r

        rewards.append(reward)

    results = torch.tensor(rewards)
    del rewards
    gc.collect()
    return results