# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import os
from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, kk, sat
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from deepscaler.rewards.math_reward import deepscaler_reward_fn, THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START
from typing import List, Union
from verl.mix_src.reward_with_format import deepscaler_reward_fn_impl1
from verl.mix_src.math_verify_reward import reward_fn_math_verify, reward_fn_math_verify_no_think

def deepscaler_reward_fn_nothink(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
    solution_str = f"{THOUGHT_DELIMITER_START}\n{THOUGHT_DELIMITER_END}\n{solution_str}"
    return deepscaler_reward_fn(solution_str, ground_truth, enable_llm)

def _select_rm_score_fn(data_source, reward_impl_version):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    elif 'kk' in data_source.lower():
        return kk.compute_score
    elif 'sat' in data_source.lower():
        return sat.compute_score
    else:
        if reward_impl_version == 0:
            return deepscaler_reward_fn
        elif reward_impl_version == 1:
            return deepscaler_reward_fn_impl1
        elif reward_impl_version == 2:
            return deepscaler_reward_fn_nothink
        elif reward_impl_version == 3:
            return reward_fn_math_verify
        elif reward_impl_version == 4:
            return reward_fn_math_verify_no_think
        else:
            raise NotImplementedError

class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, reward_impl_version) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_impl_version = reward_impl_version

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor
        from typing import Dict, Any
        #import threading
        # Thread-safe dict for tracking printed data sources
        # print_lock = threading.Lock()
        
        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences = valid_response_ids
            sequences_str = self.tokenizer.decode(sequences)
            # if not "no_think" in self.reward_impl_version:
            from deepscaler.globals import THOUGHT_DELIMITER_START
            # sequences_str = [THOUGHT_DELIMITER_START + seq.strip() for seq in sequences_str]
            if self.reward_impl_version != 4:
                sequences_str = THOUGHT_DELIMITER_START + '\n' + sequences_str
            # else:
            #     breakpoint()

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source, reward_impl_version=self.reward_impl_version)
            if data_source != 'sat':
                score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            elif data_source == 'sat':
                print('data_item.non_tensor_batch[reward_model]', data_item.non_tensor_batch['reward_model'])
                score = compute_score_fn(solution_str=sequences_str, clause=data_item.non_tensor_batch['reward_model']['clause'])
            
            # with print_lock:
            #     if data_source not in already_print_data_sources:
            #         already_print_data_sources[data_source] = 0

            #     if already_print_data_sources[data_source] < self.num_examine:
            #         already_print_data_sources[data_source] += 1
            #         print(sequences_str)      
            return i, score, valid_response_length

        if self.reward_impl_version in {3, 4}:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(process_item(args[i]) for i in range(len(args)))
        else:
            # Process items in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=96) as executor:
                args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
                results = list(executor.map(process_item, args))

        # Fill reward tensor with results
        for i, score, valid_response_length in results:
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='mix_ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        print(f"RAY_CLIENT_MODE is set to: {os.environ.get('RAY_CLIENT_MODE')}")
        ray.init(
            _node_ip_address="127.0.0.1",           # 只绑定本地回环
            # dashboard_port=8265,                    # 避开默认 8265
            # _temp_dir="/data2/taoyongding/tmp_ray",  # 临时目录到可写路径
            runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}},
            # include_dashboard=False                  # 禁用 dashboard 避免多占端口
        )

        # ray.init(
        #     runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}}
        # )

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        raise NotImplementedError('megatron is not supported')
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from .mix_fsdp_worker import MIXActorRolloutRefWorker

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }

    if config.actor_rollout_ref.ref.use_ref:
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(MIXActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(MIXActorRolloutRefWorker)
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
    else:
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(MIXActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, reward_impl_version=config.data.reward_impl_version)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, reward_impl_version=config.data.reward_impl_version)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    from .mix_trainer import MIXRayPPOTrainer
    if not config.trainer.acc_rebatch:
        trainer = MIXRayPPOTrainer(config=config,
                                tokenizer=tokenizer,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=val_reward_fn)
    else:
        from .mix_trainer_acc_rebatch import MIXRayPPOTrainerAccRebatch
        trainer = MIXRayPPOTrainerAccRebatch(config=config,
                                tokenizer=tokenizer,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
