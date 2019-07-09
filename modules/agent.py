import torch
import numpy as np
import os
from .utils import shuffle_tensor, policy

def rl_agent_train(model, env, num_epoch, step_max, epsilon, device, memory_size, train_freq, batch_size, mode, model_ast,
                   discount_rate, criterion, optimizer, update_model_ast_freq, epsilon_min, epsilon_reduce_freq, epsilon_reduce,
                   writer, rewards, losses, save_freq, checkpoint_dir, global_step):
    for epoch in range(num_epoch):
        state = env.reset()
        step = 0
        total_loss = 0
        total_reward = 0

        # add replay memory buffer for the agent
        state_memory = []
        action_memory = []
        reward_memory = []
        observation_memory = [] # observation is actually the next state for boostrap
        while True and step < step_max:
            action = policy(model=model, state=state, epsilon=epsilon, device=device)

            observation, reward, profit, _, __ = env(action)

            # add to memory buffer
            state_memory.append(state)
            action_memory.append(action)
            reward_memory.append(reward)
            observation_memory.append(observation)
            if len(state_memory) > memory_size:
                state_memory.pop(0)
                action_memory.pop(0)
                reward_memory.pop(0)
                observation_memory.pop(0)

            memory = (state_memory, action_memory, reward_memory, observation_memory)

            if len(state_memory) == memory_size:
                # train or update only in every train freq
                if global_step % train_freq == 0:
                    state_tensor = torch.Tensor(np.array(memory[0], dtype=np.float32)).to(device)
                    action_tensor = torch.Tensor(np.array(memory[1], dtype=np.int32)).to(device)
                    reward_tensor = torch.Tensor(np.array(memory[2], dtype=np.int32)).to(device)
                    observation_tensor = torch.Tensor(np.array(memory[3], dtype=np.float32)).to(device)

                    shuffle_index = shuffle_tensor(memory_size, device)
                    shuffle_state = torch.index_select(state_tensor, 0, shuffle_index)
                    shuffle_reward = torch.index_select(reward_tensor, 0, shuffle_index)
                    shuffle_action = torch.index_select(action_tensor, 0, shuffle_index)
                    shuffle_observation = torch.index_select(observation_tensor, 0, shuffle_index)

                    for i in range(memory_size)[::batch_size]:
                        batch_state = shuffle_state[i: i+batch_size, :].type('torch.FloatTensor').to(device)
                        batch_action = shuffle_action[i: i+batch_size].type('torch.FloatTensor').to(device)
                        batch_reward = shuffle_reward[i: i+batch_size].type('torch.FloatTensor').to(device)
                        batch_observation = shuffle_observation[i: i+batch_size, :].type('torch.FloatTensor').to(device)

                        q_eval = model(batch_state).gather(1, batch_action.long().unsqueeze(1))
                        q_next = model_ast(batch_observation).detach()

                        if mode == 'dqn':
                            q_target = batch_reward + discount_rate * q_next.max(1)[0]

                        elif mode == 'ddqn':
                            q_target = batch_reward + discount_rate * q_next.gather(1, torch.argmax(
                                model(batch_observation), dim=1, keepdim=True))

                        else:
                            raise ValueError('please input correct mode for rl agent, either "dqn", or "ddqn"')

                        optimizer.zero_grad()
                        loss = criterion(input=q_eval, target=q_target)
                        total_loss += loss.item()
                        loss.backward()
                        optimizer.step()

                if global_step % update_model_ast_freq == 0:
                    para = {k: 0.3 * v + 0.7 * model.state_dict()[k] for k, v in model_ast.state_dict().items()}
                    model_ast.load_state_dict(para)

            # epsilon
            if epsilon > epsilon_min and global_step % epsilon_reduce_freq == 0:
                epsilon -= epsilon_reduce

            total_reward +=reward
            state = observation
            step += 1
            global_step += 1

            # here writer is tensorboardX
            if writer and global_step % 500 == 0:
                writer.add_scalar('loss', total_loss, global_step=global_step)
                writer.add_scalar('reward', total_reward, global_step=global_step)

            if global_step % 500 == 0:
                print('-----------------------------------global step {}------------------------------------'.format(global_step))
                print('total_reward : {}'.format(total_reward))
                print('total_loss : {}'.format(total_loss))
                print('-----------------------------------------------------------------------------------------')

        rewards.append(total_reward)
        losses.append(total_loss)

        print('------------------------------------------epoch {}--------------------------------------------',format(epoch))
        print('total loss : {}'.format(total_loss))
        print('total reward : {}'.format(total_reward))
        print('-----------------------------------------------------------------------------------------------')

        if (epoch + 1) % save_freq == 0:
            checkpoint_state = {'epoch': epoch, 'state_dict': model.state_dict()}
            torch.save(checkpoint_state, os.path.join(checkpoint_dir, '{}_checkpoint.pth.tar'.format(str(epoch)+ '_' + str(global_step))))

        return rewards, losses