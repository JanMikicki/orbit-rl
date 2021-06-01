import matplotlib.pyplot as plt 
import numpy as np
#from orbit_dqn import Agent
from orbit_reinforce import PG_Agent
from simple_dqn_torch_2020 import DQN_Agent
from ddpg_torch import DDPG_Agent
import keyboard
import random

class Env():
    def __init__(self, R, mu, m, timestep):
        self.timestep = timestep
        self.mu = mu
        self.R = R
        self.x = 0. #self.R
        self.y = -self.R #0.
        self.v_x = np.sqrt(self.mu / self.R)#0.
        self.v_y = 0.#np.sqrt(self.mu / self.R) # that's for circular
        self.m = m
        #self.targets = list(zip(x_targets, y_targets))
        self.max_v = 0
        self.max_acc = 0
        self.a = 0
        self.ecc = [0.0, 0.0]
    
    def reset(self):
        angle = random.random() * 2*np.pi
        self.x = np.cos(angle) * self.R
        self.y = np.sin(angle) * self.R
        # self.x = 0 #self.R
        # self.y = -self.R #0.
        #h = np.sqrt(1*self.mu*1.5)
        #a = h**2/(self.mu*(1-0.5**2))
        #print("a target: ", a)
        #print("h precalculated: ", h)
        if self.x >= 0 and self.y >= 0:
            self.v_x = -np.sin(angle) * np.sqrt(self.mu / self.R)
            self.v_y = np.cos(angle) * np.sqrt(self.mu / self.R)
        elif self.x < 0 and self.y >= 0:
            self.v_x = -np.cos(angle-np.pi/2) * np.sqrt(self.mu / self.R)
            self.v_y = -np.sin(angle-np.pi/2) * np.sqrt(self.mu / self.R)
        elif self.x <= 0 and self.y <= 0:
            self.v_x = np.sin(angle-np.pi) * np.sqrt(self.mu / self.R)
            self.v_y = -np.cos(angle-np.pi) * np.sqrt(self.mu / self.R)
        else:
            self.v_x = np.sin(-angle) * np.sqrt(self.mu / self.R)
            self.v_y = np.cos(-angle) * np.sqrt(self.mu / self.R)
        # self.v_x = np.sqrt(self.mu / self.R)#0.#h
        # self.v_y = 0#np.sqrt(self.mu / self.R) # that's for circular
        h = np.cross([self.x, self.y], [self.v_x, self.v_y])
        r = np.sqrt(self.x**2 + self.y**2)
        vh = np.cross([self.v_x, self.v_y, 0], [0, 0, h])[:2] / self.mu
        ecc = vh - [self.x, self.y]/r    
        a = h**2 / (self.mu * (1 - (ecc[0]**2 + ecc[1]**2) ))
        
        self.ecc = ecc
        self.a = a

        acc_x = (- self.mu * self.x / np.sqrt(self.x**2 + self.y**2)**3)
        acc_y = (- self.mu * self.y / np.sqrt(self.x**2 + self.y**2)**3)

        return self.x, self.y, self.v_x, self.v_y, acc_x, acc_y, a, ecc[0], ecc[1]

    def step(self, ux, uy):
        acc_x = (- self.mu * self.x / np.sqrt(self.x**2 + self.y**2)**3) # if self.x != 0 else 0.
        acc_y = (- self.mu * self.y / np.sqrt(self.x**2 + self.y**2)**3) # if self.y != 0 else 0.
        self.max_acc = max(self.max_acc, np.sqrt(acc_x**2 + acc_y**2))

        #ux = 1000 if abs(x) < 1e-1 and y < -0.9 else 0
        acc_x += ux/self.m
        acc_y += uy/self.m

        self.v_x += self.timestep * acc_x
        self.v_y += self.timestep * acc_y
        self.max_v = max(self.max_v, np.sqrt(self.v_x**2 + self.v_y**2))

        self.x   += self.timestep * self.v_x
        self.y   += self.timestep * self.v_y

        h = np.cross([self.x, self.y], [self.v_x, self.v_y])
        r = np.sqrt(self.x**2 + self.y**2)
        vh = np.cross([self.v_x, self.v_y, 0], [0, 0, h])[:2] / self.mu
        ecc = vh - [self.x, self.y]/r
        self.ecc = ecc

        a = h**2 / (self.mu * (1 - (ecc[0]**2 + ecc[1]**2) ))
        self.a = a

        target_a = np.sqrt(3/2)
        ra = ra = np.sqrt((target_a-a)**2)/2.0
        
        rex = np.sqrt((0 - ecc[0])**2)
        rey = np.sqrt((-0.2 - ecc[1])**2)
        reward = -(rex + rey + ra)/3
        
        if np.sqrt(self.x**2 + self.y**2) < 0.5*self.R:
            done = True
            reason = 'R too small'
            reward = -0.1
        elif np.sqrt(self.x**2 + self.y**2) > 4*self.R:
            done = True
            reason = 'R too big'
            reward = -0.1
        elif np.sqrt(self.v_x**2 + self.v_y**2) > 100:
            done = True
            reason = 'V'
            reward = -0.1
        elif rex < 0.01 and rey < 0.01 and ra < 0.01:
            reward = 0.
            done = True
            reason = 'reached orbit'
            print("reached orbit?")
        else:
            done = False
            reason = 'unknown'

        # closest_target = min(self.targets, key=lambda t: np.sqrt((t[0]-self.x)**2 + (t[1]-self.y)**2))
        # error_margin = 3*0.015

        # if np.sqrt((closest_target[0]-self.x)**2 + (closest_target[1]-self.y)**2) <= error_margin:
        #     reward = 1
        # else:
        #     reward = 0

        # not sure if accs are helpful
        return self.x, self.y, self.v_x, self.v_y, acc_x, acc_y, a, ecc[0], ecc[1], reward, done, reason



if __name__ == '__main__':

    R = 1
    mu = 1e3
    m = 2
    dt = 1e-3

    #scfig, scax = plt.subplots()
    fig, axis = plt.subplots(figsize=(5, 5))

    axis.set_xlim(-3*R, 3*R) 
    axis.set_ylim(-2*R, 4*R) 

    x = np.linspace(-3.0*R, 3.0*R, 100*R)
    y = np.linspace(-2.0*R, 4.0*R, 100*R)
    X, Y = np.meshgrid(x,y)
    C = X**2 + Y**2 - 1                 # Circle
    #E = 4*X**2 + 3*Y**2 - 6*Y - 9       # Ellipse (eccentricyty 0.5, a=2, b=sqrt(3))
    E = 25*X**2 + 24*Y**2 - 12*Y - 36    # Smaller ellipse (ecc 0.2, a=sqrt(3/2), b=5/4)
    plt.contour(X,Y,C,[0])
    elipse_contour = plt.contour(X,Y,E,[0], colors='red')
    # v = elipse_contour.collections[0].get_paths()[0].vertices
    # target_xs = v[::10,0]
    # target_ys = v[::10,1]
    # target_xs = np.delete(target_xs, [0, 1, 23, 24])
    # target_ys = np.delete(target_ys, [0, 1, 23, 24])
    # plt.scatter(target_xs, target_ys, color='green')

    env = Env(R, mu, m, dt)

    score_history = []
    score = 0
    n_episodes = 1000
    skip_disp = int(0.01/dt)
    #ep_len = int(10 / dt) / 2
    ep_len = 2800
    print("n episodes: ", n_episodes)
    print("skip_disp: ", skip_disp)
    print("ep len: ", ep_len, '\n')

    evalmode = True
    epsilon = 1.0 if not evalmode else 0.01
    # dqn_agent = DQN_Agent(gamma=0.99, epsilon=epsilon, batch_size=64, n_actions=3, eps_end=0.01,
    #               input_dims=[9], lr=0.001, eps_dec=1/(ep_len*0.5*n_episodes))          
    # dqn_agent.load_models(iteration='_final')
    

    ddpg_agent =  DDPG_Agent(alpha=0.000025, beta=0.00025, input_dims=[9], tau=0.001, env=env,
              batch_size=64,  layer1_size=256, layer2_size=128, n_actions=1)
    np.random.seed(0)

    # dueling_dqn_agent = Agent(gamma=0.99, epsilon=1.0, lr=5e-3, input_dims=[8],
    #                 n_actions=2, mem_size=1000000, eps_min=0.01, batch_size=32,  # was 64
    #                 eps_dec=1/n_episodes, replace=1000)

    # policy_gradient_agent = PG_Agent(lr=0.001, input_dims=[6], gamma=0.99, n_actions=2,
    #         l1_size=128, l2_size=128)

    #policy_gradient_agent.load_model("pg_net_197")
    #burn = np.random.randint(ep_len//10)
    ddpg_agent.load_models(folder='hopefully good ddpg weights/600', iteration='600')


    for i in range(n_episodes):
        display = False
        display_every = 100 if not evalmode else 1

        if not i % display_every:
            print(f'episode: {i}, prev score: {score}')
            #print(f'epsilon: {dqn_agent.epsilon}')

            # Make a new canvas and give it to our figure
            fig_manager = plt.figure().canvas.manager
            fig_manager.canvas.figure = fig
            fig.set_canvas(fig_manager.canvas)
            # if not evalmode:
            #     dqn_agent.save_models(i)
            display = True

        # if not i % 100:
        #     if not evalmode:
        #         dqn_agent.update_target_net()

        done = False
        score = 0
        reward = 0
        action = 5

        x, y, vx, vy, ax, ay, a, ecx, ecy = env.reset()
        ux, uy = 0, 0

        if display:
            points, = axis.plot(x, y, marker='o', linestyle='None')
        
        ctupid_cnt = 0
        while not done:
            
            if display and not ctupid_cnt % skip_disp:
                points.set_data(x, y)
                plt.pause(0.02)
                # print("a:", env.a, ", ecc: ", env.ecc)
                # print("reward: ", reward)
                # ecc (and tehrefore reward) oscillates slightly but not terribly (in 0.01 range)
                # ideal ec for this: [ 0.  -0.5]
                

            #action = dqn_agent.choose_action([x/2, y/2, vx/50, vy/50, ax/2000, ay/2000, a, ecx, ecy])
            action = ddpg_agent.choose_action([x/2, y/2, vx/50, vy/50, ax/2000, ay/2000, a, ecx, ecy])
            #action = dueling_dqn_agent.choose_action([x, y, vx, vy, ax, ay, ux, uy])
            #action = policy_gradient_agent.choose_action([x/2, y/2, vx/50, vy/50, ax/2000, ay/2000])

            prev_ux, prev_uy = ux, uy

            #this is specific to ddpg:
            if abs(action[0]) <= 0.1:
                ux = 0
            else:
                ux = action[0]

            # if action == 1:
            #     ux = 1.0
            #     uy = 0.0
            # elif action == 2:
            #     ux = -1.0
            #     uy = 0.0
            # elif action == 3:
            #     ux = 0.0
            #     uy = 1.0
            # elif action == 4:
            #     ux = 0.0
            #     uy = -1.0
            # else:
            #     ux = 0.0
            #     uy = 0.0

            # if keyboard.is_pressed('o'):  # if key 'q' is pressed 
            #     ux = 0.5
            #     uy = 0
            
            ux *= 50
            uy *= 50

            new_x, new_y, new_vx, new_vy, new_ax, new_ay, new_a, new_ecx, new_ecy, reward, done, reason = env.step(ux, uy)
            #reward -= ux/3000

            if ctupid_cnt == ep_len:
                done = True

            # dueling_dqn_agent.store_transition([x, y, vx, vy, ax, ay, prev_ux, prev_uy], action, 
            #                                 reward, [new_x, new_y, new_vx, new_vy, new_ax, new_ay, ux, uy], done)
            # dqn_agent.store_transition([x/2, y/2, vx/50, vy/50, ax/2000, ay/2000, a, ecx, ecy], action, 
            #                     reward, [new_x/2, new_y/2, new_vx/50, new_vy/50, new_ax/2000, new_ay/2000, new_a, new_ecx, new_ecy], done)
            # ddpg_agent.remember([x, y, vx, vy, ax, ay, prev_ux, prev_uy], action, reward,
            #                        [new_x, new_y, new_vx, new_vy, new_ax, new_ay, ux, uy], int(done))

            x, y = new_x, new_y
            vx, vy = new_vx, new_vy
            ax, ay = new_ax, new_ay
            a = new_a
            ecx, ecy = new_ecx, new_ecy

            score += reward
            ctupid_cnt += 1
            #policy_gradient_agent.store_rewards(reward)

            if not evalmode:
                ddpg_agent.learn()
                #dqn_agent.learn()
                #dueling_dqn_agent.learn()
                #dqn_agent.decrement_epsilon()

        if display:
            #points.remove()
            plt.figure()
            plt.plot(score_history)
            plt.pause(3)
            plt.close()
            print(f'animated episode: {i}, animation score: {score}')
            print("reason for ending: ", reason)
            plt.close()
        
        score_history.append(score)
        #policy_gradient_agent.learn()

    #ddpg_agent.save_models()
    #dqn_agent.save_models('_final')
    #dueling_dqn_agent.save_models()
    #policy_gradient_agent.save_model()

    rolling_avg = []

    for i in range(100, len(score_history)):
        rolling_avg.append(np.mean(score_history[i-100:i]))

    plt.figure()
    plt.plot(score_history)
    plt.plot(rolling_avg)
    plt.show()