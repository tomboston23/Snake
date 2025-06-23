import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
      
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
  
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
       
        if state is None or action is None:
            return
        index = state + (action,)
        self.N[index] += 1
        # print(f"updating N for index {index}")

    def update_q(self, s, a, r, s_prime):
      
        if s is None or a is None:
            return
        Q = self.Q
        old = Q[s + (a,)]
        alpha = self.C / (self.C + self.N[s + (a,)])
        maxq = float('-inf')
        if s_prime is None:
            maxq = 0
        else:
            for action in self.actions:
                maxq = max(maxq, Q[s_prime + (action,)])

        new = old + alpha * (r + self.gamma * maxq - old)

        Q[s + (a,)] = new        

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        '''
        s_prime = self.generate_state(environment)
        if dead is True:
            self.update_q(self.s, self.a, -1, s_prime)
            self.update_n(self.s, self.a)
            self.reset()
            return None
        Q = self.Q
        N = self.N
        maxq = float('-inf')
        best = -1
        f_values = []
        for action in self.actions:
            index = s_prime + (action,)
            f = -1
            if N[index] < self.Ne:
                f = 1
            else:
                f = Q[index]
            if f >= maxq:
                maxq = f
                best = action
        self.update_n(s_prime, best)
        r = 0
        if points > self.points:
            r = 1
        self.update_q(self.s, self.a, r, s_prime)
        self.s = s_prime
        self.a = best
        self.points = points
        return best

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] = environment
        width = self.display_width
        height = self.display_height
        food_dir_x = 0
        if snake_head_x > food_x:
            food_dir_x = 1
        elif snake_head_x < food_x:
            food_dir_x = 2
        food_dir_y = 0
        if snake_head_y > food_y:
            food_dir_y = 1
        elif snake_head_y < food_y:
            food_dir_y = 2

        adjoining_wall_x = 0
        if snake_head_x == 1 or (snake_head_x - 2 == rock_x and snake_head_y == rock_y):
            adjoining_wall_x = 1
        elif snake_head_x == self.display_width - 2 or (snake_head_x + 1 == rock_x and snake_head_y == rock_y):
            adjoining_wall_x = 2

        adjoining_wall_y = 0
        if snake_head_y == 1 or (snake_head_y - 1 == rock_y and (snake_head_x == rock_x or snake_head_x - 1 == rock_x)):
            adjoining_wall_y = 1 
        elif snake_head_y == self.display_height - 2 or (snake_head_y + 1 == rock_y and (snake_head_x == rock_x or snake_head_x == rock_x + 1)):
            adjoining_wall_y = 2

        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0

        if (snake_head_x + 1, snake_head_y) in snake_body:
            adjoining_body_right = 1
        if (snake_head_x - 1, snake_head_y) in snake_body:
            adjoining_body_left = 1
        if (snake_head_x, snake_head_y + 1) in snake_body:
            adjoining_body_bottom = 1
        if (snake_head_x, snake_head_y - 1) in snake_body:
            adjoining_body_top = 1
        return (food_dir_x,food_dir_y,adjoining_wall_x,adjoining_wall_y,adjoining_body_top,
                adjoining_body_bottom,adjoining_body_left,adjoining_body_right)
