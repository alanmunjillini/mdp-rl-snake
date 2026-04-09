import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
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
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        # TODO - MP10: Update the N-table. 
        self.N[state][action] += 1

    def update_q(self, s, a, r, s_prime):
        # TODO - MP10: Update the Q-table. 
        alpha = self.C / (self.C + self.N[s][a])
        td_target = r + self.gamma * np.max(self.Q[s_prime])
        self.Q[s][a] = self.Q[s][a] + alpha * (td_target - self.Q[s][a])

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y, bigfood_x, bigfood_y, bigfood_val] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        Note: bigfood gives bigfood_val points when eaten (vs 1 point for regular food). You may want to give higher rewards for eating bigfood.
        You may want to prioritize bigfood if its distance/value ratio is better than regular food's.
        '''

        # From a logical perspective, you can view s and a as prev_s and prev_a, 
        # while s_prime and a_prime are curr_s and curr_a. 

        # TODO - compute the reward. This depends on snake's death and food consumption.
        # Tip: Use points and self.points to calculate if, and what type of, food was consumed
        if dead:
            reward = -1
        else:
            reward = points - self.points
            if reward == 0:
                reward = -0.1

        # Update agent's self.points
        self.points = points

        # Generate the next game state s'. This will be used to update the Q and N tables.
        s_prime = self.generate_state(environment)

        # Choose the best action a' based on the Q-values and N-values.

        if self._train:
            # Tip: Update N and Q tables in here
            # TODO - MP10
            if self.s is not None and self.a is not None:
                self.update_n(self.s, self.a)
                if dead:
                    alpha = self.C / (self.C + self.N[self.s][self.a])
                    self.Q[self.s][self.a] = self.Q[self.s][self.a] + alpha * (reward - self.Q[self.s][self.a])
                else:
                    self.update_q(self.s, self.a, reward, s_prime)
            a_prime = self.get_best_action(s_prime, train_mode=True)
        else:
            # TODO - MP10
            a_prime = self.get_best_action(s_prime, train_mode=False)

        if dead:
            # Between episodes, the environment should be reset, but the Q and N tables remain
            # so that the agent can use information it's learned in its next try
            self.reset()
        else:
            self.s = s_prime
            self.a = a_prime

        return self.a

    
    def get_best_action(self, state, train_mode=True):
        '''
        :param state: the current state
        :param train_mode: boolean, if True, the agent will explore the environment
        :return: the best action to take
        '''
        # Given the Q-states for a state, choose the optimal action
        # For Q-update, this is just argmax_a Q(s, a)
        # For choosing action, we also consider exploration policy

        q_vals = self.Q[state]
        if train_mode:
            n_vals = self.N[state]
            action_values = np.where(n_vals < self.Ne, 1.0, q_vals)
        else:
            action_values = q_vals

        best_value = np.max(action_values)
        for action in (utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP):
            if action_values[action] == best_value:
                return action
        return int(np.argmax(action_values))

    def _calculate_target_food(self, snake_head_x, snake_head_y, food_x, food_y, 
                           bigfood_x, bigfood_y, bigfood_val):
        """
        :return: the coordinates of the target food
        Determines which food to target based on efficiency,
        the ratio between distance and value of the food.
        (Tip: Think about whether to use distance/value or value/distance,
        one of them avoids worrying about dividing by zero)
        """
        # TODO - MP10: Implement this helper function
        if bigfood_x is None or bigfood_y is None:
            return food_x, food_y

        small_dist = abs(food_x - snake_head_x) + abs(food_y - snake_head_y)
        big_dist = abs(bigfood_x - snake_head_x) + abs(bigfood_y - snake_head_y)

        small_score = float("inf") if small_dist == 0 else 1.0 / small_dist
        big_score = float("inf") if big_dist == 0 else float(bigfood_val) / big_dist

        if big_score > small_score:
            return bigfood_x, bigfood_y
        return food_x, food_y

    def _calculate_food_direction(self, snake_head_x, snake_head_y, target_food_x, target_food_y):
        """
        :return: the direction of the target food relative to the snake head

        Calculates the direction of the target food relative to the snake head.
        Returns food_dir_x and food_dir_y (0=same, 1=negative direction, 2=positive direction).
        """
        # TODO - MP10: Implement this helper function
        if target_food_x == snake_head_x:
            food_dir_x = 0
        elif target_food_x < snake_head_x:
            food_dir_x = 1
        else:
            food_dir_x = 2

        if target_food_y == snake_head_y:
            food_dir_y = 0
        elif target_food_y < snake_head_y:
            food_dir_y = 1
        else:
            food_dir_y = 2

        return food_dir_x, food_dir_y
    
    def _calculate_adjoining_obstacles(self, snake_head_x, snake_head_y, snake_body, rock_x, rock_y):
        """
        Determines if walls, rocks, or body segments are adjacent to the snake head.
        Returns 8 binary flags for obstacles in each direction.
        """
        # TODO - MP10: Implement this helper function
        rock_positions = {
            (rock_x, rock_y),
            (rock_x + 1, rock_y),
            (rock_x + 1, rock_y + 1),
        }

        left_blocked = (snake_head_x <= 1) or ((snake_head_x - 1, snake_head_y) in rock_positions)
        right_blocked = (snake_head_x >= self.display_width - 2) or ((snake_head_x + 1, snake_head_y) in rock_positions)
        top_blocked = (snake_head_y <= 1) or ((snake_head_x, snake_head_y - 1) in rock_positions)
        bottom_blocked = (snake_head_y >= self.display_height - 2) or ((snake_head_x, snake_head_y + 1) in rock_positions)

        if left_blocked:
            adjoining_wall_x = 1
        elif right_blocked:
            adjoining_wall_x = 2
        else:
            adjoining_wall_x = 0

        if top_blocked:
            adjoining_wall_y = 1
        elif bottom_blocked:
            adjoining_wall_y = 2
        else:
            adjoining_wall_y = 0

        body_set = set(snake_body)
        adjoining_body_top = 1 if (snake_head_x, snake_head_y - 1) in body_set else 0
        adjoining_body_bottom = 1 if (snake_head_x, snake_head_y + 1) in body_set else 0
        adjoining_body_left = 1 if (snake_head_x - 1, snake_head_y) in body_set else 0
        adjoining_body_right = 1 if (snake_head_x + 1, snake_head_y) in body_set else 0

        return (
            adjoining_wall_x,
            adjoining_wall_y,
            adjoining_body_top,
            adjoining_body_bottom,
            adjoining_body_left,
            adjoining_body_right,
        )

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y, bigfood_x, bigfood_y, bigfood_val] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        
        Takes in information from the environment and generates the state used by Q-learning.
        Prioritizes food based on distance/value efficiency.
        '''
        # Uses three helper functions you must implement

        [snake_head_x, snake_head_y, snake_body, food_x, food_y, 
        rock_x, rock_y, bigfood_x, bigfood_y, bigfood_val] = environment
        
        # Determine which food to target
        target_food_x, target_food_y = self._calculate_target_food(
            snake_head_x, snake_head_y, food_x, food_y, bigfood_x, bigfood_y, bigfood_val
        )
        
        # Calculate food direction
        food_dir_x, food_dir_y = self._calculate_food_direction(
            snake_head_x, snake_head_y, target_food_x, target_food_y
        )
        
        # Calculate adjoining obstacles
        adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, \
        adjoining_body_left, adjoining_body_right = self._calculate_adjoining_obstacles(
            snake_head_x, snake_head_y, snake_body, rock_x, rock_y
        )
        
        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y,
                adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
