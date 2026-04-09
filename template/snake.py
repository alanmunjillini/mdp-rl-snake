import random
import pygame
import utils

LENGTH_FACTOR = 40 # for all lengths, multiply it with this factor for pygame display

class SnakeEnv:
    def __init__(self, snake_head_x, snake_head_y, food_x, food_y,
                 display_width, display_height, rock_x, rock_y, bigfood_x=None, bigfood_y=None, bigfood_val=2, random_seed=None):
        self.game = Snake(snake_head_x, snake_head_y, food_x, food_y, display_width, display_height, rock_x, rock_y, bigfood_x, bigfood_y, bigfood_val)
        self.render = False
        if random_seed:
            random.seed(random_seed)

    def get_actions(self):
        return self.game.get_actions()

    def reset(self):
        return self.game.reset()
    
    def get_points(self):
        return self.game.get_points()

    def get_environment(self):
        return self.game.get_environment()

    def step(self, action):
        environment, points, dead = self.game.step(action)
        if self.render:
            self.draw(environment, points, dead)
        return environment, points, dead

    def draw(self, environment, points, dead):
        snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y, bigfood_x, bigfood_y, bigfood_val = environment
        self.display.fill(utils.BLUE)    
        pygame.draw.rect( self.display, utils.BLACK,
                [
                    LENGTH_FACTOR,
                    LENGTH_FACTOR,
                    (self.game.display_width -  2) * LENGTH_FACTOR,
                    (self.game.display_height - 2) * LENGTH_FACTOR
                ])
        
        # Draw rock
        pygame.draw.rect(
                    self.display, 
                    utils.GREY,
                    [
                        self.game.rock_x * LENGTH_FACTOR,
                        self.game.rock_y * LENGTH_FACTOR,
                        2 * LENGTH_FACTOR,
                        LENGTH_FACTOR
                    ]
                )
        
        # Below is added to draw L shaped rock
        pygame.draw.rect(
                    self.display, 
                    utils.GREY,
                    [
                        (self.game.rock_x + 1) * LENGTH_FACTOR,
                        self.game.rock_y * LENGTH_FACTOR,
                        LENGTH_FACTOR,
                        2 * LENGTH_FACTOR
                    ]
                )

        # Draw snake head
        pygame.draw.rect(
                    self.display, 
                    utils.GREEN,
                    [
                        snake_head_x * LENGTH_FACTOR,
                        snake_head_y * LENGTH_FACTOR,
                        LENGTH_FACTOR,
                        LENGTH_FACTOR
                    ],
                    3
                )

        # Draw snake body
        for seg in snake_body:
            pygame.draw.rect(
                self.display, 
                utils.GREEN,
                [
                    seg[0] * LENGTH_FACTOR,
                    seg[1] * LENGTH_FACTOR,
                    LENGTH_FACTOR,
                    LENGTH_FACTOR,
                ],
                1
            )

        # Draw food (small yellow square, 20% size, centered)
        small_food_size = 0.5 * LENGTH_FACTOR
        food_offset = (LENGTH_FACTOR - small_food_size) / 2  # Center the small square
        pygame.draw.rect(
                    self.display, 
                    utils.RED,
                    [
                        food_x * LENGTH_FACTOR + food_offset,
                        food_y * LENGTH_FACTOR + food_offset,
                        small_food_size,
                        small_food_size
                    ]
                )

        # Draw bigfood (full-size yellow square)
        if bigfood_x is not None and bigfood_y is not None:
            pygame.draw.rect(
                        self.display, 
                        utils.RED,
                    [
                        bigfood_x * LENGTH_FACTOR,
                        bigfood_y * LENGTH_FACTOR,
                        LENGTH_FACTOR,
                        LENGTH_FACTOR
                    ]
                )

        text_surface = self.font.render("Points: " + str(points), True, utils.BLACK)
        text_rect = text_surface.get_rect()
        text_rect.center = ((280),(25))
        self.display.blit(text_surface, text_rect)
        pygame.display.flip()
        if dead:
            # slow clock if dead
            self.clock.tick(1)
        else:
            self.clock.tick(5)

        return 

    def display(self):
        pygame.init()
        pygame.display.set_caption('MP6: Snake')
        self.clock = pygame.time.Clock()
        pygame.font.init()

        self.font = pygame.font.Font(pygame.font.get_default_font(), 15)
        self.display = pygame.display.set_mode((self.game.display_width * LENGTH_FACTOR, self.game.display_height * LENGTH_FACTOR), pygame.HWSURFACE)
        self.draw(self.game.get_environment(), self.game.get_points(), False)
        self.render = True
            
class Snake:
    def __init__(self, snake_head_x, snake_head_y, food_x, food_y, display_width, display_height, rock_x, rock_y, bigfood_x=None, bigfood_y=None, bigfood_val=2):
        self.init_snake_head_x = snake_head_x
        self.init_snake_head_y = snake_head_y
        self.init_food_x = food_x
        self.init_food_y = food_y
        self.init_bigfood_x = bigfood_x
        self.init_bigfood_y = bigfood_y
        self.display_width = display_width
        self.display_height = display_height
        self.rock_x = rock_x
        self.rock_y = rock_y
        self.bigfood_val = bigfood_val
        # todo: check if rock is in feasible region
        
        # This quantity mentioned in the spec, 8*16*8
        self.starve_steps = 8 * (self.display_height) * (self.display_width)
        self.reset()

    def reset(self):
        self.points = 0
        self.num_foods_eaten = 0
        self.steps = 0
        self.snake_head_x = self.init_snake_head_x
        self.snake_head_y = self.init_snake_head_y
        self.snake_body = []
        self.food_x = self.init_food_x
        self.food_y = self.init_food_y
        # Initialize bigfood position - use provided coordinates or randomize
        if self.init_bigfood_x is not None and self.init_bigfood_y is not None:
            self.bigfood_x = self.init_bigfood_x
            self.bigfood_y = self.init_bigfood_y
        else:
            self.random_bigfood()

    def get_points(self):
        # These points only updated when food eaten
        return self.points

    def get_actions(self):
        # Corresponds to up, down, left, right
        # return [0, 1, 2, 3]
        return utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

    def get_environment(self):
        return [
            self.snake_head_x,
            self.snake_head_y,
            self.snake_body,
            self.food_x,
            self.food_y, 
            self.rock_x,
            self.rock_y,
            self.bigfood_x,
            self.bigfood_y,
            self.bigfood_val
        ]

    def step(self, action):
        is_dead = self.move(action)
        return self.get_environment(), self.get_points(), is_dead

    def move(self, action):
        self.steps += 1

        delta_x = delta_y = 0

        # Up
        if action == utils.UP:
            delta_y = -1 
        # Down
        elif action == utils.DOWN:
            delta_y = 1
        # Left
        elif action == utils.LEFT:
            delta_x = -1 
        # Right
        elif action == utils.RIGHT:
            delta_x = 1

        old_body_head = None
        if len(self.snake_body) == 1:
            old_body_head = self.snake_body[0]

        # Snake "moves" by 1. adding previous head location to body, 
        self.snake_body.append((self.snake_head_x, self.snake_head_y))
        # 2. updating new head location via delta_x/y
        self.snake_head_x += delta_x
        self.snake_head_y += delta_y

        
        # 3. removing tail if body size greater than number of foods eaten
        if len(self.snake_body) > self.num_foods_eaten:
            del(self.snake_body[0])

        # Eats food, updates points, and randomly generates new food, if appropriate
        self.handle_eatfood()
        
        # Eats bigfood, updates points, and randomly generates new bigfood, if appropriate
        self.handle_eatbigfood()

        # Colliding with the snake body or going backwards while its body length
        # greater than 1
        if len(self.snake_body) >= 1:
            for seg in self.snake_body:
                if self.snake_head_x == seg[0] and self.snake_head_y == seg[1]:
                    return True

        # Moving towards body direction, not allowing snake to go backwards while 
        # its body length is 1
        if len(self.snake_body) == 1:
            if old_body_head == (self.snake_head_x, self.snake_head_y):
                return True

        # Check if collide with the wall (left, top, right, bottom)
        # Again, views grid position wrt top left corner of square
        if (self.snake_head_x < 1 or self.snake_head_y < 1 or
            self.snake_head_x + 1 > self.display_width-1 or self.snake_head_y + 1 > self.display_height-1):
            return True
            
        # Check if collide with the rock
        # Old Solution:
        # if ((self.snake_head_x == self.rock_x or self.snake_head_x == self.rock_x + 1) and self.snake_head_y == self.rock_y):
        #     return True

        # New Solution:
        if ((self.snake_head_x, self.snake_head_y) in [(self.rock_x, self.rock_y), (self.rock_x + 1, self.rock_y), (self.rock_x + 1, self.rock_y + 1)]):
            return True
        
        # looping for too long and starved
        if self.steps > self.starve_steps:
            return True

        return False

    def handle_eatfood(self):
        if (self.snake_head_x == self.food_x) and (self.snake_head_y == self.food_y):
            self.random_food()
            self.points += 1
            self.num_foods_eaten += 1
            self.steps = 0

    def handle_eatbigfood(self):
        if (self.bigfood_x is not None and self.bigfood_y is not None and
            self.snake_head_x == self.bigfood_x and self.snake_head_y == self.bigfood_y):
            self.random_bigfood()
            self.points += self.bigfood_val
            self.num_foods_eaten += 1
            self.steps = 0

    def _check_position_on_rock(self, x, y):
        """Generic helper to check if a position overlaps with rock."""
        # Old Solution:
        # return ((x == self.rock_x or x == self.rock_x + 1) and y == self.rock_y)

        # New Solution for L shaped rock:
        return ((x, y) in [(self.rock_x, self.rock_y), (self.rock_x + 1, self.rock_y), (self.rock_x + 1, self.rock_y + 1)])
               
    def _check_position_on_snake(self, x, y):
        """Generic helper to check if a position overlaps with snake."""
        if x == self.snake_head_x and y == self.snake_head_y:
            return True 
        for seg in self.snake_body:
            if x == seg[0] and y == seg[1]:
                return True
        return False

    def _random_position(self, max_x, max_y, exclude_rock=True, exclude_snake=True, exclude_food=False, exclude_bigfood=False):
        """Generic helper to generate a random position that doesn't overlap with specified obstacles."""
        x = random.randint(1, max_x)
        y = random.randint(1, max_y)
        
        while ((exclude_rock and self._check_position_on_rock(x, y)) or
               (exclude_snake and self._check_position_on_snake(x, y)) or
               (exclude_food and x == self.food_x and y == self.food_y) or
               (exclude_bigfood and self.bigfood_x is not None and self.bigfood_y is not None and 
                x == self.bigfood_x and y == self.bigfood_y)):
            x = random.randint(1, max_x)
            y = random.randint(1, max_y)
        
        return x, y

    def random_food(self):
        # Math looks at upper left corner wrt DISPLAY_SIZE for grid coordinates
        max_x = self.display_width - 2
        max_y = self.display_height - 2
        self.food_x, self.food_y = self._random_position(max_x, max_y, exclude_bigfood=True)
    
    def check_food_on_rock(self):
        return self._check_position_on_rock(self.food_x, self.food_y)
               
    def check_food_on_snake(self):
        return self._check_position_on_snake(self.food_x, self.food_y)

    def random_bigfood(self):
        # Math looks at upper left corner wrt DISPLAY_SIZE for grid coordinates
        max_x = self.display_width - 2
        max_y = self.display_height - 2
        self.bigfood_x, self.bigfood_y = self._random_position(max_x, max_y, exclude_food=True)

    def check_food_on_bigfood(self):
        if self.bigfood_x is None or self.bigfood_y is None:
            return False
        return (self.food_x == self.bigfood_x and self.food_y == self.bigfood_y)

    def check_bigfood_on_rock(self):
        if self.bigfood_x is None or self.bigfood_y is None:
            return False
        return self._check_position_on_rock(self.bigfood_x, self.bigfood_y)

    def check_bigfood_on_snake(self):
        if self.bigfood_x is None or self.bigfood_y is None:
            return False
        return self._check_position_on_snake(self.bigfood_x, self.bigfood_y)

    def check_bigfood_on_food(self):
        if self.bigfood_x is None or self.bigfood_y is None:
            return False
        return (self.bigfood_x == self.food_x and self.bigfood_y == self.food_y)
        
    
