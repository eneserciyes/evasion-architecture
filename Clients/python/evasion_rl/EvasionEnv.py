import gymnasium as gym
from gymnasium import spaces
# import pygame

import numpy as np
import random
from typing import Optional, List, Union, Tuple

from evasion_rl.Game import State, Config, HunterAction, PreyAction, Outcome, new_game, step_outcome, MAX_TICKS
from evasion_rl.Field import Horizontal, Vertical, Velocity, Point, MaxHeight, MaxWidth

def random_prey_move(game) -> PreyAction:
    """Calculate a random prey move and return it.
    """

    # Dummy random player: delete the code below and replace with your player.
    roll = random.random()
    if roll <= 0.8:
        return PreyAction(PreyAction.ActionType.NOOP)
    else:
        x = random.randint(-1, 1)
        y = random.randint(-1, 1)
        return PreyAction(PreyAction.ActionType.CHANGE_VELOCITY, Velocity(x, y))

def get_observation_from_game_state(game: State) -> np.ndarray:
    obs = [
        game.hunter_position.x / MaxWidth,
        game.hunter_position.y / MaxHeight,
        game.prey_position.x / MaxWidth,
        game.prey_position.y / MaxHeight,
        game.hunter_velocity.x,
        game.hunter_velocity.y,
        game.prey_velocity.x,
        game.prey_velocity.y,
    ]
    # add can_place_walls
    if game.hunter_last_wall is not None:
        obs.append(1 if game.ticker - game.hunter_last_wall >= game.config.next_wall_interval else 0)
    else:
        obs.append(1)
    # add ticker
    obs.append(game.ticker / MAX_TICKS)
    # add walls
    walls = []
    for wall in game.walls:
        if isinstance(wall, Horizontal):
            walls += [wall.x1 / MaxWidth, wall.y / MaxHeight, wall.x2 / MaxWidth, wall.y / MaxHeight]
        elif isinstance(wall, Vertical):
            walls += [wall.x / MaxWidth, wall.y1 / MaxHeight, wall.x / MaxWidth, wall.y2 / MaxHeight]
    for _ in range(game.config.max_walls - len(game.walls)):
        walls += [0, 0, 0, 0]
    obs += walls
    return np.array(obs, dtype=np.float32)

def find_compartment(position: Point, walls: List[Union[Horizontal, Vertical]]) -> Tuple[Vertical, Vertical, Horizontal, Horizontal]:
    """Find the compartment that the given position is in.
    """
    vertical_walls_right = [wall for wall in walls if isinstance(wall, Vertical) and wall.x > position.x]
    vertical_walls_left = [wall for wall in walls if isinstance(wall, Vertical) and wall.x < position.x]
    horizontal_walls_above = [wall for wall in walls if isinstance(wall, Horizontal) and wall.y > position.y]
    horizontal_walls_below = [wall for wall in walls if isinstance(wall, Horizontal) and wall.y < position.y]
    closest_right = min(vertical_walls_right, key=lambda wall: wall.x) if len(vertical_walls_right) > 0 else Vertical(MaxWidth, 0, MaxHeight)
    closest_left = max(vertical_walls_left, key=lambda wall: wall.x) if len(vertical_walls_left) > 0 else Vertical(0, 0, MaxHeight)
    closest_above = min(horizontal_walls_above, key=lambda wall: wall.y) if len(horizontal_walls_above) > 0 else Horizontal(MaxHeight, 0, MaxWidth)
    closest_below = max(horizontal_walls_below, key=lambda wall: wall.y) if len(horizontal_walls_below) > 0 else Horizontal(0, 0, MaxWidth)
    return closest_right, closest_left, closest_above, closest_below

def find_area(right: Vertical, left: Vertical, above: Horizontal, below: Horizontal) -> float:
    """Find the area of the compartment.
    """
    return (right.x - left.x) * (above.y - below.y)

def compute_reward(game: State) -> float:
    # compute reward during the game
    # check if hunter and prey is in the same compartment
    hunter_right, hunter_left, hunter_above, hunter_below = find_compartment(game.hunter_position, game.walls)
    prey_right, prey_left, prey_above, prey_below = find_compartment(game.prey_position, game.walls)
    if hunter_right == prey_right and hunter_left == prey_left and hunter_above == prey_above and hunter_below == prey_below:
        compartment_reward = -find_area(hunter_right, hunter_left, hunter_above, hunter_below) / (MaxWidth * MaxHeight)
    else:
        compartment_reward = -5.# * (1 - (find_area(hunter_right, hunter_left, hunter_above, hunter_below) / (MaxWidth * MaxHeight)))
    
    return compartment_reward # positive for existing in the right compartment, and negative in the wrong compartment.
    


class EvasionEnv(gym.Env):
    metadata = {"render.modes": ["human"], "render.fps": 30}

    def __init__(self, max_walls: int, next_wall_interval: int):
        super().__init__()
        # Define action and observation space
        # actions;
        # 0: noop
        # 1: create horizontal wall
        # 2: create vertical wall
        # 3-3+max_walls: remove n'th wall
        self.action_space = spaces.Discrete(3 + max_walls)
        # observations (all normalized to [0,1])
        # 0: hunter x
        # 1: hunter y
        # 2: prey x
        # 3: prey y
        # 4: hunter velocity x
        # 5: hunter velocity y
        # 6: prey velocity x
        # 7: prey velocity y
        # 8: can_place_walls (0 or 1)
        # 9: ticker (normalized by MAX_TICKS)
        # For 'current wall count' times:
        # 9 + (4n+1): wall.x1
        # 9 + (4n+2): wall.y1
        # 9 + (4n+3): wall.x2
        # 9 + (4n+4): wall.y2
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(10 + 4 * max_walls,), dtype=np.float32
        )
        # create the game
        self.max_walls = max_walls
        self.next_wall_interval = next_wall_interval
        self.game: State = new_game(
            Config(max_walls=max_walls, next_wall_interval=next_wall_interval)
        )
        # for rendering, use pygame
        # pygame.init()
        # self.screen = pygame.display.set_mode((MaxWidth, MaxHeight))

    def step_the_game(self, hunter_action: HunterAction, prey_action: Optional[PreyAction] = None):
        if prey_action is None:
            prey_action = random_prey_move(self.game) # get a random prey move if none is given    
        outcome = step_outcome(self.game, hunter_action=hunter_action, prey_action=prey_action)
        self.game = outcome.state
        if outcome.type == Outcome.OutcomeType.CONTINUES:
            reward = compute_reward(self.game)
            return get_observation_from_game_state(self.game), reward, False, False, {} 
        elif outcome.type == Outcome.OutcomeType.TIMEOUT:
            print("Prey escaped")
            return get_observation_from_game_state(self.game), 0, True, False, {}
        elif outcome.type == Outcome.OutcomeType.PREY_IS_CAUGHT:
            reward = 100*(1-self.game.ticker/MAX_TICKS) 
            print("Reward: ", reward)
            return get_observation_from_game_state(self.game), reward, True, False, {}

    def step(self, action):
        if action == 0:
            return self.step_the_game(HunterAction(HunterAction.ActionType.NOOP))
        elif action == 1:
            # if len(self.game.walls) >= self.max_walls or (self.game.hunter_last_wall is not None 
            #                                               and self.game.ticker - self.game.hunter_last_wall > self.next_wall_interval): 
            #     return self.step(0)
            hunter_x, hunter_y = self.game.hunter_position.x, self.game.hunter_position.y
            # find the closest vertical wall to the right of hunter
            vertical_walls_right = [wall.x for wall in self.game.walls if isinstance(wall, Vertical) and wall.x > hunter_x]
            wall_x2 = min(vertical_walls_right) if len(vertical_walls_right) > 0 else MaxWidth # -1 will come
            # find the closest vertical wall to the left of hunter
            vertical_walls_left = [wall.x for wall in self.game.walls if isinstance(wall, Vertical) and wall.x < hunter_x]
            wall_x1 = max(vertical_walls_left) if len(vertical_walls_left) > 0 else -1 # +1 will come

            hunter_action = HunterAction(HunterAction.ActionType.CREATE_WALL, Horizontal(hunter_y, wall_x1+1, wall_x2-1))
            return self.step_the_game(hunter_action=hunter_action)
        elif action == 2:
            # if len(self.game.walls) >= self.max_walls or (self.game.hunter_last_wall is not None 
            #                                               and self.game.ticker - self.game.hunter_last_wall > self.next_wall_interval): 
            #     return self.step(0)
            hunter_x, hunter_y = self.game.hunter_position.x, self.game.hunter_position.y
            # find the closest horizontal wall above hunter
            horizontal_walls_above = [wall.y for wall in self.game.walls if isinstance(wall, Horizontal) and wall.y > hunter_y]
            wall_y2 = min(horizontal_walls_above) if len(horizontal_walls_above) > 0 else MaxHeight
            # find the closest horizontal wall below hunter
            horizontal_walls_below = [wall.y for wall in self.game.walls if isinstance(wall, Horizontal) and wall.y < hunter_y]
            wall_y1 = max(horizontal_walls_below) if len(horizontal_walls_below) > 0 else -1

            hunter_action = HunterAction(HunterAction.ActionType.CREATE_WALL, Vertical(hunter_x, wall_y1+1, wall_y2-1))
            return self.step_the_game(hunter_action=hunter_action)
        else:
            # remove the n'th wall
            if len(self.game.walls) < (action-2): return self.step(0)
            wall_to_remove = self.game.walls[action-3]
            hunter_action = HunterAction(HunterAction.ActionType.REMOVE_WALLS, [wall_to_remove])
            return self.step_the_game(hunter_action=hunter_action)

    def reset(self, seed=None, options=None):
        self.game = new_game(Config(max_walls=self.max_walls, next_wall_interval=self.next_wall_interval))
        return get_observation_from_game_state(self.game), {} # obs, info

    def render(self):
        pass
        # self.screen.fill((255, 255, 255))
        # # draw hunter
        # pygame.draw.circle(self.screen, (0, 0, 255), (self.game.hunter_position.x, self.game.hunter_position.y), 5)
        # # draw prey
        # pygame.draw.circle(self.screen, (255, 0, 0), (self.game.prey_position.x, self.game.prey_position.y), 5)
        # # draw walls
        # for wall in self.game.walls:
        #     if isinstance(wall, Horizontal):
        #         pygame.draw.line(self.screen, (0, 0, 0), (wall.x1, wall.y), (wall.x2, wall.y))
        #     elif isinstance(wall, Vertical):
        #         pygame.draw.line(self.screen, (0, 0, 0), (wall.x, wall.y1), (wall.x, wall.y2))
        # # update the screen
        # pygame.display.update()

    def close(self):
        pass


if __name__ == "__main__":
    env = EvasionEnv(max_walls=5, next_wall_interval=20)
    obs = env.reset()
    done = False
    while not done:
        wall_to_remove = None if len(env.game.walls) == 0 else (2 + random.randint(0, len(env.game.walls)-1))
        roll = random.random()
        if roll <= 0.98:
            obs, reward, done, _, _ = env.step(0)
        elif roll <= 0.99:
            if len(env.game.walls) < env.max_walls and obs[8] > 0: # can_place_walls
                roll = random.random()
                if roll <= 0.5:
                    if roll <= 0.05 and wall_to_remove is not None:
                        obs, reward, done, _, _ = env.step(wall_to_remove)
                    obs, reward, done, _, _ = env.step(1)
                else:
                    if roll >= 0.95 and wall_to_remove is not None:
                        obs, reward, done, _, _ = env.step(wall_to_remove)
                    obs, reward, done, _, _ = env.step(2)
            else:
                obs, reward, done, _, _ = env.step(0)
        else:
            if wall_to_remove is not None:
                obs, reward, done, _, _ = env.step(wall_to_remove)
        env.render()
        pygame.time.wait(10)
        

    print("Reward:", reward)
    print("Walls:", env.game.walls)
