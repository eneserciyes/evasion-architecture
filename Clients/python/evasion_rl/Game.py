from evasion_rl.Field import *
from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto
from datetime import datetime

MAX_TICKS = 24000

@dataclass
class Config:
    next_wall_interval: int
    max_walls: int

@dataclass
class State:
    config: Config
    ticker: int
    hunter_position: Point
    hunter_velocity: Velocity
    hunter_last_wall: Optional[int]
    prey_position: Point
    prey_velocity: Velocity
    walls: List[Union[Horizontal, Vertical]]


class HunterAction:
    class ActionType(Enum): REMOVE_AND_CREATE = auto(); REMOVE_WALLS=auto(); CREATE_WALL=auto(); NOOP=auto()
    def __init__(self, type: ActionType, *args):
        self.type=type
        self.args=args

class PreyAction:
    class ActionType(Enum): CHANGE_VELOCITY = auto(); NOOP=auto()
    def __init__(self, type: ActionType, *args):
        self.type=type
        self.args=args

class Outcome:
    class OutcomeType(Enum): CONTINUES = auto(); PREY_IS_CAUGHT = auto(); TIMEOUT = auto()
    def __init__(self, type: OutcomeType, state: State):
        self.type=type
        self.state=state

# --------------------------------
# LOGGING FUNCTIONS
# --------------------------------

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(message: str):
    print(f"[{now()}] {message}")

def log_yellow(message: str):
    print(f"\033[93m[{now()}] {message}\033[0m")

def log_green(message: str):
    print(f"\033[92m[{now()}] {message}\033[0m")

def log_blue(message: str):
    print(f"\033[94m[{now()}] {message}\033[0m")

def log_red(message: str):
    print(f"\033[91m[{now()}] {message}\033[0m")

# --------------------------------
# UTILITY FUNCTIONS
# --------------------------------

def new_game(config: Config) -> State:
    return State(
        config=config,
        ticker=0,
        hunter_position=Point(0, 0),
        hunter_velocity=Velocity(1, 1),
        hunter_last_wall=None,
        prey_position=Point(230, 200),
        prey_velocity=Velocity(0, 0),
        walls=[]
    )

def wall_is_valid(game: State, wall: Union[Horizontal, Vertical]) -> bool:
    # The wall collides the current hunter (i.e. touches hunter's position before moving)
    touches_hunter_before_move = wall_collides_point(wall, game.hunter_position)
    # The wall does not collide with the hunter after the hunter moves (i.e. checks edge case if hunter bounces off wall)
    does_not_collide_hunter = not wall_collides_point(wall, step_and_bounce(game.hunter_velocity, game.walls, game.hunter_position)[0])
    # The hunter can create a wall at this time step
    after_wall_interval = game.hunter_last_wall is None or game.ticker - game.hunter_last_wall >= game.config.next_wall_interval
    # The hunter has not exceeded the maximum number of walls
    has_not_exceeded_max_walls = len(game.walls) < game.config.max_walls
    # The wall does not collide with the current position of the prey
    does_not_collide_prey = not wall_collides_point(wall, game.prey_position)
    # The wall is within bounds
    within_bounds = wall_in_bounds(wall)
    # The wall does not collide with any other wall
    does_not_collide_walls = not wall_collides_walls(wall, game.walls)
    # The end points of the wall are greater than or equal to the start points
    if isinstance(wall, Horizontal):
        end_points_after_start_points = wall.x2 >= wall.x1
    elif isinstance(wall, Vertical):
        end_points_after_start_points = wall.y2 >= wall.y1

    return (touches_hunter_before_move 
            and does_not_collide_hunter 
            and after_wall_interval 
            and has_not_exceeded_max_walls 
            and does_not_collide_prey 
            and within_bounds 
            and does_not_collide_walls
            and end_points_after_start_points)    

# Step the game forward by one unit of time, given a hunter action and a prey action.
# Note that:
# - Prey actions can only be taken on odd ticks. Actions on even ticks are ignored.
# - The order of calculation is create/destroy walls, move the hunter, move the prey.

def step(game: State, hunter_action: HunterAction, prey_action: PreyAction) -> State:
    # create and destroy walls

    def createDestroyWalls(g:State, ha:HunterAction) -> Tuple[List[Union[Horizontal, Vertical]], Optional[int]]:
        if ha.type == HunterAction.ActionType.REMOVE_AND_CREATE:
            walls, wall = ha.args
            new_walls = remove_walls(walls, g.walls)
            g.walls = new_walls
            if wall_is_valid(g, wall):
                new_walls.append(wall)
                return new_walls, g.ticker
            else:
                return new_walls, g.hunter_last_wall
        elif ha.type == HunterAction.ActionType.CREATE_WALL:
            wall, = ha.args # expand the tuple
            if wall_is_valid(g, wall):
                g.walls.append(wall)
                return g.walls, g.ticker
            else:
                return g.walls, g.hunter_last_wall
        elif ha.type == HunterAction.ActionType.REMOVE_WALLS:
            walls, = ha.args # expand the tuple
            return remove_walls(walls, g.walls), g.hunter_last_wall
        elif ha.type == HunterAction.ActionType.NOOP:
            return g.walls, g.hunter_last_wall
    
    walls, lastWall = createDestroyWalls(game, hunter_action) 

    # Calculate the new hunter position
    new_hunter_position, new_hunter_velocity = step_and_bounce(game.hunter_velocity, walls, game.hunter_position)

    # Change the prey velocity if the prey action is valid, i.e. ticker is odd tick
    proposed_prey_velocity = (prey_action.args[0] 
                              if prey_action.type == PreyAction.ActionType.CHANGE_VELOCITY and game.ticker % 2 == 1 
                              else game.prey_velocity)
    
    # Calculate the new prey position
    new_prey_position, new_prey_velocity = (step_and_bounce(proposed_prey_velocity, walls, game.prey_position) if game.ticker % 2 == 1
                                            else (game.prey_position, proposed_prey_velocity))

    # Return the new game state
    return State(
        config=game.config,
        ticker=game.ticker + 1,
        hunter_position=new_hunter_position,
        hunter_velocity=new_hunter_velocity,
        hunter_last_wall=lastWall,
        prey_position=new_prey_position,
        prey_velocity=new_prey_velocity,
        walls=walls
    )

def points_between(p1: Point, p2: Point):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    D = 2*dy - dx
    y = p1.y
    points = []
    for x in range(p1.x, p2.x+1): # +1 to include the end point
        points.append(Point(x, y))
        if D > 0:
            y = y + 1
            D = D - 2*dx
        D = D + 2*dy
    return points

def step_outcome(game: State, hunter_action:HunterAction, prey_action: PreyAction) -> Outcome:
    new_state = step(game, hunter_action, prey_action)
    if new_state.ticker >= MAX_TICKS:
        return Outcome(Outcome.OutcomeType.TIMEOUT, new_state)
    if distance(new_state.hunter_position, new_state.prey_position) <= 4:
        points_to_check = points_between(new_state.hunter_position, new_state.prey_position)
        # If a wall is between hunter and prey then the game continues: the prey is not caught yet
        is_there_a_wall = any([any([wall_collides_point(wall, point) for wall in new_state.walls]) for point in points_to_check])
        if is_there_a_wall:
            return Outcome(Outcome.OutcomeType.CONTINUES, new_state)
        else:
            return Outcome(Outcome.OutcomeType.PREY_IS_CAUGHT, new_state)
    else:
        return Outcome(Outcome.OutcomeType.CONTINUES, new_state)
