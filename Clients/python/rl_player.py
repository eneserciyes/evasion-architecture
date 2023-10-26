import sys
from client import EvasionClient, GameState, Wall, MAX_HEIGHT, MAX_WIDTH, Velocity as ClientVelocity
from evasion_rl.Game import State, Config
from evasion_rl.Field import Vertical, Horizontal, Point, Velocity
from evasion_rl.EvasionEnv import EvasionEnv, get_observation_from_game_state
from stable_baselines3 import PPO

MODEL_DIR = "Clients/python/evasion_rl/model.zip"

prey_table = {
    (1,1): (-1,1),
    (1,0): (-1,1),
    (1,-1): (-1, 1),
    (0,1): (0, -1),
    (0, -1): (0, 1),
    (-1, 1): (1,0),
    (-1, 0): (1,1),
    (-1, -1): (1,-1)
}

def convert_game_state_to_state(game: GameState, next_wall_time:int) -> State:
    walls = []
    for wall in game.walls:
        if wall.x1 == wall.x2:
            walls.append(Vertical(wall.x1, wall.y1, wall.y2))
        else:
            walls.append(Horizontal(wall.y1, wall.x1, wall.x2))
    return State(
        config=Config(max_walls=5, next_wall_interval=next_wall_time),
        ticker=game.ticker,
        hunter_position=Point(game.hunter_position.x, game.hunter_position.y),
        hunter_velocity=Velocity(game.hunter_velocity.x, game.hunter_velocity.y),
        hunter_last_wall=game.hunter_last_wall_time,
        prey_position=Point(game.prey_position.x, game.prey_position.y),
        prey_velocity=Velocity(game.prey_velocity.x, game.prey_velocity.y),
        walls=walls
    )

class RLPlayer(EvasionClient):
    def __init__(self, port=4000):
        # TODO: Change this to your team name!
        self.team_name = "Enes"

        super().__init__(self.team_name, port)
        self.env = EvasionEnv(max_walls=5, next_wall_interval=self.config.next_wall_time)
        self.env.reset()
        self.model = PPO.load(MODEL_DIR, env=self.env, device="cpu")
    
    def calculate_hunter_move(self, game: GameState) -> str:
        state = convert_game_state_to_state(game, self.config.next_wall_time)
        obs = get_observation_from_game_state(state)
        action, _states = self.model.predict(obs, deterministic=True)
        if action == 0:
            return self.move_no_op()
        elif action == 1:
            hunter_x, hunter_y = state.hunter_position.x, state.hunter_position.y
            # find the closest vertical wall to the right of hunter
            vertical_walls_right = [wall.x for wall in state.walls if isinstance(wall, Vertical) and wall.x > hunter_x]
            wall_x2 = min(vertical_walls_right)-1 if len(vertical_walls_right) > 0 else MAX_WIDTH-1 # -1 will come
            # find the closest vertical wall to the left of hunter
            vertical_walls_left = [wall.x for wall in state.walls if isinstance(wall, Vertical) and wall.x < hunter_x]
            wall_x1 = max(vertical_walls_left)+1 if len(vertical_walls_left) > 0 else 0 # +1 will come
            return self.move_create_wall(Wall(wall_x1, hunter_y, wall_x2, hunter_y))
        elif action == 2:
            hunter_x, hunter_y = state.hunter_position.x, state.hunter_position.y
            # find the closest vertical wall to the right of hunter
            horizontal_walls_above = [wall.y for wall in state.walls if isinstance(wall, Horizontal) and wall.y > hunter_y]
            wall_y2 = min(horizontal_walls_above)-1 if len(horizontal_walls_above) > 0 else MAX_HEIGHT-1
            # find the closest horizontal wall below hunter
            horizontal_walls_below = [wall.y for wall in state.walls if isinstance(wall, Horizontal) and wall.y < hunter_y]
            wall_y1 = max(horizontal_walls_below)+1 if len(horizontal_walls_below) > 0 else 0
            return self.move_create_wall(Wall(hunter_x, wall_y1, hunter_x, wall_y2))
        else:
            if len(state.walls) < (action-2): return self.move_no_op()
            wall_to_remove = state.walls[action-3]
            if isinstance(wall_to_remove, Vertical):
                wall_to_remove = Wall(wall_to_remove.x, wall_to_remove.y1, wall_to_remove.x, wall_to_remove.y2)
            else:
                wall_to_remove = Wall(wall_to_remove.x1, wall_to_remove.y, wall_to_remove.x2, wall_to_remove.y)
            return self.move_only_remove_walls([wall_to_remove])

    def calculate_prey_move(self, game: GameState) -> str:
        hunter_vel = (game.hunter_velocity.x, game.hunter_velocity.y)
        prey_vel = prey_table[hunter_vel]
        return self.move_change_velocity(ClientVelocity(prey_vel[0], prey_vel[1]))


# -------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) == 1: port = 4000
    else: port = int(sys.argv[1])
    client = RLPlayer(port)
    client.run()
