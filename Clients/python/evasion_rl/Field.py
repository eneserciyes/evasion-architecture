from math import sqrt
from dataclasses import dataclass
from typing import List, Tuple, Union

@dataclass
class Point:
    x:int
    y:int

@dataclass    
class Horizontal:
    y:int
    x1:int
    x2:int

@dataclass
class Vertical:
    x: int
    y1: int
    y2: int

@dataclass
class Velocity:
    x: int
    y: int 

# --------------------------------
# CONSTANTS
# --------------------------------

# Maximum width of the field
MaxWidth:int = 300
MaxHeight:int = 300

# --------------------------------
# UTILITY FUNCTIONS
# --------------------------------

def signum(v: Velocity) -> Velocity:
    x = 1 if v.x > 0 else -1 if v.x < 0 else 0 
    y = 1 if v.y > 0 else -1 if v.y < 0 else 0
    return Velocity(x, y)

def distance(p1: Point, p2: Point) -> float:
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return sqrt(dx * dx + dy * dy)

def step_by(v: Velocity, p: Point) -> Point:
    return Point(p.x + v.x, p.y + v.y)

def collides_boundary(p: Point) -> bool:
    return p.x == -1 or p.x == MaxWidth or p.y == -1 or p.y == MaxHeight

def wall_collides_point(wall:Union[Horizontal, Vertical], p: Point) -> bool:
    if isinstance(wall, Horizontal):
        return p.y == wall.y and p.x >= wall.x1 and p.x <= wall.x2
    elif isinstance(wall, Vertical):
        return p.x == wall.x and p.y >= wall.y1 and p.y <= wall.y2
    else:
        raise Exception("Invalid wall type")

def walls_or_bounds_collide_point(walls, p):
    for wall in walls:
        if wall_collides_point(wall, p):
            return True
    return collides_boundary(p)

def step_and_bounce(v: Velocity, walls: List[Union[Horizontal, Vertical]], p: Point) -> Tuple[Point, Velocity]:
    projected = step_by(v, p)
    projected_collides = walls_or_bounds_collide_point(walls, projected)
    if projected_collides:
        if v.x == 0:
            return p, Velocity(v.x, -v.y)
        elif v.y == 0:
            return p, Velocity(-v.x, v.y)
        else:
            v_adjacent = Point(projected.x, projected.y - v.y)
            h_adjacent = Point(projected.x - v.x, projected.y)
            v_adj_collides = walls_or_bounds_collide_point(walls, v_adjacent)
            h_adj_collides = walls_or_bounds_collide_point(walls, h_adjacent)
            # If both adjacent points collide, then the projected point is a corner and the velocity is flipped in both directions with same position
            if v_adj_collides and h_adj_collides:
                return p, Velocity(-v.x, -v.y)
            # If just v collides, then the point moves vertically with flipped x velocity
            elif v_adj_collides:
                return Point(p.x, projected.y), Velocity(-v.x, v.y)
            # If just h collides, then the point moves horizontally with flipped y velocity
            elif h_adj_collides:
                return Point(projected.x, p.y), Velocity(v.x, -v.y)
            # If neither collides, then need to check the points beyond
            else:
                v_extended = Point(projected.x, projected.y + v.y)
                h_extended = Point(projected.x + v.x, projected.y)
                v_ext_collides = walls_or_bounds_collide_point(walls, v_extended)
                h_ext_collides = walls_or_bounds_collide_point(walls, h_extended)
                # If both extended points collide, then the projected point is a corner and the velocity is flipped in both directions with same position
                if v_ext_collides and h_ext_collides:
                    return p, Velocity(-v.x, -v.y)
                elif v_ext_collides:
                    return Point(p.x, projected.y), Velocity(-v.x, v.y)
                elif h_ext_collides:
                    return Point(projected.x, p.y), Velocity(v.x, -v.y)
                # If neither extended points collide, the wall is a single point and acts like a corner
                else:
                    return p, Velocity(-v.x, -v.y)
    else:
        # if the projected point doesn't collide with anything, it moves to that point with the same velocity
        return projected, v

def wall_collides_walls(newWall:Union[Horizontal, Vertical], walls: List[Union[Horizontal, Vertical]]):
    for wall in walls:
        if isinstance(wall, Horizontal) and isinstance(newWall, Horizontal):
            if wall.y == newWall.y and newWall.x1 >= wall.x1 and newWall.x1 <= wall.x2:
                return True
            elif wall.y == newWall.y and newWall.x2 >= wall.x1 and newWall.x2 <= wall.x2:
                return True
            elif wall.y == newWall.y and newWall.x1 <= wall.x1 and newWall.x2 >= wall.x2:
                return True
            else:
                continue
        elif isinstance(wall, Vertical) and isinstance(newWall, Vertical):
            if wall.x == newWall.x and newWall.y1 >= wall.y1 and newWall.y1 <= wall.y2:
                return True
            elif wall.x == newWall.x and newWall.y2 >= wall.y1 and newWall.y2 <= wall.y2:
                return True
            elif wall.x == newWall.x and newWall.y1 <= wall.y1 and newWall.y2 >= wall.y2:
                return True
            else:
                continue
        elif isinstance(wall, Vertical) and isinstance(newWall, Horizontal):
            if wall.x >= newWall.x1 and wall.x <= newWall.x2 and newWall.y >= wall.y1 and newWall.y <= wall.y2:
                return True
            else:
                continue
        elif isinstance(wall, Horizontal) and isinstance(newWall, Vertical):
            if wall.y >= newWall.y1 and wall.y <= newWall.y2 and newWall.x >= wall.x1 and newWall.x <= wall.x2:
                return True
            else:
                continue
    return False

def wall_in_bounds(wall: Union[Horizontal, Vertical]) -> bool:
    if isinstance(wall, Horizontal):
        return wall.y >= 0 and wall.y < MaxHeight and wall.x1 >= 0 and wall.x1 < MaxWidth and wall.x2 >= 0 and wall.x2 < MaxWidth
    elif isinstance(wall, Vertical):
        return wall.x >= 0 and wall.x < MaxWidth and wall.y1 >= 0 and wall.y1 < MaxHeight and wall.y2 >= 0 and wall.y2 < MaxHeight
    else:
        raise Exception("Invalid wall type")

def remove_wall(wall: Union[Horizontal, Vertical], walls: List[Union[Horizontal, Vertical]]) -> List[Union[Horizontal, Vertical]]:
    return list(filter(lambda w: w != wall, walls))

def remove_walls(wallsToRemove: List[Union[Horizontal, Vertical]], walls: List[Union[Horizontal, Vertical]]) -> List[Union[Horizontal, Vertical]]:
    return list(filter(lambda w: w not in wallsToRemove, walls))
