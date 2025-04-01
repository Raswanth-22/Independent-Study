import math
import numbers
import numpy as np
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon as ShPolygon
import shapely.geometry as geom

def numel(var):
    """
    Counts the number of entries in a numpy array, or returns 1 for fundamental numerical types.
    """
    if isinstance(var, (bool, numbers.Number, np.number, np.bool_)):
        return 1
    elif isinstance(var, np.ndarray):
        return var.size
    else:
        raise NotImplementedError(f'number of elements for type {type(var)}')

def rot2d(theta):
    """
    Create a 2-D rotation matrix from the angle theta.
    """
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta),  math.cos(theta)]])

class Edge:
    def __init__(self, vertices):
        """
        vertices: 2 x 2 array, each column is [x; y].
        """
        self.vertices = vertices

    @property
    def direction(self):
        return self.vertices[:, [1]] - self.vertices[:, [0]]

    @property
    def base(self):
        return self.vertices[:, [0]]

    def is_collision(self, edge):
        tol = 1e-6
        a_dirs = np.hstack([self.direction, -edge.direction])
        det_val = np.linalg.det(a_dirs)
        if abs(det_val) < tol:
            return False
        b_bases = np.hstack([edge.base - self.base])
        t_param = np.linalg.solve(a_dirs, b_bases)
        t_self, t_other = t_param[0, 0], t_param[1, 0]
        return (tol < t_self < 1.0 - tol) and (tol < t_other < 1.0 - tol)

def angle(vertex0, vertex1, vertex2, angle_type='signed'):
    tol = 2.22e-16
    vec1 = vertex1 - vertex0
    vec2 = vertex2 - vertex0
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 < tol or norm2 < tol:
        return math.nan
    vec1, vec2 = vec1 / norm1, vec2 / norm2
    vec1flat = np.vstack([vec1, 0]).flatten()
    vec2flat = np.vstack([vec2, 0]).flatten()
    c_angle = np.inner(vec1flat, vec2flat)
    s_angle = np.inner(np.array([0, 0, 1]), np.cross(vec1flat, vec2flat))
    edge_angle = math.atan2(s_angle, c_angle)
    if angle_type.lower() == 'unsigned':
        edge_angle = (edge_angle + 2 * math.pi) % (2 * math.pi)
    elif angle_type.lower() != 'signed':
        raise ValueError('Invalid argument angle_type')
    return edge_angle

class Polygon:
    def __init__(self, vertices, idx_vertices=None):
        self.vertices = vertices
        self.idx_vertices = idx_vertices

    @property
    def nb_vertices(self):
        return self.vertices.shape[1]

    def plot(self, style):
        if not style:
            style = 'k'
        directions = np.diff(self.vertices_loop)
        plt.quiver(self.vertices[0, :],
                   self.vertices[1, :],
                   directions[0, :],
                   directions[1, :],
                   color=style,
                   angles='xy',
                   scale_units='xy',
                   scale=1.)
        if self.idx_vertices is not None:
            for idx in range(len(self.idx_vertices)):
                plt.text(self.vertices[0, idx], self.vertices[1, idx],
                         str(self.idx_vertices[idx]), color=style)

    @property
    def vertices_loop(self):
        return np.hstack((self.vertices, self.vertices[:, [0]]))

class PolygonWorld:
    def __init__(self):
        self.world = []
        # Outer boundary
        outer_vertices = np.array([
            [0.0, 5.0, 5.0, 0.0],
            [0.0, 0.0, 3.0, 3.0]
        ])
        self.world.append(Polygon(outer_vertices))
        # Inner rectangle  
        inner_vertices = np.array([
            [1.0, 4.0, 4.0, 1.0],
            [0.0, 0.0, 1.3, 1.3]
        ])
        self.world.append(Polygon(inner_vertices))
        # Goal point
        self.x_goal = np.array([[4.5], [1.0]])

    def plot(self):
        # Plot goal
        ax = plt.gca()
        ax.plot(self.x_goal[0, 0], self.x_goal[1, 0], 'r*', markersize=10, label='Goal')
        ax.set_xlim([-1, 6])
        ax.set_ylim([-1, 4])
        plt.axis('equal')
        plt.legend()
        plt.title("Reactive Navigation in Free Space")

    def visibility_polygon(self, x_robot):
        visible_points = []
        num_rays = 360
        angles = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
        for angle_val in angles:
            direction = np.array([[math.cos(angle_val)], [math.sin(angle_val)]], dtype=np.float64)
            ray = np.hstack([x_robot, x_robot + 1000 * direction])
            closest_intersection = None
            min_distance = float('inf')
            for polygon in self.world:
                for i in range(polygon.nb_vertices):
                    edge = Edge(polygon.vertices[:, [i, (i + 1) % polygon.nb_vertices]])
                    if edge.is_collision(Edge(ray)):
                        a_dirs = np.hstack([ray[:, 1:2] - ray[:, :1],
                                            edge.vertices[:, :1] - edge.vertices[:, 1:2]])
                        b_bases = edge.vertices[:, :1] - ray[:, :1]
                        try:
                            t = np.linalg.solve(a_dirs, b_bases)
                            t_ray, t_edge = t[0, 0], t[1, 0]
                            if 0 <= t_ray <= 1 and 0 <= t_edge <= 1:
                                intersection = ray[:, :1] + t_ray * (ray[:, 1:2] - ray[:, :1])
                                distance = np.linalg.norm(intersection - x_robot)
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_intersection = intersection
                        except np.linalg.LinAlgError:
                            continue
            if closest_intersection is not None:
                visible_points.append(closest_intersection)
        if not visible_points:
            return None
        visibility_poly_vertices = np.hstack(visible_points)
        diffs = visibility_poly_vertices - x_robot
        angles_sort = np.arctan2(diffs[1, :], diffs[0, :])
        sort_idx = np.argsort(angles_sort)
        visibility_poly_vertices = visibility_poly_vertices[:, sort_idx]
        return Polygon(visibility_poly_vertices)

def polygon_to_shapely(poly):
    if poly is None:
        return None
    coords = poly.vertices.T
    return ShPolygon(coords)

def waypoint_find(polygon_intersection, x_goal, environment_polygons):
    if polygon_intersection is None:
        return None
    if isinstance(polygon_intersection, list) and len(polygon_intersection) == 0:
        return None
    if isinstance(polygon_intersection, Polygon):
        intersection_list = [polygon_intersection]
    else:
        intersection_list = polygon_intersection

    env_edges = []
    for poly in environment_polygons:
        for i in range(poly.nb_vertices):
            env_edge_vertices = poly.vertices[:, [i, (i+1) % poly.nb_vertices]]
            env_edges.append(Edge(env_edge_vertices))

    candidates = []
    for poly_int in intersection_list:
        nbv = poly_int.nb_vertices
        for i in range(nbv):
            int_edge_vertices = poly_int.vertices[:, [i, (i+1) % nbv]]
            int_edge = Edge(int_edge_vertices)
            is_env_edge = False
            for env_edge in env_edges:
                A1 = int_edge.vertices[:, 0]
                A2 = int_edge.vertices[:, 1]
                B1 = env_edge.vertices[:, 0]
                B2 = env_edge.vertices[:, 1]
                forward_match = (np.linalg.norm(A1 - B1) < 1e-5 and np.linalg.norm(A2 - B2) < 1e-5)
                reverse_match = (np.linalg.norm(A1 - B2) < 1e-5 and np.linalg.norm(A2 - B1) < 1e-5)
                if forward_match or reverse_match:
                    is_env_edge = True
                    break
            if not is_env_edge:
                p1 = int_edge.vertices[:, 0]
                p2 = int_edge.vertices[:, 1]
                seg_vec = p2 - p1
                pt_vec = x_goal.flatten() - p1
                seg_len_sqr = seg_vec.dot(seg_vec)
                if seg_len_sqr < 1e-12:
                    cpt = p1.reshape((2,1))
                else:
                    t = pt_vec.dot(seg_vec) / seg_len_sqr
                    t = max(0, min(1, t))
                    closest = p1 + t * seg_vec
                    cpt = closest.reshape((2,1))
                dist = np.linalg.norm(cpt - x_goal)
                candidates.append((dist, cpt))
    if len(candidates) == 0:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def plot_goal(x_goal):
    plt.plot(x_goal[0, 0], x_goal[1, 0], 'r*', markersize=10, label='Goal')

def safe_move(pos, desired_move, free_space, polygon_world):
    """
    Try to move from pos by desired_move. If the new position is in a region close to obstacles,
    try small angular deviations. If those fail, compute a waypoint based on the visibility polygon.
    """
    new_pos = pos + desired_move
    if free_space.contains(Point(new_pos[0,0], new_pos[1,0])):
        return new_pos

    for offset in np.linspace(np.deg2rad(5), np.deg2rad(90), 18):
        for sign in [1, -1]:
            rot_matrix = rot2d(sign * offset)
            alt_move = rot_matrix.dot(desired_move)
            alt_new_pos = pos + alt_move
            if free_space.contains(Point(alt_new_pos[0,0], alt_new_pos[1,0])):
                return alt_new_pos

    # Use waypoint approach if deviations fail
    robot_vis_poly = polygon_world.visibility_polygon(pos)
    if robot_vis_poly is not None:
        shapely_vis = polygon_to_shapely(robot_vis_poly)
        shapely_intersection = shapely_vis.intersection(free_space)
        if shapely_intersection.is_empty:
            intersection_poly = None
        elif shapely_intersection.geom_type == 'Polygon':
            intersection_poly = Polygon(np.array(shapely_intersection.exterior.coords).T)
        elif shapely_intersection.geom_type == 'MultiPolygon':
            intersection_poly = []
            for g in shapely_intersection.geoms:
                coords = np.array(g.exterior.coords).T
                intersection_poly.append(Polygon(coords))
        else:
            intersection_poly = None

        waypoint = waypoint_find(intersection_poly, polygon_world.x_goal, polygon_world.world)
        if waypoint is not None:
            print("Using waypoint:", waypoint.flatten())
            return waypoint

    # Remain in place
    print("Move blocked; staying put.")
    return pos

# Main Loop
if __name__ == '__main__':
    # Build the environment and compute free space.
    polygon_world = PolygonWorld()
    outer_poly = ShPolygon(polygon_world.world[0].vertices.T)
    # Create safe obstacles by buffering obstacles with a safety margin.
    safe_margin = 0.2
    interior_polys = [ShPolygon(obs.vertices.T).buffer(safe_margin) for obs in polygon_world.world[1:]]
    obstacles_union = unary_union(interior_polys)
    free_space = outer_poly.difference(obstacles_union)

    # Initialize robot positions.
    robot_positions = [
        np.array([[0.3], [1.0]]),   # Robot 1
        np.array([[0.5], [0.7]]),   # Robot 2
        np.array([[0.2], [0.7]])    # Robot 3
    ]
    robot_colors = ['g', 'b', 'c']
    arrived = [False, False, False]
    step_size = 0.5
    goal_tol = 0.1

    # Define the minimum gap between robots along the formation direction.
    gap = 0.1

    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-1, 6])
    ax.set_ylim([-1, 4])

    iteration = 0
    # Main loop: update the robots until all have reached the goal.
    while not all(arrived):
        # Compute formation unit direction d using the current positions centroid.
        positions_mat = np.hstack(robot_positions)
        centroid = np.mean(positions_mat, axis=1, keepdims=True)
        d_vec = polygon_world.x_goal - centroid
        d_norm = np.linalg.norm(d_vec)
        if d_norm == 0:
            d = np.array([[1.0], [0.0]])
        else:
            d = d_vec / d_norm

        # --- Update Order Based on Closeness to the Goal ---
        # Compute distance from each robot to the goal.
        distances = []
        for idx, pos in enumerate(robot_positions):
            if arrived[idx]:
                distances.append(np.inf)
            else:
                distances.append(np.linalg.norm(polygon_world.x_goal - pos))
        # Get indices sorted so that the robot closest to the goal moves first.
        sorted_indices = np.argsort(distances)

        # Update robots in sorted order.
        for idx in sorted_indices:
            if arrived[idx]:
                continue

            pos = robot_positions[idx]
            diff = polygon_world.x_goal - pos
            dist = np.linalg.norm(diff)
            if dist > step_size:
                desired_move = step_size * diff / dist
            else:
                desired_move = diff

            new_pos = pos + desired_move

            # Check free space; if not free, try alternative moves.
            if not free_space.contains(Point(new_pos[0,0], new_pos[1,0])):
                new_pos = safe_move(pos, desired_move, free_space, polygon_world)

            robot_positions[idx] = new_pos

            # If the robot reaches the goal, snap it in place.
            if np.linalg.norm(polygon_world.x_goal - new_pos) <= goal_tol:
                print(f"Robot {idx+1} has reached the goal.")
                robot_positions[idx] = polygon_world.x_goal.copy()
                arrived[idx] = True

            # Re-draw the scene after each robot update.
            ax.clear()
            for obstacle in polygon_world.world:
                obstacle.plot('r')
            plt.plot([1, 1], [0, 1], 'r-', linewidth=2)
            plt.plot([4, 4], [0, 1], 'r-', linewidth=2)
            for j, pos_j in enumerate(robot_positions):
                col = robot_colors[j]
                ax.plot(pos_j[0,0], pos_j[1,0], col+'o', markersize=8, label=f'Robot {j+1}')
                vis_poly = polygon_world.visibility_polygon(pos_j)
                if vis_poly is not None:
                    vis_poly.plot(col)
            plot_goal(polygon_world.x_goal)

            ax.set_aspect('equal', 'box')
            ax.set_xlim([-1, 6])
            ax.set_ylim([-1, 4])
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles))
            ax.legend(unique_labels.values(), unique_labels.keys())
            plt.pause(0.5)
            iteration += 1
            plt.savefig(f"simulation_output_{iteration:02d}.png")

            if all(arrived):
                break

    print("All robots have reached the goal.")
    plt.show()
