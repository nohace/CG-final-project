import random
import math
import tkinter as tk
from tkinter import ttk
import time
import numpy as np
from collections import deque

##############################################################################
# LEFT-SIDE: GRID DRAWING (REFERENCE CODE UNCHANGED)
##############################################################################

def map_coordinates(x, y, canvas_width, canvas_height, graph_range):
    """Map (-graph_range..+graph_range) -> (0..square_size) for x,y."""
    square_size = min(canvas_width, canvas_height)
    scaled_x = ((x + graph_range) / (2 * graph_range)) * square_size
    scaled_y = ((-y + graph_range) / (2 * graph_range)) * square_size
    return scaled_x, scaled_y

def draw_graph(canvas, left_width, canvas_height, graph_range=1000):
    """Draw the 2D grid on the left half of the canvas only."""
    canvas.delete("all")

    square_size = min(left_width, canvas_height)
    offset_x = (left_width - square_size) // 2
    offset_y = (canvas_height - square_size) // 2

    for i in range(-graph_range, graph_range + 1, 200):
        x1, y1 = map_coordinates(i, -graph_range, square_size, square_size, graph_range)
        x2, y2 = map_coordinates(i, graph_range,  square_size, square_size, graph_range)
        y3, y4 = map_coordinates(-graph_range, i, square_size, square_size, graph_range)
        y5, y6 = map_coordinates(graph_range, i,  square_size, square_size, graph_range)

        # Draw vertical lines
        canvas.create_line(offset_x + x1, offset_y + y1,
                           offset_x + x2, offset_y + y2,
                           fill="#e0e0e0")
        # Draw horizontal lines
        canvas.create_line(offset_x + y3, offset_y + y4,
                           offset_x + y5, offset_y + y6,
                           fill="#e0e0e0")

        if i != 0:
            # X-axis label
            xx, _ = map_coordinates(i, 0, square_size, square_size, graph_range)
            canvas.create_text(offset_x + xx,
                               offset_y + square_size//2 + 15,
                               text=str(i), font=("Arial", 8))

            # Y-axis label
            _, yy = map_coordinates(0, i, square_size, square_size, graph_range)
            canvas.create_text(offset_x + square_size//2 - 15,
                               offset_y + yy,
                               text=str(i), font=("Arial", 8))

    # Axes
    center_x = offset_x + square_size // 2
    center_y = offset_y + square_size // 2
    canvas.create_line(offset_x, center_y,
                       offset_x + square_size, center_y,
                       fill="black", width=2)
    canvas.create_line(center_x, offset_y,
                       center_x, offset_y + square_size,
                       fill="black", width=2)

def display_points(canvas, main_point, other_points, graph_range,
                   left_width, canvas_height):
    """
    Draw all points on the left side. Return:
       point_ids: list of IDs for 'other_points'
       pt_id_map: dict from (x,y) -> canvas item ID (for highlighting)
    """
    square_size = min(left_width, canvas_height)
    offset_x = (left_width - square_size) // 2
    offset_y = (canvas_height - square_size) // 2

    point_ids = []
    pt_id_map = {}
    for x, y in other_points:
        px, py = map_coordinates(x, y, square_size, square_size, graph_range)
        pt_id = canvas.create_oval(offset_x + px - 3, offset_y + py - 3,
                                   offset_x + px + 3, offset_y + py + 3,
                                   fill="black")
        point_ids.append(pt_id)
        pt_id_map[(x, y)] = pt_id  # store for quick highlight

    # main point in red
    px, py = map_coordinates(main_point[0], main_point[1], square_size, square_size, graph_range)
    canvas.create_oval(offset_x + px - 6, offset_y + py - 6,
                       offset_x + px + 6, offset_y + py + 6,
                       fill="red")

    return point_ids, pt_id_map

def reset_canvas(canvas, points, graph_range, left_width, canvas_height):
    """Clear and redraw left-side grid and points."""
    if not points["main_point"] or not points["other_points"]:
        return
    draw_graph(canvas, left_width, canvas_height, graph_range)
    new_ids, new_map = display_points(canvas,
                                      points["main_point"],
                                      points["other_points"],
                                      graph_range,
                                      left_width,
                                      canvas_height)
    points["point_ids"] = new_ids
    points["pt_id_map"] = new_map  # update dict

##############################################################################
# BRUTE FORCE METHOD (REFERENCE CODE UNCHANGED)
##############################################################################

def brute_force_method(canvas, points, footer_label, graph_range=1000, animation_speed=1.0):
    if not points["main_point"] or not points["other_points"]:
        footer_label.config(text="Please generate points first.")
        return

    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    left_width = canvas_width // 2

    reset_canvas(canvas, points, graph_range, left_width, canvas_height)

    main_point = points["main_point"]
    other_points = points["other_points"]
    point_ids = points["point_ids"]

    square_size = min(left_width, canvas_height)
    offset_x = (left_width - square_size)//2
    offset_y = (canvas_height - square_size)//2

    def map_point(pt):
        return map_coordinates(pt[0], pt[1], square_size, square_size, graph_range)

    def circle_point(pt, color="red", size=6, outline_size=2):
        px, py = map_point(pt)
        canvas.create_oval(
            offset_x + px - size - outline_size,
            offset_y + py - size - outline_size,
            offset_x + px + size + outline_size,
            offset_y + py + size + outline_size,
            outline=color, width=outline_size
        )

    start_time = time.time()
    closest = None
    min_dist = float('inf')
    for p in other_points:
        d = math.dist(main_point, p)
        if d < min_dist:
            min_dist = d
            closest = p
    internal_time = time.time() - start_time

    footer_label.config(text="Processing...")

    sorted_pts_ids = sorted(zip(other_points, point_ids),
                            key=lambda item: (item[0][1], item[0][0]))
    sorted_points, sorted_ids = zip(*sorted_pts_ids)

    def animate(i):
        if i >= len(sorted_points):
            # done
            px1, py1 = map_point(main_point)
            px2, py2 = map_point(closest)
            canvas.create_line(offset_x + px1, offset_y + py1,
                               offset_x + px2, offset_y + py2,
                               dash=(3,3), fill="red", width=2)
            # Circle final point in red
            circle_point(closest, color="red", size=3)

            cx, cy = int(closest[0]), int(closest[1])
            footer_label.config(
                text=f"Closest Point: ({cx}, {cy}) | Distance: {round(min_dist,4)} "
                     f"| Internal Time: {round(internal_time,4)}s"
            )
            return

        cur_pt, cur_id = sorted_points[i], sorted_ids[i]
        px1, py1 = map_point(main_point)
        px2, py2 = map_point(cur_pt)
        line_id = canvas.create_line(offset_x + px1, offset_y + py1,
                                     offset_x + px2, offset_y + py2,
                                     fill="gray", dash=(2,2), width=1)

        delay = int(200 / animation_speed)

        def next_step():
            # turn that point grey
            canvas.itemconfig(cur_id, fill="#808080")
            canvas.delete(line_id)
            animate(i+1)

        canvas.after(delay, next_step)

    animate(0)

##############################################################################
# KD-TREE (REFERENCE CODE UNCHANGED)
##############################################################################

def build_kdtree(points_arr, depth=0, parent=None, build_sequence=None):
    if build_sequence is None:
        build_sequence = []

    if len(points_arr) == 0:
        return None, build_sequence

    k = 2
    axis = depth % k
    pts_sorted = points_arr[points_arr[:, axis].argsort()]
    median = len(pts_sorted)//2
    node_pt = pts_sorted[median]

    node = {
        'point': node_pt,
        'axis': axis,
        'parent': parent,
        'left': None,
        'right': None,
        'tree_x': 0,
        'tree_y': 0,
        'canvas_id': None
    }

    build_sequence.append(node)

    left_child, build_sequence = build_kdtree(pts_sorted[:median], depth+1, node, build_sequence)
    right_child, build_sequence = build_kdtree(pts_sorted[median+1:], depth+1, node, build_sequence)
    node['left'] = left_child
    node['right'] = right_child

    return node, build_sequence

def compute_levels(root):
    from collections import deque
    if not root:
        return []
    levels = []
    queue = deque([(root, 0)])
    while queue:
        node, d = queue.popleft()
        if node is None:
            continue
        if len(levels) <= d:
            levels.append([])
        levels[d].append(node)
        queue.append((node['left'], d+1))
        queue.append((node['right'], d+1))
    return levels

def assign_positions(root, x_start, width, total_height, top_margin=50):
    if root is None:
        return
    lvls = compute_levels(root)
    num_levels = len(lvls)
    if num_levels <= 1:
        vertical_gap = (total_height - 2*top_margin)
    else:
        vertical_gap = (total_height - 2*top_margin)/ (num_levels - 1)

    for depth_idx, nodes_at_depth in enumerate(lvls):
        count_lvl = len(nodes_at_depth)
        if count_lvl == 0:
            continue
        horizontal_gap = width/(count_lvl+1)
        y_coord = top_margin + depth_idx*vertical_gap
        for i, node in enumerate(nodes_at_depth):
            node['tree_x'] = x_start + (i+1)*horizontal_gap
            node['tree_y'] = y_coord

def kd_nearest_steps(root, target, best=None, steps=None):
    if steps is None:
        steps = []
    if root is None:
        return best, steps

    def dist(a,b):
        return math.dist(a,b)
    node_pt = root['point']
    axis = root['axis']

    if best is None:
        best = (node_pt, dist(node_pt, target))

    d_here = dist(node_pt, target)
    if d_here < best[1]:
        best = (node_pt, d_here)

    steps.append(("visit", root, best[0]))

    diff = target[axis] - node_pt[axis]
    if diff <= 0:
        best, steps = kd_nearest_steps(root['left'], target, best, steps)
        if abs(diff) < best[1]:
            best, steps = kd_nearest_steps(root['right'], target, best, steps)
    else:
        best, steps = kd_nearest_steps(root['right'], target, best, steps)
        if abs(diff) < best[1]:
            best, steps = kd_nearest_steps(root['left'], target, best, steps)
    return best, steps

def draw_kd_node(canvas, node):
    r = 15
    x, y = node['tree_x'], node['tree_y']
    color = "lightblue" if node['axis'] == 0 else "lightgreen"
    node_id = canvas.create_oval(x-r, y-r, x+r, y+r, fill=color)
    node_label = canvas.create_text(x, y, text=f"{int(node['point'][0])}, {int(node['point'][1])}",
                                    font=("Arial", 8, "bold"))
    node['canvas_id'] = node_id

def animate_tree_build_sequence(canvas, build_sequence, pt_id_map,
                                animation_speed, on_finish=None):
    step_delay = int(300 / animation_speed)

    def left_canvas_coords(x, y):
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        left_w = cw // 2
        sq = min(left_w, ch)
        offx = (left_w - sq)//2
        offy = (ch - sq)//2
        px, py = map_coordinates(x, y, sq, sq, 1000)
        return (offx + px, offy + py)

    def do_step(i):
        if i >= len(build_sequence):
            if on_finish:
                on_finish()
            return

        node = build_sequence[i]
        xval, yval = float(node['point'][0]), float(node['point'][1])

        # highlight circle in blue on the left
        circle_id = None
        pid = pt_id_map.get((xval, yval))
        if pid is not None:
            sx, sy = left_canvas_coords(xval, yval)
            r = 10
            circle_id = canvas.create_oval(sx - r, sy - r,
                                           sx + r, sy + r,
                                           outline="blue", width=2)

        # draw node on the right
        draw_kd_node(canvas, node)
        if node['parent']:
            pxp, pyp = node['parent']['tree_x'], node['parent']['tree_y']
            canvas.create_line(pxp, pyp, node['tree_x'], node['tree_y'],
                               fill="gray", width=2)

        def revert():
            if circle_id is not None:
                canvas.delete(circle_id)
            if pid is not None:
                canvas.itemconfig(pid, fill="#808080")
            do_step(i+1)

        canvas.after(step_delay, revert)

    do_step(0)

def animate_kd_search(canvas, root, main_point, steps,
                      left_width, canvas_height,
                      graph_range, speed=5.0,
                      on_finish=None):
    square_size = min(left_width, canvas_height)
    offset_x = (left_width - square_size)//2
    offset_y = (canvas_height - square_size)//2

    def map_pt(p):
        return map_coordinates(p[0], p[1], square_size, square_size, graph_range)

    step_delay = int(800 / speed)  # slower

    def do_search_step(i):
        if i >= len(steps):
            if on_finish:
                on_finish()
            return

        _, node, _ = steps[i]
        # color node red
        if node.get('canvas_id') is not None:
            canvas.itemconfig(node['canvas_id'], fill="red", outline="black")

        px1, py1 = map_pt(main_point)
        px2, py2 = map_pt(node['point'])
        line_id = canvas.create_line(offset_x + px1, offset_y + py1,
                                     offset_x + px2, offset_y + py2,
                                     fill="gray", dash=(2,2), width=1)

        def after_delay():
            canvas.delete(line_id)
            do_search_step(i+1)

        canvas.after(step_delay, after_delay)

    do_search_step(0)

def find_kd_node(root, point):
    if root is None:
        return None
    if np.allclose(root['point'], point):
        return root
    left_res = find_kd_node(root['left'], point)
    if left_res is not None:
        return left_res
    right_res = find_kd_node(root['right'], point)
    return right_res

def circle_point_on_left(canvas, pt, left_width, canvas_height,
                         color="red", size=3, outline_size=2, graph_range=1000):
    square_size = min(left_width, canvas_height)
    offset_x = (left_width - square_size)//2
    offset_y = (canvas_height - square_size)//2
    px, py = map_coordinates(pt[0], pt[1], square_size, square_size, graph_range)
    canvas.create_oval(
        offset_x + px - size - outline_size,
        offset_y + py - size - outline_size,
        offset_x + px + size + outline_size,
        offset_y + py + size + outline_size,
        outline=color, width=outline_size
    )

def divide_and_conquer_method(canvas, points, footer_label,
                              graph_range=1000, animation_speed=1.0):
    if not points["main_point"] or not points["other_points"]:
        footer_label.config(text="Please generate points first.")
        return

    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    left_width = canvas_width // 2
    reset_canvas(canvas, points, graph_range, left_width, canvas_height)

    main_pt = np.array(points["main_point"], dtype=float)
    other_pts = np.array(points["other_points"], dtype=float)
    pt_id_map = points["pt_id_map"]

    start_build = time.time()
    kdroot, build_sequence = build_kdtree(other_pts, depth=0, parent=None)
    build_time = time.time() - start_build

    right_width = canvas_width - left_width
    assign_positions(kdroot, x_start=left_width, width=right_width,
                     total_height=canvas_height, top_margin=50)

    def after_build():
        search_start = time.time()
        (best_pt, best_dist), search_steps = kd_nearest_steps(kdroot, main_pt)
        search_time = time.time() - search_start
        total_time = build_time + search_time

        def after_search():
            final_node = find_kd_node(kdroot, best_pt)
            if final_node is not None:
                canvas.itemconfig(final_node['canvas_id'], fill="darkred", outline="black", width=2)

            px_final, py_final = best_pt
            circle_point_on_left(canvas, (px_final, py_final),
                                 left_width, canvas_height,
                                 color="red", size=3, graph_range=graph_range)

            square_size = min(left_width, canvas_height)
            offset_x = (left_width - square_size)//2
            offset_y = (canvas_height - square_size)//2
            px1, py1 = map_coordinates(main_pt[0], main_pt[1],
                                       square_size, square_size, graph_range)
            px2, py2 = map_coordinates(px_final, py_final,
                                       square_size, square_size, graph_range)
            canvas.create_line(offset_x + px1, offset_y + py1,
                               offset_x + px2, offset_y + py2,
                               fill="red", dash=(3,3), width=2)

            bx, by = int(best_pt[0]), int(best_pt[1])
            footer_label.config(text=f"Closest Point: ({bx}, {by}) | "
                                     f"Distance: {round(best_dist,4)} | "
                                     f"Time: {round(total_time,4)}s")

        animate_kd_search(canvas, kdroot, main_pt, search_steps,
                          left_width=left_width,
                          canvas_height=canvas_height,
                          graph_range=graph_range,
                          speed=animation_speed,
                          on_finish=after_search)

    animate_tree_build_sequence(canvas, build_sequence, pt_id_map,
                                animation_speed, on_finish=after_build)

##############################################################################
# RABIN'S ALGORITHM (VANTAGE-STYLE BOUNDING SPHERE)
##############################################################################

def build_rabin_tree(points_arr, depth=0, parent=None, build_seq=None):
    """
    Build a 'Rabin' vantage tree:
      1) Pick random pivot from points_arr.
      2) Compute bounding sphere radius = max distance to pivot in this subtree.
      3) Partition points into 'inside' vs 'outside' (like vantage point approach).
    Return (root, build_seq).
    """
    if build_seq is None:
        build_seq = []
    if len(points_arr) == 0:
        return None, build_seq

    # random pivot
    pivot_idx = random.randint(0, len(points_arr) - 1)
    pivot_pt = points_arr[pivot_idx]
    rest = np.delete(points_arr, pivot_idx, axis=0)

    node = {
        'point': pivot_pt,
        'axis': depth % 2,  # for coloring (like KD)
        'parent': parent,
        'left': None,
        'right': None,
        'tree_x': 0,
        'tree_y': 0,
        'canvas_id': None,
        'radius': 0.0
    }
    build_seq.append(node)

    if len(rest) == 0:
        return node, build_seq

    def dist(a,b):
        return math.dist(a,b)
    # bounding radius = max dist from pivot to rest
    dists = [dist(pivot_pt, p) for p in rest]
    node['radius'] = max(dists) if dists else 0.0

    # We'll define threshold as half the radius
    threshold = node['radius']/2.0
    inside = []
    outside = []
    for p in rest:
        if dist(p, pivot_pt) <= threshold:
            inside.append(p)
        else:
            outside.append(p)

    left_arr = np.array(inside)
    right_arr = np.array(outside)

    left_node, build_seq = build_rabin_tree(left_arr, depth+1, node, build_seq)
    right_node, build_seq = build_rabin_tree(right_arr, depth+1, node, build_seq)
    node['left'] = left_node
    node['right'] = right_node

    return node, build_seq

def rabin_nearest_steps(root, target, best=None, steps=None):
    """
    Vantage search:
      1) Visit pivot, update best if needed.
      2) Decide inside vs. outside child based on pivot_dist vs radius/2
      3) Descend that child
      4) Only descend the other child if bounding sphere might contain a closer point
    """
    if steps is None:
        steps = []
    if root is None:
        return best, steps

    def dist(a,b):
        return math.dist(a,b)

    node_pt = root['point']
    pivot_dist = dist(node_pt, target)
    if best is None:
        best = (node_pt, pivot_dist)

    # record step
    steps.append(("visit", root, best[0]))

    if pivot_dist < best[1]:
        best = (node_pt, pivot_dist)

    # define threshold
    threshold = root['radius']/2.0

    # decide inside vs outside
    inside_side = root['left']
    outside_side = root['right']

    # if pivot_dist <= threshold => target is likely inside
    if pivot_dist <= threshold:
        # search inside first
        best, steps = rabin_nearest_steps(inside_side, target, best, steps)
        # bounding check for outside
        # compute child pivot dist
        if outside_side is not None:
            child_pivot = outside_side['point']
            child_dist = dist(child_pivot, target)
            child_radius = outside_side['radius']
            # if child_pivot_dist - child_radius < best[1], search
            if child_dist - child_radius < best[1]:
                best, steps = rabin_nearest_steps(outside_side, target, best, steps)
    else:
        # search outside first
        best, steps = rabin_nearest_steps(outside_side, target, best, steps)
        # bounding check for inside
        if inside_side is not None:
            child_pivot = inside_side['point']
            child_dist = dist(child_pivot, target)
            child_radius = inside_side['radius']
            if child_dist - child_radius < best[1]:
                best, steps = rabin_nearest_steps(inside_side, target, best, steps)

    return best, steps

def compute_levels_rabin(root):
    """Same BFS approach to get levels for vantage tree."""
    if root is None:
        return []
    lvls = []
    queue = deque([(root, 0)])
    while queue:
        nd, depth = queue.popleft()
        if nd is None:
            continue
        if len(lvls) <= depth:
            lvls.append([])
        lvls[depth].append(nd)
        queue.append((nd['left'], depth+1))
        queue.append((nd['right'], depth+1))
    return lvls

def assign_positions_rabin(root, x_start, width, total_height, top_margin=50):
    """
    BFS to assign x,y so the vantage tree won't go offscreen.
    """
    lvls = compute_levels_rabin(root)
    num_levels = len(lvls)
    if num_levels <= 1:
        vertical_gap = (total_height - 2*top_margin)
    else:
        vertical_gap = (total_height - 2*top_margin)/(num_levels-1)

    for d_idx, nodes_at_depth in enumerate(lvls):
        c_lvl = len(nodes_at_depth)
        if c_lvl == 0:
            continue
        horiz_gap = width/(c_lvl+1)
        y_coord = top_margin + d_idx*vertical_gap
        for i, node in enumerate(nodes_at_depth):
            node['tree_x'] = x_start + (i+1)*horiz_gap
            node['tree_y'] = y_coord

def draw_rabin_node(canvas, node):
    """Draw vantage node (like KD node, different color? We'll do 'plum'."""
    r = 15
    x, y = node['tree_x'], node['tree_y']
    color = "plum"
    node_id = canvas.create_oval(x-r, y-r, x+r, y+r, fill=color)
    label = f"{int(node['point'][0])}, {int(node['point'][1])}"
    canvas.create_text(x, y, text=label, font=("Arial", 8, "bold"))
    node['canvas_id'] = node_id

def animate_rabin_build(canvas, build_seq, pt_id_map, animation_speed, on_finish=None):
    """
    Step by step build. Exactly like animate_tree_build_sequence but for vantage tree.
    """
    step_delay = int(300 / animation_speed)

    def left_canvas_coords(x, y):
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        left_w = cw // 2
        sq = min(left_w, ch)
        offx = (left_w - sq)//2
        offy = (ch - sq)//2
        px, py = map_coordinates(x, y, sq, sq, 1000)
        return (offx + px, offy + py)

    def do_step(i):
        if i >= len(build_seq):
            if on_finish:
                on_finish()
            return

        node = build_seq[i]
        px, py = node['point']
        circle_id = None
        pid = pt_id_map.get((float(px), float(py)))

        # highlight pivot in blue on the left
        if pid is not None:
            sx, sy = left_canvas_coords(px, py)
            r = 10
            circle_id = canvas.create_oval(sx-r, sy-r,
                                           sx+r, sy+r,
                                           outline="blue", width=2)

        # draw vantage node
        draw_rabin_node(canvas, node)
        if node['parent']:
            pxp, pyp = node['parent']['tree_x'], node['parent']['tree_y']
            canvas.create_line(pxp, pyp, node['tree_x'], node['tree_y'],
                               fill="gray", width=2)

        def revert():
            if circle_id is not None:
                canvas.delete(circle_id)
            if pid is not None:
                canvas.itemconfig(pid, fill="#808080")
            do_step(i+1)

        canvas.after(step_delay, revert)

    do_step(0)

def animate_rabin_search(canvas, root, main_pt, steps,
                         left_width, canvas_height,
                         graph_range, speed=5.0,
                         on_finish=None):
    """
    Exactly like animate_kd_search, but for vantage approach. We fill nodes in red,
    draw dashed lines from main pt, not reverting color.
    """
    sq_size = min(left_width, canvas_height)
    offx = (left_width - sq_size)//2
    offy = (canvas_height - sq_size)//2

    def map_pt(p):
        return map_coordinates(p[0], p[1], sq_size, sq_size, graph_range)

    step_delay = int(800 / speed)

    def do_step(i):
        if i >= len(steps):
            if on_finish:
                on_finish()
            return

        _, node, bestp = steps[i]
        if node.get('canvas_id'):
            # fill vantage node in red
            canvas.itemconfig(node['canvas_id'], fill="red", outline="black")

        px1, py1 = map_pt(main_pt)
        px2, py2 = map_pt(node['point'])
        line_id = canvas.create_line(offx + px1, offy + py1,
                                     offx + px2, offy + py2,
                                     fill="gray", dash=(2,2), width=1)

        def after_delay():
            canvas.delete(line_id)
            do_step(i+1)

        canvas.after(step_delay, after_delay)

    do_step(0)

def find_rabin_node(root, point):
    """Locate vantage node with 'point' in vantage tree."""
    if root is None:
        return None
    if np.allclose(root['point'], point):
        return root
    left_res = find_rabin_node(root['left'], point)
    if left_res is not None:
        return left_res
    right_res = find_rabin_node(root['right'], point)
    return right_res

def rabin_method(canvas, points, footer_label,
                 graph_range=1000, animation_speed=1.0):
    """
    vantage-based 'rabin' approach:
      1) build_rabin_tree => vantage approach with bounding sphere
      2) animate build
      3) vantage search => skip subtrees that can't have better point
      4) animate search
    """
    if not points["main_point"] or not points["other_points"]:
        footer_label.config(text="Please generate points first.")
        return

    # reset
    cw = canvas.winfo_width()
    ch = canvas.winfo_height()
    left_w = cw // 2
    reset_canvas(canvas, points, graph_range, left_w, ch)

    main_pt = np.array(points["main_point"], dtype=float)
    other_pts = np.array(points["other_points"], dtype=float)
    pt_id_map = points["pt_id_map"]

    # build vantage tree
    start_build = time.time()
    rabinroot, build_seq = build_rabin_tree(other_pts, depth=0, parent=None)
    build_time = time.time() - start_build

    # assign positions vantage style
    right_w = cw - left_w
    assign_positions_rabin(rabinroot, x_start=left_w, width=right_w,
                           total_height=ch, top_margin=50)

    def after_build():
        # vantage search
        stime = time.time()
        (best_pt, best_dist), search_steps = rabin_nearest_steps(rabinroot, main_pt)
        search_time = time.time() - stime
        total_time = build_time + search_time

        def after_search():
            # final node color
            final_node = find_rabin_node(rabinroot, best_pt)
            if final_node and final_node.get('canvas_id'):
                canvas.itemconfig(final_node['canvas_id'], fill="darkred", outline="black", width=2)

            # circle final on left
            px_f, py_f = best_pt
            circle_point_on_left(canvas, (px_f, py_f),
                                 left_w, ch,
                                 color="red", size=3, graph_range=graph_range)

            # dashed line
            sq_size = min(left_w, ch)
            offx = (left_w - sq_size)//2
            offy = (ch - sq_size)//2
            px1, py1 = map_coordinates(main_pt[0], main_pt[1], sq_size, sq_size, graph_range)
            px2, py2 = map_coordinates(px_f, py_f, sq_size, sq_size, graph_range)
            canvas.create_line(offx + px1, offy + py1,
                               offx + px2, offy + py2,
                               fill="red", dash=(3,3), width=2)

            bx, by = int(best_pt[0]), int(best_pt[1])
            footer_label.config(
                text=f"Rabin's Closest: ({bx}, {by}) | Dist: {round(best_dist,4)} | Time: {round(total_time,4)}s"
            )

        # animate vantage search
        animate_rabin_search(canvas, rabinroot, main_pt, search_steps,
                             left_width=left_w,
                             canvas_height=ch,
                             graph_range=graph_range,
                             speed=animation_speed,
                             on_finish=after_search)

    # animate vantage build
    animate_rabin_build(canvas, build_seq, pt_id_map, animation_speed, on_finish=after_build)

##############################################################################
# POINT GENERATION + REFERENCE CREATE_GUI
##############################################################################

def generate_points(points, canvas, footer_label, main_x, main_y,
                    num_points=20, graph_range=1000, clustered=False):
    main_point = (main_x, main_y)

    if clustered:
        num_clusters = random.randint(1, 5)
        cluster_centers = [
            (random.randint(-graph_range, graph_range),
             random.randint(-graph_range, graph_range))
            for _ in range(num_clusters)
        ]
        other_pts = []
        cluster_size = max(num_points // num_clusters, 1)
        for cx, cy in cluster_centers:
            for _ in range(cluster_size):
                x = cx + random.randint(-graph_range//10, graph_range//10)
                y = cy + random.randint(-graph_range//10, graph_range//10)
                other_pts.append((x,y))
        other_pts = other_pts[:num_points]
    else:
        other_pts = [
            (random.randint(-graph_range, graph_range),
             random.randint(-graph_range, graph_range))
            for _ in range(num_points)
        ]

    points["main_point"] = main_point
    points["other_points"] = other_pts
    points["point_ids"] = []
    points["pt_id_map"] = {}

    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    left_width = canvas_width // 2

    draw_graph(canvas, left_width, canvas_height, graph_range)
    new_ids, new_map = display_points(canvas, main_point, other_pts,
                                      graph_range, left_width, canvas_height)
    points["point_ids"] = new_ids
    points["pt_id_map"] = new_map

    footer_label.config(text="Points generated. Ready to run algorithms.")

def create_gui():
    graph_range = 1000
    points = {
        "main_point": None,
        "other_points": None,
        "point_ids": [],
        "pt_id_map": {}
    }

    window = tk.Tk()
    window.title("Closest Point Finder w/ KD-Tree Visualization")
    window.geometry("1200x800")
    window.resizable(True, True)

    style = ttk.Style()
    style.configure("TFrame", background="#f5f5f5")
    style.configure("TLabel", background="#f5f5f5", font=("Arial", 12))
    style.configure("TButton", font=("Arial", 12, "bold"))
    style.configure("TEntry", font=("Arial", 12))

    input_frame = ttk.Frame(window, padding=10)
    input_frame.pack(fill="x", pady=10)

    ttk.Label(input_frame, text="Main X:").grid(row=0, column=0, padx=5, pady=5)
    main_x_entry = ttk.Entry(input_frame, width=10)
    main_x_entry.insert(0, "500")
    main_x_entry.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(input_frame, text="Main Y:").grid(row=0, column=2, padx=5, pady=5)
    main_y_entry = ttk.Entry(input_frame, width=10)
    main_y_entry.insert(0, "500")
    main_y_entry.grid(row=0, column=3, padx=5, pady=5)

    ttk.Label(input_frame, text="# Points:").grid(row=0, column=4, padx=5, pady=5)
    points_entry = ttk.Entry(input_frame, width=10)
    points_entry.insert(0, "20")
    points_entry.grid(row=0, column=5, padx=5, pady=5)

    ttk.Label(input_frame, text="Speed:").grid(row=0, column=6, padx=5, pady=5)
    speed_entry = ttk.Entry(input_frame, width=10)
    speed_entry.insert(0, "5")
    speed_entry.grid(row=0, column=7, padx=5, pady=5)

    cluster_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(input_frame, text="Clusters?", variable=cluster_var)\
       .grid(row=0, column=8, padx=5, pady=5)

    # Generate
    ttk.Button(input_frame, text="Generate Points",
               command=lambda: generate_points(
                   points, canvas, footer_label,
                   main_x=int(main_x_entry.get()),
                   main_y=int(main_y_entry.get()),
                   num_points=int(points_entry.get()),
                   graph_range=graph_range,
                   clustered=cluster_var.get()
               )).grid(row=0, column=9, padx=5, pady=5)

    # Brute Force
    ttk.Button(input_frame, text="Brute Force",
               command=lambda: brute_force_method(
                   canvas, points, footer_label,
                   graph_range=graph_range,
                   animation_speed=float(speed_entry.get())
               )).grid(row=1, column=4, columnspan=2, padx=5, pady=10)

    # KD-Tree
    ttk.Button(input_frame, text="Divide & Conquer (KD-Tree)",
               command=lambda: divide_and_conquer_method(
                   canvas, points, footer_label,
                   graph_range=graph_range,
                   animation_speed=float(speed_entry.get())
               )).grid(row=2, column=4, columnspan=2, padx=5, pady=10)

    ####### NEW RABIN ALGORITHM BUTTON #######
    rabin_button = ttk.Button(
        input_frame,
        text="Rabin's Algorithm",
        command=lambda: rabin_method(
            canvas, points, footer_label,
            graph_range=graph_range,
            animation_speed=float(speed_entry.get())
        )
    )
    rabin_button.grid(row=3, column=4, columnspan=2, padx=5, pady=10)
    ##########################################

    canvas_frame = ttk.Frame(window, padding=10)
    canvas_frame.pack(fill="both", expand=True)
    canvas = tk.Canvas(canvas_frame, bg="white")
    canvas.pack(fill="both", expand=True)

    footer_frame = ttk.Frame(window, padding=10)
    footer_frame.pack(fill="x")
    footer_label = ttk.Label(footer_frame, text="© 2024 KD-Tree Visualizer")
    footer_label.pack()

    window.mainloop()

if __name__ == "__main__":
    create_gui()
