import random
import math
import tkinter as tk
from tkinter import ttk
import time
import numpy as np

def map_coordinates(x, y, canvas_width, canvas_height, graph_range):
    """Map graph coordinates (-graph_range to +graph_range) to canvas pixel coordinates."""
    square_size = min(canvas_width, canvas_height)  # Ensure square scaling
    scaled_x = ((x + graph_range) / (2 * graph_range)) * square_size
    scaled_y = ((-y + graph_range) / (2 * graph_range)) * square_size
    return scaled_x, scaled_y

def draw_graph(canvas, canvas_width, canvas_height, graph_range=1000):
    """Draw the graph grid with square cells and labeled axes."""
    canvas.delete("all")
    square_size = min(canvas_width, canvas_height)
    offset_x = (canvas_width - square_size) // 2
    offset_y = (canvas_height - square_size) // 2

    # Draw gridlines and labels
    for i in range(-graph_range, graph_range + 1, 200):  # Every 200 units for readability
        x1, y1 = map_coordinates(i, -graph_range, square_size, square_size, graph_range)
        x2, y2 = map_coordinates(i, graph_range, square_size, square_size, graph_range)
        y3, y4 = map_coordinates(-graph_range, i, square_size, square_size, graph_range)
        y5, y6 = map_coordinates(graph_range, i, square_size, square_size, graph_range)

        # Vertical gridlines
        canvas.create_line(offset_x + x1, offset_y + y1, offset_x + x2, offset_y + y2, fill="#e0e0e0", width=1)

        # Horizontal gridlines
        canvas.create_line(offset_x + y3, offset_y + y4, offset_x + y5, offset_y + y6, fill="#e0e0e0", width=1)

        # X-axis labels
        if i != 0:
            x, _ = map_coordinates(i, 0, square_size, square_size, graph_range)
            canvas.create_text(offset_x + x, offset_y + square_size // 2 + 15, text=str(i), font=("Arial", 8),
                               fill="black")

            # Y-axis labels
            _, y = map_coordinates(0, i, square_size, square_size, graph_range)
            canvas.create_text(offset_x + square_size // 2 - 15, offset_y + y, text=str(i), font=("Arial", 8),
                               fill="black")

    # Draw axes
    center_x = offset_x + square_size // 2
    center_y = offset_y + square_size // 2
    canvas.create_line(offset_x, center_y, offset_x + square_size, center_y, fill="black", width=2)  # X-axis
    canvas.create_line(center_x, offset_y, center_x, offset_y + square_size, fill="black", width=2)  # Y-axis

def reset_canvas(canvas, points, graph_range=1000):
    """Reset the canvas for a new animation."""
    if not points["main_point"] or not points["other_points"]:
        return
    draw_graph(canvas, canvas.winfo_width(), canvas.winfo_height(), graph_range)
    point_ids = display_points(canvas, points["main_point"], points["other_points"], graph_range)
    points["point_ids"] = point_ids

def display_points(canvas, main_point, other_points, graph_range=1000):
    """Display points on the graph."""
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    square_size = min(canvas_width, canvas_height)
    offset_x = (canvas_width - square_size) // 2
    offset_y = (canvas_height - square_size) // 2

    # Draw all random points
    point_ids = []
    for x, y in other_points:
        px, py = map_coordinates(x, y, square_size, square_size, graph_range)
        point_id = canvas.create_oval(
            offset_x + px - 3, offset_y + py - 3,
            offset_x + px + 3, offset_y + py + 3,
            fill="black"
        )
        point_ids.append(point_id)

    # Draw the main point in red (larger)
    px, py = map_coordinates(main_point[0], main_point[1], square_size, square_size, graph_range)
    canvas.create_oval(
        offset_x + px - 6, offset_y + py - 6,
        offset_x + px + 6, offset_y + py + 6,
        fill="red"
    )

    return point_ids

def brute_force_method(canvas, points, footer_label, graph_range=1000, animation_speed=1.0):
    """Run the brute force method and animate the process."""
    if not points["main_point"] or not points["other_points"]:
        footer_label.config(text="Please generate points first.")
        return

    # Reset the canvas for a clean animation
    reset_canvas(canvas, points, graph_range)

    main_point = points["main_point"]
    other_points = points["other_points"]
    point_ids = points["point_ids"]

    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    square_size = min(canvas_width, canvas_height)
    offset_x = (canvas_width - square_size) // 2
    offset_y = (canvas_height - square_size) // 2

    def map_point(point):
        """Convert graph coordinates to canvas coordinates."""
        return map_coordinates(point[0], point[1], square_size, square_size, graph_range)

    def circle_point(point, color="red", size=6, outline_size=2):
        """Draw a circle around a point."""
        px, py = map_point(point)
        canvas.create_oval(
            offset_x + px - size - outline_size, offset_y + py - size - outline_size,
            offset_x + px + size + outline_size, offset_y + py + size + outline_size,
            outline=color, width=outline_size
        )

    # Sort both points and their IDs together (top-to-bottom, left-to-right)
    sorted_points_with_ids = sorted(zip(other_points, point_ids), key=lambda p: (p[0][1], p[0][0]))
    sorted_points, sorted_ids = zip(*sorted_points_with_ids)

    # Measure internal processing time
    start_time = time.time()
    closest_point = None
    min_distance = float('inf')
    for point in sorted_points:
        distance = math.sqrt((main_point[0] - point[0]) ** 2 +
                             (main_point[1] - point[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_point = point
    internal_time = time.time() - start_time

    # Update footer to show processing time
    footer_label.config(text=f"Processing...")

    # Animation
    def animate(i):
        if i >= len(sorted_points):
            # Highlight closest point and draw the final dotted line
            px1, py1 = map_point(main_point)
            px2, py2 = map_point(closest_point)
            canvas.create_line(
                offset_x + px1, offset_y + py1,
                offset_x + px2, offset_y + py2,
                dash=(3, 3), fill="red", width=2
            )
            circle_point(closest_point, color="red", size=3)
            footer_label.config(
                text=f"Closest Point: {closest_point} | Distance: {round(min_distance, 4)} | "
                     f"Internal Time: {internal_time:.4f} seconds")
            return

        current_point = sorted_points[i]
        current_point_id = sorted_ids[i]
        px1, py1 = map_point(main_point)
        px2, py2 = map_point(current_point)

        # Draw line to the current point
        line_id = canvas.create_line(
            offset_x + px1, offset_y + py1,
            offset_x + px2, offset_y + py2,
            fill="gray", dash=(2, 2), width=1
        )

        # Delay to allow line to render and then turn point to original darker gray
        delay = int(200 / animation_speed)

        def update_point_and_continue():
            # Turn the current point darker gray
            canvas.itemconfig(current_point_id, fill="#808080")
            # Remove the line
            canvas.delete(line_id)
            # Proceed to the next point
            animate(i + 1)

        canvas.after(delay, update_point_and_continue)

    animate(0)

def divide_and_conquer_method(canvas, points, footer_label, graph_range=1000):
    """
    Run a divide-and-conquer style approach to find the single nearest neighbor
    to the main_point. Here, we build a kd-tree (which is O(n log n)) and then
    perform a single nearest-neighbor query (O(log n) average).
    """

    if not points["main_point"] or not points["other_points"]:
        footer_label.config(text="Please generate points first.")
        return

    reset_canvas(canvas, points, graph_range)

    main_point = np.array(points["main_point"], dtype=float)  # We'll search nearest to this
    other_points = np.array(points["other_points"], dtype=float)

    # Euclidean distance helper
    def distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    # Build a kd-tree in a divide-and-conquer manner
    def build_kdtree(pts, depth=0):
        if len(pts) == 0:
            return None
        k = 2  # 2D
        axis = depth % k
        # Sort by the current axis
        pts_sorted = pts[pts[:, axis].argsort()]
        median = len(pts_sorted) // 2

        return {
            'point': pts_sorted[median],
            'left': build_kdtree(pts_sorted[:median], depth + 1),
            'right': build_kdtree(pts_sorted[median + 1:], depth + 1),
            'axis': axis
        }

    # Nearest-neighbor search in the kd-tree
    def kd_nearest(root, target, best=None):
        if root is None:
            return best
        node_point = root['point']
        axis = root['axis']

        # If best is None, initialize it with current node
        if best is None:
            best = (node_point, distance(target, node_point))

        # Check distance to current node
        node_dist = distance(target, node_point)
        if node_dist < best[1]:
            best = (node_point, node_dist)

        # Figure out which side of the split to go
        diff = target[axis] - node_point[axis]
        if diff <= 0:
            best = kd_nearest(root['left'], target, best)
            # Check if we need to explore the other side
            if abs(diff) < best[1]:
                best = kd_nearest(root['right'], target, best)
        else:
            best = kd_nearest(root['right'], target, best)
            if abs(diff) < best[1]:
                best = kd_nearest(root['left'], target, best)

        return best

    # Measure algorithm execution time
    start_time = time.time()

    # Build the tree (divide-and-conquer in spirit)
    kdtree = build_kdtree(other_points)

    # Single nearest neighbor query
    nearest_point, min_distance = kd_nearest(kdtree, main_point)
    internal_time = time.time() - start_time

    # We label the result as a "pair" just to fit the old printing style:
    closest_pair = (tuple(main_point), tuple(nearest_point))

    # Update footer label with results
    footer_label.config(
        text=f"Closest Pair: {closest_pair} | Distance: {round(min_distance, 4)} | Internal Time: {internal_time:.4f} seconds"
    )

def generate_points(points, canvas, footer_label, main_x, main_y, num_points=20, graph_range=1000, clustered=False):
    """Generate random points or clustered points and update the canvas."""
    main_point = (main_x, main_y)

    if clustered:
        num_clusters = random.randint(1, 7)
        cluster_centers = [
            (random.randint(-graph_range, graph_range), random.randint(-graph_range, graph_range))
            for _ in range(num_clusters)
        ]
        other_points = []
        points_per_cluster = max(num_points // num_clusters, 1)
        for center_x, center_y in cluster_centers:
            for _ in range(points_per_cluster):
                x = center_x + random.randint(-graph_range // 10, graph_range // 10)
                y = center_y + random.randint(-graph_range // 10, graph_range // 10)
                other_points.append((x, y))
        other_points = other_points[:num_points]
    else:
        other_points = [
            (random.randint(-graph_range, graph_range), random.randint(-graph_range, graph_range))
            for _ in range(num_points)
        ]

    points["main_point"] = main_point
    points["other_points"] = other_points

    draw_graph(canvas, canvas.winfo_width(), canvas.winfo_height(), graph_range)
    points["point_ids"] = display_points(canvas, main_point, other_points, graph_range)
    footer_label.config(text="Points generated. Ready to run algorithms.")

def create_gui():
    """Create the GUI for the application."""
    graph_range = 1000
    points = {"main_point": None, "other_points": None, "point_ids": []}  # Store points and canvas object IDs
    window = tk.Tk()
    window.title("Closest Point Finder with Animation")
    window.geometry("800x800")  # Window size
    window.resizable(True, True)

    style = ttk.Style()
    style.configure("TFrame", background="#f5f5f5")
    style.configure("TLabel", background="#f5f5f5", font=("Arial", 12))
    style.configure("TButton", font=("Arial", 12, "bold"))
    style.configure("TEntry", font=("Arial", 12))

    input_frame = ttk.Frame(window, padding=10)
    input_frame.pack(fill="x", pady=10)

    ttk.Label(input_frame, text="Main Point X:").grid(row=0, column=0, padx=5, pady=5)
    main_x_entry = ttk.Entry(input_frame, width=10)
    main_x_entry.insert(0, "500")
    main_x_entry.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(input_frame, text="Main Point Y:").grid(row=0, column=2, padx=5, pady=5)
    main_y_entry = ttk.Entry(input_frame, width=10)
    main_y_entry.insert(0, "500")
    main_y_entry.grid(row=0, column=3, padx=5, pady=5)

    ttk.Label(input_frame, text="Number of Points:").grid(row=0, column=4, padx=5, pady=5)
    points_entry = ttk.Entry(input_frame, width=10)
    points_entry.insert(0, "200")
    points_entry.grid(row=0, column=5, padx=5, pady=5)

    ttk.Label(input_frame, text="Animation Speed:").grid(row=0, column=6, padx=5, pady=5)
    speed_entry = ttk.Entry(input_frame, width=10)
    speed_entry.insert(0, "5")
    speed_entry.grid(row=0, column=7, padx=5, pady=5)

    cluster_var = tk.BooleanVar(value=False)
    cluster_checkbox = ttk.Checkbutton(input_frame, text="Generate Clusters", variable=cluster_var)
    cluster_checkbox.grid(row=0, column=8, padx=5, pady=5)

    generate_button = ttk.Button(input_frame, text="Generate Points",
                                 command=lambda: generate_points(points, canvas, footer_label,
                                                                 main_x=int(main_x_entry.get()),
                                                                 main_y=int(main_y_entry.get()),
                                                                 num_points=int(points_entry.get()),
                                                                 graph_range=graph_range,
                                                                 clustered=cluster_var.get()))
    generate_button.grid(row=0, column=9, padx=5, pady=5)

    brute_force_button = ttk.Button(input_frame, text="Brute Force Method",
                                    command=lambda: brute_force_method(canvas, points, footer_label,
                                                                       graph_range=graph_range,
                                                                       animation_speed=float(speed_entry.get())))
    brute_force_button.grid(row=1, column=4, columnspan=3, padx=5, pady=10)

    divide_conquer_button = ttk.Button(input_frame, text="Divide and Conquer Method",
                                       command=lambda: divide_and_conquer_method(canvas, points, footer_label,
                                                                                  graph_range=graph_range))
    divide_conquer_button.grid(row=2, column=4, columnspan=3, padx=5, pady=10)

    canvas_frame = ttk.Frame(window, padding=10)
    canvas_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(canvas_frame, bg="white")
    canvas.pack(fill="both", expand=True)

    footer_frame = ttk.Frame(window, padding=10)
    footer_frame.pack(fill="x")
    footer_label = ttk.Label(footer_frame, text="Closest Point Finder © 2024")
    footer_label.pack()

    # Run the application
    window.mainloop()

if __name__ == "__main__":
    create_gui()
