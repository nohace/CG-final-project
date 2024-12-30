import random
import math
import tkinter as tk
from tkinter import ttk
import time


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

        # Delay to allow line to render and then turn point gray
        delay = int(200 / animation_speed)

        def update_point_and_continue():
            # Turn the current point gray
            canvas.itemconfig(current_point_id, fill="gray")
            # Remove the line
            canvas.delete(line_id)
            # Proceed to the next point
            animate(i + 1)

        canvas.after(delay, update_point_and_continue)

    animate(0)






def generate_points(points, canvas, footer_label, main_x, main_y, num_points=20, graph_range=1000):
    """Generate random points and update the canvas."""
    main_point = (main_x, main_y)
    other_points = [(random.randint(-graph_range, graph_range), random.randint(-graph_range, graph_range))
                    for _ in range(num_points)]
    points["main_point"] = main_point
    points["other_points"] = other_points

    # Update the canvas with new points
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

    # Style configuration
    style = ttk.Style()
    style.configure("TFrame", background="#f5f5f5")
    style.configure("TLabel", background="#f5f5f5", font=("Arial", 12))
    style.configure("TButton", font=("Arial", 12, "bold"))
    style.configure("TEntry", font=("Arial", 12))

    # Top Frame for inputs and buttons
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
    points_entry.insert(0, "20")
    points_entry.grid(row=0, column=5, padx=5, pady=5)

    ttk.Label(input_frame, text="Animation Speed:").grid(row=0, column=6, padx=5, pady=5)
    speed_entry = ttk.Entry(input_frame, width=10)
    speed_entry.insert(0, "1.0")
    speed_entry.grid(row=0, column=7, padx=5, pady=5)

    generate_button = ttk.Button(input_frame, text="Generate Points",
                                 command=lambda: generate_points(points, canvas, footer_label,
                                                                 main_x=int(main_x_entry.get()),
                                                                 main_y=int(main_y_entry.get()),
                                                                 num_points=int(points_entry.get()),
                                                                 graph_range=graph_range))
    generate_button.grid(row=0, column=8, padx=5, pady=5)

    brute_force_button = ttk.Button(input_frame, text="Brute Force Method",
                                    command=lambda: brute_force_method(canvas, points, footer_label,
                                                                       graph_range=graph_range,
                                                                       animation_speed=float(speed_entry.get())))
    brute_force_button.grid(row=0, column=9, padx=5, pady=5)

    # Canvas for visualization
    canvas_frame = ttk.Frame(window, padding=10)
    canvas_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(canvas_frame, bg="white")
    canvas.pack(fill="both", expand=True)

    # Footer
    footer_frame = ttk.Frame(window, padding=10)
    footer_frame.pack(fill="x")
    footer_label = ttk.Label(footer_frame, text="Closest Point Finder © 2024")
    footer_label.pack()

    # Run the application
    window.mainloop()


if __name__ == "__main__":
    create_gui()
