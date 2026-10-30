import tkinter as tk
from tkinter import ttk, messagebox
import heapq
import math


class NetworkRoutingSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Network Routing Simulator")
        self.root.geometry("1250x820")
        self.root.minsize(1000, 650)

        self.graph = {}
        self.positions = {}
        self.shortest_path = []
        self.path_cost = None
        self.selected_router = None

        self.setup_ui()
        self.refresh_router_choices()
        self.draw_network()

    # ==================================================
    # USER INTERFACE
    # ==================================================

    def setup_ui(self):
        ttk.Label(
            self.root,
            text="Interactive Network Routing & Shortest Path Simulator",
            font=("Arial", 16, "bold")
        ).pack(pady=(5, 2))

        ttk.Label(
            self.root,
            text="Click any router to view its routing table"
        ).pack(pady=(0, 4))

        result_frame = ttk.LabelFrame(
            self.root, text="SHORTEST PATH RESULT", padding=4
        )
        result_frame.pack(fill="x", padx=8, pady=3)

        self.result_label = tk.Label(
            result_frame,
            text="Load the example network or create your own.",
            font=("Arial", 11, "bold"),
            bg="#eff6ff",
            fg="#1d4ed8",
            anchor="w",
            padx=8,
            pady=7
        )
        self.result_label.pack(fill="x")

        main = ttk.Frame(self.root, padding=6)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=0, minsize=285)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        controls = ttk.Frame(main, width=285)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        controls.grid_propagate(False)

        router_box = ttk.LabelFrame(
            controls, text="1. Manage Routers", padding=5
        )
        router_box.pack(fill="x", pady=(0, 5))

        router_row = ttk.Frame(router_box)
        router_row.pack(fill="x")

        self.router_entry = ttk.Entry(router_row)
        self.router_entry.pack(
            side="left", fill="x", expand=True, padx=(0, 4)
        )
        self.router_entry.bind(
            "<Return>", lambda event: self.add_router()
        )

        ttk.Button(
            router_row, text="Add", command=self.add_router
        ).pack(side="right")

        self.router_list = tk.Listbox(
            router_box, height=3, exportselection=False
        )
        self.router_list.pack(fill="x", pady=4)

        ttk.Button(
            router_box,
            text="Remove Selected Router",
            command=self.remove_router
        ).pack(fill="x")

        link_box = ttk.LabelFrame(
            controls, text="2. Manage Connections", padding=5
        )
        link_box.pack(fill="x", pady=(0, 5))

        ttk.Label(link_box, text="Router 1").grid(
            row=0, column=0, sticky="w", pady=2
        )
        self.link_from = ttk.Entry(link_box)
        self.link_from.grid(
            row=0, column=1, sticky="ew", padx=4, pady=2
        )

        ttk.Label(link_box, text="Router 2").grid(
            row=1, column=0, sticky="w", pady=2
        )
        self.link_to = ttk.Entry(link_box)
        self.link_to.grid(
            row=1, column=1, sticky="ew", padx=4, pady=2
        )

        ttk.Label(link_box, text="Link Cost").grid(
            row=2, column=0, sticky="w", pady=2
        )
        self.link_cost = ttk.Entry(link_box)
        self.link_cost.grid(
            row=2, column=1, sticky="ew", padx=4, pady=2
        )
        link_box.columnconfigure(1, weight=1)

        ttk.Button(
            link_box,
            text="Add / Update Connection",
            command=self.add_link
        ).grid(
            row=3, column=0, columnspan=2,
            sticky="ew", pady=(4, 2)
        )

        ttk.Button(
            link_box,
            text="Remove Connection",
            command=self.remove_link
        ).grid(
            row=4, column=0, columnspan=2, sticky="ew"
        )

        path_box = ttk.LabelFrame(
            controls, text="3. Find Shortest Path", padding=5
        )
        path_box.pack(fill="x", pady=(0, 5))

        ttk.Label(path_box, text="Source").grid(
            row=0, column=0, sticky="w", pady=2
        )
        self.source_combo = ttk.Combobox(
            path_box, state="readonly"
        )
        self.source_combo.grid(
            row=0, column=1, sticky="ew", padx=4, pady=2
        )

        ttk.Label(path_box, text="Destination").grid(
            row=1, column=0, sticky="w", pady=2
        )
        self.destination_combo = ttk.Combobox(
            path_box, state="readonly"
        )
        self.destination_combo.grid(
            row=1, column=1, sticky="ew", padx=4, pady=2
        )
        path_box.columnconfigure(1, weight=1)

        ttk.Button(
            path_box,
            text="CALCULATE SHORTEST PATH",
            command=self.find_shortest_path
        ).grid(
            row=2, column=0, columnspan=2,
            sticky="ew", pady=(5, 3), ipady=3
        )

        ttk.Button(
            path_box,
            text="LOAD EXAMPLE NETWORK",
            command=self.load_example
        ).grid(
            row=3, column=0, columnspan=2,
            sticky="ew", pady=2, ipady=2
        )

        ttk.Button(
            path_box,
            text="Reset Network",
            command=self.reset_network
        ).grid(
            row=4, column=0, columnspan=2,
            sticky="ew", pady=2
        )

        ttk.Label(
            controls,
            text="Connections are bidirectional.\n"
                 "Click a router in the diagram to\n"
                 "view its individual routing table.",
            font=("Arial", 9),
            justify="left"
        ).pack(anchor="w", pady=3)

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=2)

        network_box = ttk.LabelFrame(
            right, text="Network Topology — Click a Router", padding=4
        )
        network_box.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        network_box.columnconfigure(0, weight=1)
        network_box.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            network_box,
            background="white",
            highlightthickness=1,
            highlightbackground="#cccccc"
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind(
            "<Configure>", lambda event: self.draw_network()
        )
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        table_box = ttk.LabelFrame(
            right, text="Router Routing Table", padding=5
        )
        table_box.grid(row=1, column=0, sticky="nsew")
        table_box.columnconfigure(0, weight=1)
        table_box.rowconfigure(1, weight=1)

        self.table_heading = ttk.Label(
            table_box,
            text="Click a router in the network diagram",
            font=("Arial", 10, "bold")
        )
        self.table_heading.grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )

        # UPDATED: Added Prev Node column
        self.routing_table = ttk.Treeview(
            table_box,
            columns=("destination", "cost", "prev_node", "next_hop"),
            show="headings",
            height=5
        )

        self.routing_table.heading(
            "destination", text="Destination Router"
        )
        self.routing_table.heading(
            "cost", text="Minimum Cost"
        )
        self.routing_table.heading(
            "prev_node", text="Prev Node"
        )
        self.routing_table.heading(
            "next_hop", text="Next Hop"
        )

        self.routing_table.column(
            "destination", width=160, anchor="center"
        )
        self.routing_table.column(
            "cost", width=110, anchor="center"
        )
        self.routing_table.column(
            "prev_node", width=120, anchor="center"
        )
        self.routing_table.column(
            "next_hop", width=120, anchor="center"
        )

        self.routing_table.grid(
            row=1, column=0, sticky="nsew"
        )

        table_scroll = ttk.Scrollbar(
            table_box,
            orient="vertical",
            command=self.routing_table.yview
        )
        table_scroll.grid(row=1, column=1, sticky="ns")
        self.routing_table.configure(
            yscrollcommand=table_scroll.set
        )

        self.status = ttk.Label(
            self.root,
            text="Ready",
            relief="sunken",
            anchor="w",
            padding=3
        )
        self.status.pack(side="bottom", fill="x")

    # ==================================================
    # ROUTING TABLE
    # ==================================================

    def clear_routing_table(self):
        for item in self.routing_table.get_children():
            self.routing_table.delete(item)

    def show_router_table(self, router):
        if router not in self.graph:
            return

        self.selected_router = router
        distances, previous = self.dijkstra(router)

        self.clear_routing_table()
        self.table_heading.config(
            text=f"Routing Table for Router {router}"
        )

        for destination in sorted(self.graph):
            if math.isinf(distances[destination]):
                cost_text = "Unreachable"
                prev_node = "-"
                next_hop = "-"

            elif destination == router:
                cost_text = "0"
                prev_node = "-"
                next_hop = "-"

            else:
                cost_text = f"{distances[destination]:g}"

                # Previous node before the destination
                prev_node = previous[destination] or "-"

                # First router to forward packets through
                route = self.reconstruct_path(
                    router, destination, previous
                )

                next_hop = (
                    route[1]
                    if route and len(route) > 1
                    else "-"
                )

            self.routing_table.insert(
                "",
                "end",
                values=(
                    destination,
                    cost_text,
                    prev_node,
                    next_hop
                )
            )

        self.draw_network()
        self.status.config(
            text=f"Showing routing table for {router}."
        )

    # ==================================================
    # ROUTER SELECTION
    # ==================================================

    def on_canvas_click(self, event):
        for router, (x, y) in self.positions.items():
            distance = math.sqrt(
                (event.x - x) ** 2 + (event.y - y) ** 2
            )

            if distance <= 25:
                self.show_router_table(router)
                return

    # ==================================================
    # ROUTER OPERATIONS
    # ==================================================

    def refresh_router_choices(self):
        routers = sorted(self.graph)

        self.router_list.delete(0, tk.END)
        for router in routers:
            self.router_list.insert(tk.END, router)

        old_source = self.source_combo.get()
        old_destination = self.destination_combo.get()

        self.source_combo["values"] = routers
        self.destination_combo["values"] = routers

        self.source_combo.set(
            old_source if old_source in routers
            else (routers[0] if routers else "")
        )

        self.destination_combo.set(
            old_destination if old_destination in routers
            else (
                routers[1] if len(routers) > 1
                else (routers[0] if routers else "")
            )
        )

    def add_router(self):
        name = self.router_entry.get().strip().upper()

        if not name:
            messagebox.showwarning(
                "Missing Input", "Enter a router name."
            )
            return

        if any(char.isspace() for char in name):
            messagebox.showwarning(
                "Invalid Name",
                "Router names cannot contain spaces."
            )
            return

        if name in self.graph:
            messagebox.showwarning(
                "Duplicate Router",
                f"{name} already exists."
            )
            return

        self.graph[name] = {}
        self.router_entry.delete(0, tk.END)

        self.refresh_router_choices()
        self.invalidate_results()
        self.status.config(text=f"Added router {name}.")

    def remove_router(self):
        selection = self.router_list.curselection()

        if not selection:
            messagebox.showwarning(
                "Select Router",
                "Select a router from the list first."
            )
            return

        name = self.router_list.get(selection[0])
        del self.graph[name]

        for neighbors in self.graph.values():
            neighbors.pop(name, None)

        self.refresh_router_choices()
        self.invalidate_results()

        if self.selected_router == name:
            self.selected_router = None
            self.table_heading.config(
                text="Click a router in the network diagram"
            )
            self.clear_routing_table()

        self.status.config(text=f"Removed router {name}.")

    # ==================================================
    # CONNECTION OPERATIONS
    # ==================================================

    def add_link(self):
        r1 = self.link_from.get().strip().upper()
        r2 = self.link_to.get().strip().upper()
        cost_text = self.link_cost.get().strip()

        if r1 not in self.graph or r2 not in self.graph:
            messagebox.showerror(
                "Invalid Routers",
                "Both routers must exist before connecting them."
            )
            return

        if r1 == r2:
            messagebox.showerror(
                "Invalid Connection",
                "A router cannot connect to itself."
            )
            return

        try:
            cost = float(cost_text)
            if not math.isfinite(cost) or cost < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid Cost",
                "Enter a valid non-negative link cost."
            )
            return

        self.graph[r1][r2] = cost
        self.graph[r2][r1] = cost

        self.link_from.delete(0, tk.END)
        self.link_to.delete(0, tk.END)
        self.link_cost.delete(0, tk.END)

        self.invalidate_results()

        if self.selected_router:
            self.show_router_table(self.selected_router)

        self.status.config(
            text=f"Added connection {r1} <--> {r2}, cost {cost:g}."
        )

    def remove_link(self):
        r1 = self.link_from.get().strip().upper()
        r2 = self.link_to.get().strip().upper()

        if r1 not in self.graph or r2 not in self.graph:
            messagebox.showerror(
                "Invalid Routers",
                "Enter two existing router names."
            )
            return

        if r2 not in self.graph[r1]:
            messagebox.showwarning(
                "Connection Not Found",
                "These routers are not connected."
            )
            return

        del self.graph[r1][r2]
        del self.graph[r2][r1]

        self.invalidate_results()

        if self.selected_router:
            self.show_router_table(self.selected_router)

        self.status.config(
            text=f"Removed connection between {r1} and {r2}."
        )

    def invalidate_results(self):
        self.shortest_path = []
        self.path_cost = None

        self.result_label.config(
            text="Network changed. Recalculate the shortest path.",
            fg="#1d4ed8",
            bg="#eff6ff"
        )

        self.draw_network()

    # ==================================================
    # DIJKSTRA ALGORITHM
    # ==================================================

    def dijkstra(self, source):
        distances = {
            router: math.inf for router in self.graph
        }
        previous = {
            router: None for router in self.graph
        }

        distances[source] = 0
        queue = [(0, source)]
        visited = set()

        while queue:
            distance, current = heapq.heappop(queue)

            if current in visited:
                continue

            visited.add(current)

            for neighbor, cost in self.graph[current].items():
                if neighbor in visited:
                    continue

                new_distance = distance + cost

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current
                    heapq.heappush(
                        queue, (new_distance, neighbor)
                    )

        return distances, previous

    def reconstruct_path(self, source, destination, previous):
        if source == destination:
            return [source]

        if previous[destination] is None:
            return None

        path = []
        current = destination

        while current is not None:
            path.append(current)
            current = previous[current]

        path.reverse()

        if path and path[0] == source:
            return path

        return None

    def find_shortest_path(self):
        source = self.source_combo.get()
        destination = self.destination_combo.get()

        if not source or not destination:
            messagebox.showwarning(
                "Select Routers",
                "Add routers and select source and destination."
            )
            return

        distances, previous = self.dijkstra(source)
        path = self.reconstruct_path(
            source, destination, previous
        )

        if path is None:
            self.shortest_path = []
            self.path_cost = None

            self.result_label.config(
                text=f"NO ROUTE FOUND: {source} to {destination}",
                fg="#b91c1c",
                bg="#fee2e2"
            )
            self.status.config(text="Destination is unreachable.")

        else:
            self.shortest_path = path
            self.path_cost = distances[destination]

            self.result_label.config(
                text=(
                    f"SHORTEST PATH: {' -> '.join(path)}   |   "
                    f"TOTAL COST: {self.path_cost:g}   |   "
                    f"LINKS: {len(path) - 1}"
                ),
                fg="#166534",
                bg="#dcfce7"
            )
            self.status.config(
                text="Shortest path calculated successfully."
            )

        if self.selected_router in self.graph:
            self.show_router_table(self.selected_router)
        else:
            self.draw_network()

    # ==================================================
    # EXAMPLE NETWORK AND RESET
    # ==================================================

    def load_example(self):
        self.graph = {
            "R1": {"R2": 4, "R3": 2},
            "R2": {"R1": 4, "R4": 3},
            "R3": {"R1": 2, "R4": 1, "R5": 5},
            "R4": {"R2": 3, "R3": 1, "R5": 2},
            "R5": {"R3": 5, "R4": 2}
        }

        self.shortest_path = []
        self.path_cost = None
        self.selected_router = None

        self.refresh_router_choices()
        self.source_combo.set("R1")
        self.destination_combo.set("R5")

        self.result_label.config(
            text="Example loaded. Click a router to view its table.",
            fg="#1d4ed8",
            bg="#eff6ff"
        )

        self.clear_routing_table()
        self.table_heading.config(
            text="Click a router in the network diagram"
        )

        self.draw_network()
        self.status.config(
            text="Example loaded. Select a router or calculate a path."
        )

    def reset_network(self):
        self.graph = {}
        self.shortest_path = []
        self.path_cost = None
        self.selected_router = None
        self.positions = {}

        self.refresh_router_choices()
        self.clear_routing_table()
        self.table_heading.config(
            text="Click a router in the network diagram"
        )

        self.result_label.config(
            text="Network cleared. Add routers to begin.",
            fg="#1d4ed8",
            bg="#eff6ff"
        )

        self.draw_network()
        self.status.config(text="Network cleared.")

    # ==================================================
    # NETWORK VISUALIZATION
    # ==================================================

    def draw_network(self):
        if not hasattr(self, "canvas"):
            return

        self.canvas.delete("all")

        width = max(self.canvas.winfo_width(), 400)
        height = max(self.canvas.winfo_height(), 250)
        routers = sorted(self.graph)
        self.positions = {}

        if not routers:
            self.canvas.create_text(
                width / 2, height / 2,
                text="Network appears here.\n"
                     "Click LOAD EXAMPLE NETWORK to begin.",
                font=("Arial", 12),
                fill="#64748b",
                justify="center"
            )
            return

        cx, cy = width / 2, height / 2
        node_radius = 23
        margin = node_radius + 28
        usable_w = max(100, width - 2 * margin)
        usable_h = max(100, height - 2 * margin)
        n = len(routers)

        if n == 1:
            self.positions[routers[0]] = (cx, cy)

        else:
            # Initial circular positions; all custom networks use
            # the same automatic layout procedure.
            radius_x = max(45, usable_w * 0.36)
            radius_y = max(45, usable_h * 0.36)

            pos = {}
            for i, router in enumerate(routers):
                angle = -math.pi / 2 + 2 * math.pi * i / n
                pos[router] = [
                    cx + radius_x * math.cos(angle),
                    cy + radius_y * math.sin(angle)
                ]

            edges = [
                (r1, r2)
                for r1 in routers
                for r2 in self.graph[r1]
                if r1 < r2
            ]

            # Force-directed layout: repel nodes and attract linked nodes.
            ideal = max(
                65,
                min(
                    125,
                    math.sqrt(
                        (usable_w * usable_h) / max(n, 1)
                    ) * 0.62
                )
            )

            iterations = min(180, 70 + n * 5)

            for step in range(iterations):
                forces = {
                    router: [0.0, 0.0] for router in routers
                }
                temperature = max(
                    1.5, 18.0 * (1.0 - step / iterations)
                )

                # Repulsion between all routers.
                for i, r1 in enumerate(routers):
                    x1, y1 = pos[r1]

                    for r2 in routers[i + 1:]:
                        x2, y2 = pos[r2]
                        dx, dy = x1 - x2, y1 - y2
                        dist2 = dx * dx + dy * dy

                        if dist2 < 1:
                            dx, dy, dist2 = 1.0, 0.5, 1.25

                        dist = math.sqrt(dist2)
                        force = (ideal * ideal) / max(dist, 1.0)
                        fx = dx / dist * force
                        fy = dy / dist * force

                        forces[r1][0] += fx
                        forces[r1][1] += fy
                        forces[r2][0] -= fx
                        forces[r2][1] -= fy

                # Attraction between connected routers.
                for r1, r2 in edges:
                    x1, y1 = pos[r1]
                    x2, y2 = pos[r2]
                    dx, dy = x2 - x1, y2 - y1
                    dist = max(math.hypot(dx, dy), 1.0)
                    force = (dist * dist) / max(ideal, 1.0)
                    fx = dx / dist * force
                    fy = dy / dist * force

                    forces[r1][0] += fx
                    forces[r1][1] += fy
                    forces[r2][0] -= fx
                    forces[r2][1] -= fy

                # Keep the network inside the canvas.
                for router in routers:
                    x, y = pos[router]

                    fx = forces[router][0] - (x - cx) * 0.035
                    fy = forces[router][1] - (y - cy) * 0.035

                    magnitude = max(math.hypot(fx, fy), 0.001)
                    move = min(magnitude, temperature)

                    pos[router][0] += fx / magnitude * move
                    pos[router][1] += fy / magnitude * move

                    pos[router][0] = min(
                        width - margin,
                        max(margin, pos[router][0])
                    )
                    pos[router][1] = min(
                        height - margin,
                        max(margin, pos[router][1])
                    )

            self.positions = {
                router: (pos[router][0], pos[router][1])
                for router in routers
            }

            # Count crossings between lines that do not share endpoints.
            def crossing_count(candidate_positions):
                segments = []

                for a, b in edges:
                    ax, ay = candidate_positions[a]
                    bx, by = candidate_positions[b]
                    segments.append((a, b, ax, ay, bx, by))

                def orient(ax, ay, bx, by, px, py):
                    return (
                        (bx - ax) * (py - ay)
                        - (by - ay) * (px - ax)
                    )

                count = 0

                for i, first in enumerate(segments):
                    a, b, ax, ay, bx, by = first

                    for second in segments[i + 1:]:
                        c, d, cx_, cy_, dx_, dy_ = second

                        if len({a, b, c, d}) < 4:
                            continue

                        o1 = orient(ax, ay, bx, by, cx_, cy_)
                        o2 = orient(ax, ay, bx, by, dx_, dy_)
                        o3 = orient(cx_, cy_, dx_, dy_, ax, ay)
                        o4 = orient(cx_, cy_, dx_, dy_, bx, by)

                        if o1 * o2 < 0 and o3 * o4 < 0:
                            count += 1

                return count

            # Try swapping router positions to reduce crossings.
            best = crossing_count(self.positions)

            for _ in range(3):
                improved = False

                for i in range(n):
                    for j in range(i + 1, n):
                        a, b = routers[i], routers[j]

                        self.positions[a], self.positions[b] = (
                            self.positions[b],
                            self.positions[a]
                        )

                        score = crossing_count(self.positions)

                        if score < best:
                            best = score
                            improved = True
                        else:
                            self.positions[a], self.positions[b] = (
                                self.positions[b],
                                self.positions[a]
                            )

                if not improved or best == 0:
                    break

        # Draw each undirected connection once, before the routers.
        edges_to_draw = []

        for r1 in routers:
            for r2, cost in self.graph[r1].items():
                if r1 < r2:
                    edges_to_draw.append((r1, r2, cost))

        for r1, r2, cost in edges_to_draw:
            x1, y1 = self.positions[r1]
            x2, y2 = self.positions[r2]

            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)

            if length < 1:
                continue

            ux, uy = dx / length, dy / length

            # Stop lines at router-circle edges.
            sx = x1 + ux * node_radius
            sy = y1 + uy * node_radius
            ex = x2 - ux * node_radius
            ey = y2 - uy * node_radius

            highlighted = any(
                {r1, r2} == {
                    self.shortest_path[i],
                    self.shortest_path[i + 1]
                }
                for i in range(len(self.shortest_path) - 1)
            )

            self.canvas.create_line(
                sx, sy, ex, ey,
                fill="#16a34a" if highlighted else "#94a3b8",
                width=4 if highlighted else 2
            )

            # Place cost labels slightly away from each connection.
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            label_x = mx - uy * 12
            label_y = my + ux * 12

            self.canvas.create_rectangle(
                label_x - 15, label_y - 9,
                label_x + 15, label_y + 9,
                fill="white",
                outline="white"
            )

            self.canvas.create_text(
                label_x, label_y,
                text=f"{cost:g}",
                font=("Arial", 10, "bold"),
                fill="#334155"
            )

        # Draw routers last so they remain visible and clickable.
        for router in routers:
            x, y = self.positions[router]

            on_path = router in self.shortest_path
            selected = router == self.selected_router

            if selected:
                fill_color = "#fde68a"
                outline_color = "#d97706"
            elif on_path:
                fill_color = "#bbf7d0"
                outline_color = "#15803d"
            else:
                fill_color = "#dbeafe"
                outline_color = "#2563eb"

            self.canvas.create_oval(
                x - node_radius, y - node_radius,
                x + node_radius, y + node_radius,
                fill=fill_color,
                outline=outline_color,
                width=3
            )

            self.canvas.create_text(
                x, y,
                text=router,
                font=("Arial", 10, "bold"),
                fill="#1e3a8a"
            )

        self.canvas.create_text(
            8, 8,
            anchor="nw",
            text=(
                "Click a router to view its routing table\n"
                "Green = shortest path | Yellow = selected router"
            ),
            font=("Arial", 9),
            fill="#475569"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = NetworkRoutingSimulator(root)
    root.mainloop()