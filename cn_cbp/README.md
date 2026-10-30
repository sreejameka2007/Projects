# Interactive Network Routing & Shortest Path Simulator

**Computer Networks Lab Project**

A Python application built with Tkinter that lets users create and edit a weighted network of routers, calculate minimum-cost routes using Dijkstra's algorithm, and understand the results through an interactive network visualization.

## Features

- **Interactive Network Visualization:** Displays routers and weighted links on a canvas.
- **Router Management:** Add or remove routers to modify the network topology.
- **Connection Management:** Add or remove links between routers and assign link costs.
- **Shortest-Path Calculation:** Select a source and destination and calculate the minimum-cost route.
- **Visual Route Highlighting:** Highlights the calculated shortest path in the network view.
- **Result Summary:** Displays the route, total cost, and number of links.
- **Routing Table:** Shows each destination's minimum cost and next hop for a selected router.
- **Load Example Network:** Loads a predefined network for demonstration.
- **Reset Network:** Clears the network so a new topology can be created.
- **Status Messages:** Provides feedback for actions and unreachable destinations.

## Technologies Used

- **Programming Language:** Python 3
- **GUI:** Tkinter
- **Algorithm:** Dijkstra's shortest-path algorithm
- **Graph Representation:** Weighted adjacency dictionary
- **Priority Queue:** Python `heapq`
- **Additional Standard-Library Module:** `math`

No third-party Python packages are required for the application itself. Tkinter is included with many Python installations; on some Linux distributions, it may need to be installed separately.

## Requirements

- Python 3
- Tkinter available in the Python installation
- Project source file: `main.py`

## How to Run

1. Open a terminal or command prompt in the folder containing `main.py`.
2. Run the following command:

   ```bash
   python main.py
   ```

   If your system uses `python3` for Python 3, run:

   ```bash
   python3 main.py
   ```

3. The simulator window should open.

## How to Use

1. Launch the application.
2. Load the example network or create a network using the GUI controls.
3. Add routers and connections as needed and assign link weights.
4. Select the source and destination routers.
5. Run the shortest-path calculation.
6. Inspect the displayed route, total cost, and number of links.
7. Observe the shortest route highlighted on the network canvas.
8. Select a router in the network diagram to view its routing table.
9. Edit or reset the network to try another scenario.

## Example Network

The built-in example contains the following undirected connections:

| Connection | Weight |
|---|---:|
| R1 -- R2 | 4 |
| R1 -- R3 | 2 |
| R2 -- R4 | 3 |
| R3 -- R4 | 1 |
| R3 -- R5 | 5 |
| R4 -- R5 | 2 |

For source **R1** and destination **R5**, the expected shortest path is:

- **Path:** R1 → R3 → R4 → R5
- **Total Cost:** 2 + 1 + 2 = 5
- **Number of Links:** 3

### Example Routing Table for Router R1

| Destination | Minimum Cost | Next Hop |
|---|---:|---|
| R1 | 0 | --- |
| R2 | 4 | R2 |
| R3 | 2 | R3 |
| R4 | 3 | R3 |
| R5 | 5 | R3 |

## How It Works

1. The application represents routers as vertices and weighted connections as edges.
2. The source distance is initialized to 0, while the distances to all other routers are initialized to infinity.
3. Dijkstra's algorithm uses a priority queue to process routers by tentative distance.
4. Neighbor distances are updated whenever a lower-cost route is found.
5. Predecessor information is used to reconstruct the shortest path.
6. The GUI displays the route and cost and highlights the selected path.
7. The routing table shows minimum costs and next hops for the selected router.

The algorithm assumes non-negative link weights. With an adjacency-list representation and a binary-heap priority queue, the typical time complexity is **O((V + E) log V)**, where V is the number of routers and E is the number of connections.

## Project Structure

```text
cn_cbp/
├── main.py
├── README.md
└── .gitignore
```

The application is implemented as a single Python source file.

## Limitations

- This is an educational simulator, not a production routing protocol.
- It does not send or forward real network packets.
- Link costs are configured in the modeled network rather than automatically measured from live traffic.
- Route results depend on the topology and weights entered.
- Dijkstra's algorithm is not suitable for graphs containing negative edge weights.

## Learning Outcomes

- Representing a network using a weighted graph.
- Implementing shortest-path calculation with Dijkstra's algorithm.
- Using a priority queue with Python's `heapq`.
- Building a desktop GUI with Tkinter.
- Connecting algorithm output to visual feedback.
- Displaying routing costs and next-hop information.

## Author

**Amreen Fathima**

## Team Members

1. **Amreen Fathima** — Original project author and primary understanding of the project concept and implementation.
2. **Meka Sreeja** — Code modifications, project support, and understanding of the implementation.
3. **Prasanya** — CBP documentation and understanding of the project.
4. **Mudassir** — Understanding of the project concept.

## Project Information

- **Project Title:** Interactive Network Routing & Shortest Path Simulator
- **Course:** Computer Networks Lab
- **Project Type:** Course-Based Project (CBP)
- **Repository Folder:** `cn_cbp`
