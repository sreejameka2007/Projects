

Interactive Network Routing & Shortest Path Simulator
Computer Networks Lab Project 

A Python application built with Tkinter that lets users create and
edit a weighted network of routers, calculate minimum-cost routes using
Dijkstra's algorithm, and understand the result through an
interactive network visualization.
Features
- Interactive network visualization: Displays routers and weighted
  links on a canvas.
- Router management: Add or remove routers to modify the network
  topology.
- Connection management: Add or remove links between routers and
  assign link costs.
- Shortest-path calculation: Select a source and destination and
  calculate the minimum-cost route.
- Visual route highlighting: Highlights the calculated shortest
  path in the network view.
- Result summary: Displays the route, total cost, and number of
  links.
- Routing table: Shows each destination's minimum cost and next
  hop for a selected router.
- Load example network: Loads a predefined network for
  demonstration.
- Reset network: Clears the network so a new topology can be
  created.
- Status messages: Provides feedback for actions and unreachable
  destinations.
Technologies Used
- Language: Python 3
- GUI: Tkinter
- Algorithm: Dijkstra's shortest-path algorithm
- Graph representation: Weighted adjacency dictionary
- Priority queue: Python heapq
- Additional standard-library module: math
No third-party Python packages are required for the application itself.
Tkinter is included with many Python installations; on some Linux
distributions it may need to be installed separately.
Requirements
- Python 3
- Tkinter available in the Python installation
- The project source file, main.py
How to Run
1. Open a terminal or command prompt in the folder containing
   main.py.
2. Run:
   python main.py
   If your system uses python3 for Python 3, run:
   python3 main.py
3. The simulator window should open.
How to Use
1. Launch the application.
2. Load the example network, or create a network using the GUI
   controls.
3. Add routers and connections as needed and assign link weights.
4. Select the source and destination routers.
5. Run the shortest-path calculation.
6. Inspect the displayed route, total cost, and number of links.
7. Observe the shortest route highlighted on the network canvas.
8. Select a router in the network diagram to view its routing table.
9. Edit or reset the network to try another scenario.
Example Network
The built-in example contains these undirected connections:
  Connection     Weight
  R1 -- R2            4
  R1 -- R3            2
  R2 -- R4            3
  R3 -- R4            1
  R3 -- R5            5
  R4 -- R5            2
For source R1 and destination R5, the expected shortest path is:
- Path: R1 → R3 → R4 → R5
- Total cost: 2 + 1 + 2 = 5
- Number of links: 3
Example routing table for router R1:
  Destination     Minimum cost Next hop
  R1                         0 ---
  R2                         4 R2
  R3                         2 R3
  R4                         3 R3
  R5                         5 R3
How It Works
1. The application represents routers as vertices and weighted
   connections as edges.
2. The source distance is initialized to 0; other distances are
   initialized to infinity.
3. Dijkstra's algorithm uses a priority queue to process routers by
   tentative distance.
4. Neighbor distances are updated when a lower-cost route is found.
5. Predecessor information is used to reconstruct the path.
6. The GUI displays the route and cost and highlights the selected
   path.
7. The routing table shows minimum costs and next hops for the selected
   router.
The algorithm assumes non-negative link weights. With an
adjacency-list representation and a binary-heap priority queue, the
typical time complexity is O((V + E) log V), where V is the number
of routers and E is the number of connections.
Project Structure
project-folder/
└── main.py
The application is described as a single-file project. Update this
section if the code is later split into multiple modules or requires
additional assets.
Limitations
- This is an educational simulator, not a production routing protocol.
- It does not send or forward real network packets.
- Link costs are configured in the modeled network, not automatically
  measured from live traffic.
- Route results depend on the topology and weights entered.
- Dijkstra's algorithm is not suitable for graphs containing negative
  edge weights.
Learning Outcomes
- Representing a network using a weighted graph.
- Implementing shortest-path calculation with Dijkstra's algorithm.
- Using a priority queue with Python's heapq.
- Building a desktop GUI with Tkinter.
- Connecting algorithm output to visual feedback.
- Displaying routing costs and next-hop information.

## Author
Amreen fathima 