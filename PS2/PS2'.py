
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def dijkstra(grid, cost_grid, start, goal):
    rows = len(grid)
    columns = len(grid[0])

    
    frontier = [(0, start)]

    # Lowest known cost to reach each cell
    distance = {start: 0}

    
    parent = {start: None}

    explored = []

    
    directions =[(-1,0),(1,0),(0,1),(0,-1)]

    while frontier:
        current_cost, current = heapq.heappop(frontier)

       
        if current_cost != distance[current]:
            continue

        explored.append(current)

        
        if current == goal:
            break

        r, c = current

        for dr, dc in directions:
            nr = r + dr
            nc = c + dc
            next_cell = (nr, nc)

           
           
            if  not( 0<= nr <rows and 0<=nc<columns):continue
            if grid[nr][nc]==1: continue
            

            # Cost of entering the next cell
            new_cost = current_cost + cost_grid[nr][nc]

            # Update if this route is cheaper
            if new_cost < distance.get(next_cell, float("inf")):
                distance[next_cell] = new_cost
                parent[next_cell] = current

                heapq.heappush(
                    frontier, (new_cost, next_cell)
                )

   
    if goal not in parent:
        return None, explored, None

   
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()

    return path, explored, distance[goal]


# Grid: 0 = free cell, 1 = obstacle
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]

# Cost of entering each cell
# 1 = normal terrain, 5 = expensive terrain
cost_grid = [
    [1, 1, 5, 5, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 5, 5, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
]

start = (0, 0)
goal = (4, 4)

path, explored, total_cost = dijkstra(
    grid, cost_grid, start, goal
)

if path is None:
    print("NO PATH EXISTS")
else:
    print("Path:", path)
    print("Number of moves:", len(path) - 1)
    print("Total movement cost:", total_cost)

# Prepare visualization
# 0 = normal, 1 = obstacle, 2 = expensive
# 3 = explored, 4 = path, 5 = start, 6 = goal
visual_grid = np.zeros((len(grid), len(grid[0])), dtype=int)

for r in range(len(grid)):
    for c in range(len(grid[0])):
        if grid[r][c] == 1:
            visual_grid[r][c] = 1
        elif cost_grid[r][c] > 1:
            visual_grid[r][c] = 2

# Mark explored cells
for r, c in explored:
    if visual_grid[r][c] in (0, 2):
        visual_grid[r][c] = 3

# Mark final path
if path is not None:
    for r, c in path:
        visual_grid[r][c] = 4

# Mark start and goal last
visual_grid[start[0]][start[1]] = 5
visual_grid[goal[0]][goal[1]] = 6

colors = [
    "#f4f4f4",  # Normal cell
    "#333333",  # Obstacle
    "#f4a261",  # Expensive cell
    "#70b7e6",  # Explored cell
    "#ffcc33",  # Final path
    "#24a148",  # Start
    "#da1e28"   # Goal
]

cmap = ListedColormap(colors)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(visual_grid, cmap=cmap, vmin=-0.5, vmax=6.5)

rows = len(grid)
columns = len(grid[0])

ax.set_xticks(range(columns))
ax.set_yticks(range(rows))

ax.set_xticks(np.arange(-0.5, columns, 1), minor=True)
ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
ax.grid(which="minor", color="gray", linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)

# Display terrain cost in each cell
for r in range(rows):
    for c in range(columns):
        if grid[r][c] == 1:
            label = "#"
        elif (r, c) == start:
            label = "S"
        elif (r, c) == goal:
            label = "G"
        else:
            label = str(cost_grid[r][c])

        ax.text(
            c, r, label,
            ha="center", va="center",
            color="white" if visual_grid[r][c] in (1, 5, 6)
            else "black",
            weight="bold"
        )

legend_items = [
    Patch(color=colors[0], label="Normal cost (1)"),
    Patch(color=colors[1], label="Obstacle"),
    Patch(color=colors[2], label="Expensive cost (5)"),
    Patch(color=colors[3], label="Explored"),
    Patch(color=colors[4], label="Final path"),
    Patch(color=colors[5], label="Start"),
    Patch(color=colors[6], label="Goal")
]

ax.legend(handles=legend_items, loc="upper left",
          bbox_to_anchor=(1.02, 1))

ax.set_title("Dijkstra: Weighted Grid Path Planning")
ax.set_xlabel("Column")
ax.set_ylabel("Row")

plt.tight_layout()
plt.savefig("dijkstra_weighted.png", dpi=200, bbox_inches="tight")
plt.show()
