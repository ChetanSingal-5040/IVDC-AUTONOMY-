import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from collections import deque



# BREADTH FIRST SEARCH

def bfs(grid,start,goal):
    rows=len(grid)
    columns=len(grid[0])

    frontier= deque([start])
    visited={start}
    parent={start: None}
    

    directions =[(-1,0),(1,0),(0,1),(0,-1)]

    while frontier:   # operate until deque is not empty
        current=frontier.popleft()
        

        if current==goal: break
        r,c=current

        for dr,dc in directions:
            nr=r+dr
            nc=c+dc
            next_cell=(nr,nc)

            if  not( 0<= nr <rows and 0<=nc<columns):continue
            if grid[nr][nc]==1: continue
            if next_cell in visited:continue

            visited.add(next_cell)
            parent[next_cell]=current
            frontier.append(next_cell)
            
    
    if goal not in parent:
        return None

    path=[]
    current=goal
    while current is not None:
        path.append(current)
        current=parent[current]
    path.reverse()

    return path   

    #EXAMPLE


grid = [
    [0, 0, 0, 0, 0],  # 0 is free cell 1 is obstacle
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]

start=(0,0)
goal=(4,4)

path=bfs(grid,start,goal)

if path is None:
    print("NO PATH EXIST")
else:
    print("Path: ",path)    
    print("Number of moves:", len(path) - 1)







# Convert the grid into a NumPy array
image = np.array(grid)

# Make a separate array for visualization
visual_grid = image.copy()

# Mark the final path
if path is not None:
    for r, c in path:
        visual_grid[r][c] = 2

    # Mark start and goal separately
    visual_grid[start[0]][start[1]] = 3
    visual_grid[goal[0]][goal[1]] = 4

# Define colors:
# 0 = free, 1 = obstacle, 2 = path, 3 = start, 4 = goal
colors = [
    "#f4f4f4",
    "#333333",
    "#ffb000",
    "#24a148",
    "#da1e28"
]
cmap = ListedColormap(colors)

# Display the grid as an image
plt.figure(figsize=(7, 7))
plt.imshow(visual_grid, cmap=cmap, vmin=-0.5, vmax=4.5)

# Add grid lines between cells
rows = len(grid)
columns = len(grid[0])

plt.xticks(range(columns))
plt.yticks(range(rows))
plt.gca().set_xticks(np.arange(-0.5, columns, 1), minor=True)
plt.gca().set_yticks(np.arange(-0.5, rows, 1), minor=True)
plt.grid(which="minor", color="gray", linewidth=1)
plt.tick_params(which="minor", bottom=False, left=False)

# Add labels
plt.text(start[1], start[0], "S",
         ha="center", va="center", color="white", weight="bold")
plt.text(goal[1], goal[0], "G",
         ha="center", va="center", color="white", weight="bold")

plt.title("BFS Grid Path Planning")
plt.xlabel("Column")
plt.ylabel("Row")
plt.tight_layout()

# Save the image for your assignment
plt.savefig("bfs_path.png", dpi=200)
plt.show()

