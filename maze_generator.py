import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored
# Maze generator following randomized prim's algorithm


def checkSurround(maze, wall):
    size = len(maze)
    total = 0
    if (wall[0] > 0 and maze[wall[0] - 1][wall[1]] == 2):
        total += 1
    if (wall[0] < size - 1 and maze[wall[0] + 1][wall[1]] == 2):
        total += 1
    if (wall[1] > 0 and maze[wall[0]][wall[1] - 1] == 2):
        total += 1
    if (wall[1] < size - 1 and maze[wall[0]][wall[1] + 1] == 2):
        total += 1

    if total < 2:
        return True
    return False

def solveMaze(maze, mode='DFS'):
    # Implements DFS
    size = len(maze)
    parent = {}
    visited = []
    stack = []
    start = (0, size // 2)
    end = (size-1, size // 2)
    visited.append(start)
    stack.append(start)
    while len(stack) > 0:
        if mode == 'DFS':
            pos = stack.pop()
        else:
            pos = stack.pop(0)

        if pos == end:
            return parent
            
        x, y = pos

        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y  + 1)]

        for neighbor in neighbors:
            nx, ny = neighbor

            if nx >= 0 and nx <= size - 1 and ny >= 0 and ny <= size - 1 and maze[nx][ny] == 2 and neighbor not in visited:
                visited.append(neighbor)
                if mode == 'DFS':
                    stack.insert(0, neighbor)
                else:
                    stack.append(neighbor)
                parent[neighbor] = pos

    return None



def generate_maze(num_mazes, size, solveMode='DFS', save_img=True, combined=False):
    for maze_num in range(num_mazes):
        maze = np.zeros((size, size))
        # 0 represents unvisited 1 represents a wall and 2 represents a space
        start = (np.random.randint(1, size), np.random.randint(1, size))
        maze[start[0]][start[1]] = 2
        wall_list = []
        if start[0] - 1 >= 0:
            wall_list.append((start[0] - 1, start[1]))
            maze[start[0] - 1][start[1]] = 1
        if start[0] + 1 < size:
            wall_list.append((start[0] + 1, start[1]))
            maze[start[0] + 1][start[1]] = 1
        if start[1] - 1 >= 0:
            wall_list.append((start[0], start[1] - 1))
            maze[start[0]][start[1] - 1] = 1
        if start[1] + 1 < size:
            wall_list.append((start[0], start[1] + 1))
            maze[start[0]][start[1] + 1] = 1

        k = 0
        while len(wall_list) > 0:
            wall_index = np.random.randint(0, len(wall_list))
            wall = wall_list[wall_index]
            if wall[0] > 0 and wall[0] < size - 1 and maze[wall[0] - 1][wall[1]] == 0 and maze[wall[0] + 1][wall[1]] == 2:
                if checkSurround(maze, wall):
                    maze[wall[0]][wall[1]] = 2
                
                    if wall[1] > 0 and maze[wall[0]][wall[1] - 1] != 2:
                        maze[wall[0]][wall[1] - 1] = 1
                    if wall[1] > 0 and (wall[0], wall[1] - 1) not in wall_list:
                        wall_list.append((wall[0], wall[1] - 1))

                    if wall[1] < size - 1 and maze[wall[0]][wall[1] + 1] != 2:
                        maze[wall[0]][wall[1] + 1] = 1
                    if wall[1] < size - 1 and (wall[0], wall[1] + 1) not in wall_list:
                        wall_list.append((wall[0], wall[1] + 1))

                    if wall[0] > 0 and maze[wall[0] - 1][wall[1]] != 2:
                        maze[wall[0] - 1][wall[1]] = 1
                    if wall[0] > 0 and (wall[0] - 1, wall[1]) not in wall_list:
                        wall_list.append((wall[0] - 1, wall[1]))
        
            elif wall[0] > 0 and wall[0] < size - 1 and maze[wall[0] + 1][wall[1]] == 0 and maze[wall[0] - 1][wall[1]] == 2:
                if checkSurround(maze, wall):
                    maze[wall[0]][wall[1]] = 2
                
                    if wall[1] > 0 and maze[wall[0]][wall[1] - 1] != 2:
                        maze[wall[0]][wall[1] - 1] = 1
                    if wall[1] > 0 and (wall[0], wall[1] - 1) not in wall_list:
                        wall_list.append((wall[0], wall[1] - 1))

                    if wall[1] < size - 1 and maze[wall[0]][wall[1] + 1] != 2:
                        maze[wall[0]][wall[1] + 1] = 1
                    if wall[1] < size - 1 and (wall[0], wall[1] + 1) not in wall_list:
                        wall_list.append((wall[0], wall[1] + 1))

                    if wall[0] < size - 1 and maze[wall[0] + 1][wall[1]] != 2:
                        maze[wall[0] + 1][wall[1]] = 1
                    if wall[0] < size - 1 and (wall[0] + 1, wall[1]) not in wall_list:
                        wall_list.append((wall[0] + 1, wall[1]))

            elif wall[1] > 0 and wall[1] < size - 1 and maze[wall[0]][wall[1] - 1] == 0 and maze[wall[0]][wall[1] + 1] == 2:
                if checkSurround(maze, wall):
                    maze[wall[0]][wall[1]] = 2
        
                    if wall[0] > 0 and maze[wall[0] - 1][wall[1]] != 2:
                        maze[wall[0] - 1][wall[1]] = 1
                    if wall[0] > 0 and (wall[0] - 1, wall[1]) not in wall_list:
                        wall_list.append((wall[0] - 1, wall[1]))

                    if wall[0] < size - 1 and maze[wall[0] + 1][wall[1]] != 2:
                        maze[wall[0] + 1][wall[1]] = 1
                    if wall[0] < size - 1 and (wall[0] + 1, wall[1]) not in wall_list:
                        wall_list.append((wall[0] + 1, wall[1]))

                    if wall[1] > 0 and maze[wall[0]][wall[1] - 1] != 2:
                        maze[wall[0]][wall[1] - 1] = 1
                    if wall[1] > 0 and (wall[0], wall[1] - 1) not in wall_list:
                        wall_list.append((wall[0], wall[1] - 1))

            elif wall[1] > 0 and wall[1] < size - 1 and maze[wall[0]][wall[1] + 1] == 0 and maze[wall[0]][wall[1] - 1] == 2:
                if checkSurround(maze, wall):
                    maze[wall[0]][wall[1]] = 2
                
                    if wall[0] > 0 and maze[wall[0] - 1][wall[1]] != 2:
                        maze[wall[0] - 1][wall[1]] = 1
                    if wall[0] > 0 and (wall[0] - 1, wall[1]) not in wall_list:
                        wall_list.append((wall[0] - 1, wall[1]))

                    if wall[0] < size - 1 and maze[wall[0] + 1][wall[1]] != 2:
                        maze[wall[0] + 1][wall[1]] = 1
                    if wall[0] < size - 1 and (wall[0] + 1, wall[1]) not in wall_list:
                        wall_list.append((wall[0] + 1, wall[1]))

                    if wall[1] < size - 1 and maze[wall[0]][wall[1] + 1] != 2:
                        maze[wall[0]][wall[1] + 1] = 1
                    if wall[1] < size - 1 and (wall[0], wall[1] + 1) not in wall_list:
                        wall_list.append((wall[0], wall[1] + 1))
            
            wall_list.remove(wall)


        maze[size - 1][size // 2] = 2
        maze[0][size // 2] = 2
        entrance = maze[1][size // 2]
        entrance = size // 2
        exit = maze[size - 2][size // 2]
        exit = size // 2
        factor = np.random.randint(0, 2)
        if factor == 0:
            factor = -1
        while (maze[1][entrance] != 2):
            maze[1][entrance] = 2
            entrance += factor

        factor = np.random.randint(0, 2)
        if factor == 0:
            factor = -1
        while (maze[size-2][exit] != 2):
            maze[size-2][exit] = 2
            exit += factor
        
        parent = solveMaze(maze, solveMode)
        path = []
        node = (size-1, size // 2)
        while (node != (0, size // 2)):
            path.append(node)
            node = parent[node]

        path.append((0, size // 2))

        maze_img = np.zeros((size, size, 3), dtype=np.uint8)
        maze_mask = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                if (i, j) in path:
                    # Crucial: Saved image doesn't include solved path (has to figure that itself)
                    if combined:
                        maze_img[i, j] = [0, 255, 0]
                    else:
                        maze_img[i,j] = [0,0,255]
                    maze_mask[i,j] = 1
                    if save_img:
                        print(colored("\"", "green"), end = " ")
                elif maze[i][j] == 2.0:
                    maze_img[i,j] = [0,0,255]
                    maze_mask[i,j] = 0
                    if save_img:
                        print(colored("\"", "blue"), end = " ")
                else:
                    maze_img[i,j] = [255,0,0]
                    maze_mask[i,j] = 0
                    if save_img:
                        print(colored("#", "red"), end = " ")

            if save_img:
                print("\n")


        if save_img:
            print("Saving img")
            plt.imshow(maze_img)
            plt.axis('off')
            plt.savefig("saved_imgs100/maze_" + str(maze_num) + ".png")
            plt.show()

            plt.imshow(maze_mask)
            plt.axis('off')
            plt.savefig("mask_imgs100/maze_mask_" + str(maze_num) + ".png")
            plt.show()
            continue

        # Only returns the maze if not saving the image (not saving means only generate 1 maze per call)
        return (maze_img, maze_mask)



if __name__ == "__main__":
    # Running the file directly means you want to save the resulting mazes as images (higher resolution results)
    generate_maze(50000, 100, 'DFS', True, True)




            
            

        





