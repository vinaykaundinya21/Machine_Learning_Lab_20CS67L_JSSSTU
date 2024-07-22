# -*- coding: utf-8 -*-
"""1a_1b.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HMirLXWnC64KW5zaXc-PDVBu3SqI_4XZ

1A : 3D Surface Plot
"""

#1a - visulazation of 3D surface plot
#3D surface Plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('ToyotaCorolla.csv')
x = dataset['Price']
y = dataset['Age']
z = dataset['Weight']

ax = plt.axes(projection = '3d')
ax.plot_trisurf(x,y,z,cmap = 'jet')
ax.set_title("3d plot")
plt.show()

"""1B : Best first search implmentation"""

#1B -> Best First search implementation
import heapq

def best_first_search(graph,heuristics,start,goal):
  priority_queue = []
  heapq.heappush(priority_queue,(heuristics[start],start))
  visited = set()

  while priority_queue:
    current_cost,current = heapq.heappop(priority_queue)

    if current in visited:
      continue

    print(f"visiting {current}")
    visited.add(current)

    if current == goal:
      print("goal reached")
      return True

    for neighbour,cost in graph[current]:
      if neighbour not in visited:
        total_cost = cost + current_cost
        heapq.heappush(priority_queue,(heuristics[neighbour],neighbour))
        #return total_cost


  print("goal not reached")
  return False
graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('A', 1), ('D', 2)],
    'C': [('A', 3), ('E', 4)],
    'D': [('B', 2)],
    'E': [('C', 4)]
} # define yourself in the same format for your graph


heuristics =  {
    'A': 3,
    'B': 5,
    'C': 2,
    'D': 4,
    'E': 0
    # define the value of heuristics in the same manner
}

total_cost = best_first_search(graph,heuristics,'A','E')
print(result)

import heapq

def best_first_search(graph, heuristics, start, goal):
    priority_queue = []
    heapq.heappush(priority_queue, (heuristics[start], start))
    visited = set()

    while priority_queue:
        current_cost, current = heapq.heappop(priority_queue)

        if current in visited:
            continue

        print(f"Visiting {current}")
        visited.add(current)

        if current == goal:
            print("Goal reached")
            return True

        for neighbor, cost in graph.get(current, []):
            if neighbor not in visited:
                total_cost = cost + current_cost
                heapq.heappush(priority_queue, (heuristics[neighbor], neighbor))

    print("Goal not reached")
    return False

#print(total_cost)

# Example graph and heuristics
graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('A', 1), ('D', 2)],
    'C': [('A', 3), ('E', 4)],
    'D': [('B', 2)],
    'E': [('C', 4)]
}

heuristics = {
    'A': 3,
    'B': 5,
    'C': 2,
    'D': 4,
    'E': 0
}

result = best_first_search(graph, heuristics, 'A', 'E')
print(result)