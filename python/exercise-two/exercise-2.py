import numpy as np

def swap(coords):
    swapped_coords = coords.copy()  # Make a copy to avoid modifying the original array
    # Swap x1 and y1 with x2 and y2
    swapped_coords[:, [0, 1, 2, 3]] = swapped_coords[:, [2, 3, 0, 1]]
    return swapped_coords

coords = np.array([[10, 5, 15, 6, 0],
                   [11, 3, 13, 6, 0],
                   [5, 3, 13, 6, 1],
                   [4, 4, 13, 6, 1],
                   [6, 5, 13, 16, 1]])

swapped_coords = swap(coords)
print(swapped_coords)


# After fixing the obvious error of referencing the swap function without defining it, another issue arises. The current implementation swaps the first and second elements with the third and fourth elements, but it does not correctly flip the x and y coordinates.

# To properly flip the x and y coordinates, we need to swap the first and second elements (x1 and y1) with the third and fourth elements (x2 and y2) while keeping the class ID intact.

