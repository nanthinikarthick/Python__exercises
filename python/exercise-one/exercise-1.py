from typing import List

def id_to_fruit(fruit_id: int, fruits: List[str]) -> str:
    """
    This method returns the fruit name by getting the string at a specific index of the list.

    :param fruit_id: The id of the fruit to get
    :param fruits: The list of fruits to choose the id from
    :return: The string corresponding to the index ``fruit_id``

    This example demonstrates the issue:
    name1, name3, and name4 are expected to correspond to the strings at the indices 1, 3, and 4:
    'orange', 'kiwi', and 'strawberry'.
    """
    if fruit_id < len(fruits):
        return fruits[fruit_id]
    else:
        raise RuntimeError(f"Fruit with id {fruit_id} does not exist")


fruits_list = ["apple", "orange", "melon", "kiwi", "strawberry"]


name1 = id_to_fruit(1, fruits_list)
name3 = id_to_fruit(3, fruits_list)
name4 = id_to_fruit(4, fruits_list)

print(name1)  
print(name3)  
print(name4)  




# I see, you want to ensure that the output appears in the order 1, 3, and 4 consistently. Since sets in Python are unordered collections, the order of elements within the set cannot be guaranteed. However, we can use a different approach to ensure that the fruits are returned in the specified order.

# One way to achieve this is by using a list instead of a set, as lists maintain the order of elements. 


# In this code, we use a list fruits instead of a set. Since lists maintain the order of elements, the function will consistently return the fruits 'orange', 'kiwi', and 'strawberry' for the indices 1, 3, and 4 respectively.
