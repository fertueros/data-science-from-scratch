#KDTree

import numpy as np
from ..metrics.pairwise import*

class Node:
    def __init__(self, point, index, axis=None, left=None, right=None):
        self.point = point # Almacena las coordenadas del punto en este nodo.
        self.index = index # Almacena el índice del punto. Esto podría ser útil para recuperar información adicional más tarde.
        self.axis = axis # Almacena el eje en el que se realizó la división para crear este nodo.
        self.left = left # Almacena el subárbol izquierdo.
        self.right = right # Almacena el subárbol derecho.

class KDTree:
    def __init__(self, leaf_size = 30):
        self.root = None
        self.leaf_size = leaf_size # cambio a fuerza bruta

    def _build_tree(self, points, indices, depth=0):
        if len(points) == 0 or len(points) <= self.leaf_size:
            return None

        # Determinar el eje de particion (es alternado)
        k = len(points[0]) # numero de dimensiones
        axis = depth % k

        # ordenar los puntos y encontrar la mediana
        sorted_indices = np.argsort(points[:, axis])
        median_idx = len(sorted_indices) // 2
        median_point = points[sorted_indices[median_idx]]

        # crear el nodo y los subarboles - retornamos esto
        
        
        return Node(
            point=median_point,
            index=median_idx,
            axis=axis,
            left=self._build_tree(points[sorted_indices[:median_idx]], indices[sorted_indices[:median_idx]], depth+1),
            right=self._build_tree(points[sorted_indices[median_idx + 1:]], indices[sorted_indices[median_idx + 1:]], depth + 1)
        )
    
    def _query(self, point, node, k, depth=0, best=None):
        if node is None:
            return
        if best is None:
            best = []

        # numero de dimensiones
        dim = len(point)

        # calcula el eje y la distancia
        axis = depth % dim
        distance = point[axis] - node.point[axis] # calcula la distancia entre el punto objetivo y el punto en el nodo actual del árbol
        
        # La distancia calculada nos ayuda a decidir qué subárbol (izquierdo o derecho) explorar primero. 
        # Si la distancia es menor o igual a cero, exploramos el subárbol izquierdo primero; de lo contrario, exploramos el subárbol derecho.

        # Recursivamemte
        next_node = None
        oppsite_node = None

        if distance < 0:
            next_node = node.left
            oppsite_node = node.right
        else:
            next_node = node.right
            oppsite_node = node.left
        
        self._query(point, next_node, k, depth+1, best)

        # añadiendo el nodo actual a la lista de los mejores
        '''
        len(bests) < k: Si aún no hemos encontrado k vecinos más cercanos, entonces el nodo actual debe ser agregado a la lista bests.

        current_distance < bests[-1][1]: 
        Si ya hemos encontrado k vecinos más cercanos (len(bests) = k), entonces comparamos la distancia del nodo actual al punto objetivo (current_distance) con la mayor distancia en la lista bests (bests[-1][1]). 
        Si current_distance es menor, entonces el nodo actual es un vecino más cercano y debe ser agregado a bests.

        Después de verificar esta condición, si se cumple, agregamos el nodo actual y su distancia a bests. Luego, ordenamos bests en función de las distancias y, si es necesario, eliminamos el nodo más lejano para mantener solo los k vecinos más cercanos.
        Este enfoque asegura que bests siempre contiene los k vecinos más cercanos al punto objetivo que hemos encontrado hasta ahora en la búsqueda del árbol KD.
        '''
        current_distance = np.linalg.norm(np.array(point) - np.array(node.point))
        if len(bests) < k or current_distance < bests[-1][1]:
            bests.append((node, current_distance))
            bests.sort(key=lambda x: x[1])
            if len(bests) > k:
                bests.pop()
        
        # chekeando si hay puntos en otro arbol cerca del punto objetivo
        if len(bests) < k or abs(distance) < bests[-1][1]:
            self._query(point, opposite_node, k, depth + 1, bests)

        return best