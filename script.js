// Algorithm summaries data
const algorithmSummaries = {
    'linear-search': {
        name: 'Linear Search',
        timeComplexity: 'O(n)',
        spaceComplexity: 'O(1)',
        description: 'Checks each element in the array sequentially from the beginning until the target element is found or the end is reached. Simple but inefficient for large datasets.',
        useCases: [
            'Small unsorted arrays',
            'When data is not frequently searched',
            'Simple implementations where code clarity matters'
        ],
        pseudoCode: `def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1`
    },
    'binary-search': {
        name: 'Binary Search',
        timeComplexity: 'O(log n)',
        spaceComplexity: 'O(1)',
        description: 'Works on sorted arrays by repeatedly dividing the search interval in half. If the target is less than the middle element, search the left half; otherwise, search the right half. Eliminates half the remaining elements at each step.',
        useCases: [
            'Searching in sorted arrays',
            'Finding boundaries in sorted data',
            'Efficient lookups in large datasets'
        ],
        pseudoCode: `def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1`
    },
    'jump-search': {
        name: 'Jump Search',
        timeComplexity: 'O(√n)',
        spaceComplexity: 'O(1)',
        description: 'A search algorithm for sorted arrays that jumps ahead by fixed steps (typically √n), then performs a linear search in the identified block. More efficient than linear search but less efficient than binary search.',
        useCases: [
            'Searching in sorted arrays when binary search is not available',
            'When jumping is faster than binary search on some systems',
            'Educational purposes to understand search variations'
        ],
        pseudoCode: `def jump_search(arr, target):
    n = len(arr)
    step = int(n ** 0.5)
    prev = 0
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(n ** 0.5)
        if prev >= n:
            return -1
    for i in range(prev, min(step, n)):
        if arr[i] == target:
            return i
    return -1`
    },
    'interpolation-search': {
        name: 'Interpolation Search',
        timeComplexity: 'O(log log n) avg, O(n) worst',
        spaceComplexity: 'O(1)',
        description: 'An improved variant of binary search for uniformly distributed sorted arrays. Instead of always checking the middle, it estimates the position based on the value distribution, leading to better average performance.',
        useCases: [
            'Uniformly distributed sorted arrays',
            'When data distribution is known',
            'Telephone directory searches'
        ],
        pseudoCode: `def interpolation_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right and arr[left] <= target <= arr[right]:
        pos = left + ((target - arr[left]) * (right - left)) // (arr[right] - arr[left])
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1
    return -1`
    },
    'bubble-sort': {
        name: 'Bubble Sort',
        timeComplexity: 'O(n²)',
        spaceComplexity: 'O(1)',
        description: 'Repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order. The pass through the list is repeated until no swaps are needed.',
        useCases: [
            'Educational purposes',
            'Small datasets',
            'When simplicity is more important than efficiency'
        ],
        pseudoCode: `def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break`
    },
    'merge-sort': {
        name: 'Merge Sort',
        timeComplexity: 'O(n log n)',
        spaceComplexity: 'O(n)',
        description: 'Uses divide-and-conquer approach. Divides the array into two halves, recursively sorts them, and then merges the two sorted halves. Stable and guarantees O(n log n) performance.',
        useCases: [
            'Large datasets requiring stable sort',
            'External sorting (files too large for memory)',
            'When worst-case performance matters'
        ],
        pseudoCode: `def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result`
    },
    'quick-sort': {
        name: 'Quick Sort',
        timeComplexity: 'O(n log n) avg, O(n²) worst',
        spaceComplexity: 'O(log n)',
        description: 'Picks a pivot element and partitions the array around it. Elements smaller than the pivot go to the left, larger to the right. This process is repeated recursively for the sub-arrays.',
        useCases: [
            'General-purpose sorting',
            'In-place sorting with good average performance',
            'When average-case performance is more important than worst-case'
        ],
        pseudoCode: `def quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1`
    },
    'insertion-sort': {
        name: 'Insertion Sort',
        timeComplexity: 'O(n) best, O(n²) worst',
        spaceComplexity: 'O(1)',
        description: 'Builds the sorted array one element at a time. For each element, it finds the correct position in the already sorted portion and inserts it there.',
        useCases: [
            'Small datasets',
            'Nearly sorted arrays',
            'When simplicity and low overhead matter'
        ],
        pseudoCode: `def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key`
    },
    'selection-sort': {
        name: 'Selection Sort',
        timeComplexity: 'O(n²)',
        spaceComplexity: 'O(1)',
        description: 'Finds the minimum element from the unsorted portion and swaps it with the element at the current position. Repeats this process for the remaining unsorted portion.',
        useCases: [
            'Small datasets',
            'When memory writes are expensive',
            'Educational purposes'
        ],
        pseudoCode: `def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]`
    },
    'heap-sort': {
        name: 'Heap Sort',
        timeComplexity: 'O(n log n)',
        spaceComplexity: 'O(1)',
        description: 'Builds a max heap from the array, then repeatedly extracts the maximum element and places it at the end. Maintains heap property throughout the process.',
        useCases: [
            'When O(n log n) worst-case is required',
            'In-place sorting with guaranteed performance',
            'Priority queue implementations'
        ],
        pseudoCode: `def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

def heapify(arr, n, i):
    largest = i
    left, right = 2 * i + 1, 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)`
    },
    'radix-sort': {
        name: 'Radix Sort',
        timeComplexity: 'O(nk)',
        spaceComplexity: 'O(n+k)',
        description: 'Sorts numbers by processing individual digits. Starts from the least significant digit and moves to the most significant. k represents the number of digits.',
        useCases: [
            'Sorting integers with fixed number of digits',
            'When k is small compared to n',
            'Non-comparison based sorting'
        ],
        pseudoCode: `def radix_sort(arr):
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10

def counting_sort_by_digit(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    for i in range(n):
        count[(arr[i] // exp) % 10] += 1
    for i in range(1, 10):
        count[i] += count[i - 1]
    for i in range(n - 1, -1, -1):
        idx = (arr[i] // exp) % 10
        output[count[idx] - 1] = arr[i]
        count[idx] -= 1
    arr[:] = output`
    },
    'counting-sort': {
        name: 'Counting Sort',
        timeComplexity: 'O(n+k)',
        spaceComplexity: 'O(k)',
        description: 'Counts the number of occurrences of each value, then uses this information to place elements in their correct positions. k represents the range of values.',
        useCases: [
            'Sorting integers in a small range',
            'When k is small compared to n',
            'As a subroutine in radix sort'
        ],
        pseudoCode: `def counting_sort(arr):
    max_val = max(arr)
    count = [0] * (max_val + 1)
    for num in arr:
        count[num] += 1
    idx = 0
    for i in range(len(count)):
        while count[i] > 0:
            arr[idx] = i
            idx += 1
            count[i] -= 1`
    },
    'shell-sort': {
        name: 'Shell Sort',
        timeComplexity: 'O(n log² n)',
        spaceComplexity: 'O(1)',
        description: 'A generalization of insertion sort that sorts elements far apart first, then progressively reduces the gap. Uses a sequence of gaps to improve performance over insertion sort.',
        useCases: [
            'Medium-sized arrays',
            'When in-place sorting is required',
            'Embedded systems with memory constraints'
        ],
        pseudoCode: `def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2`
    },
    'bucket-sort': {
        name: 'Bucket Sort',
        timeComplexity: 'O(n+k) avg, O(n²) worst',
        spaceComplexity: 'O(n+k)',
        description: 'Distributes elements into a number of buckets, then sorts each bucket individually (often using insertion sort), and finally concatenates the buckets. Works best when input is uniformly distributed.',
        useCases: [
            'Uniformly distributed floating-point numbers',
            'When data can be partitioned into buckets',
            'External sorting algorithms'
        ],
        pseudoCode: `def bucket_sort(arr):
    n = len(arr)
    buckets = [[] for _ in range(n)]
    for num in arr:
        idx = int(n * num)
        buckets[idx].append(num)
    for bucket in buckets:
        insertion_sort(bucket)
    return [num for bucket in buckets for num in bucket]`
    },
    'dfs': {
        name: 'Depth-First Search (DFS)',
        timeComplexity: 'O(V+E)',
        spaceComplexity: 'O(V)',
        description: 'Explores as far as possible along each branch before backtracking. Uses a stack (or recursion) to keep track of vertices to visit. Useful for finding paths, detecting cycles, and topological sorting.',
        useCases: [
            'Finding paths between nodes',
            'Detecting cycles in graphs',
            'Topological sorting',
            'Solving mazes and puzzles'
        ],
        pseudoCode: `def dfs(graph, start, visited):
    visited.add(start)
    print(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)`
    },
    'bfs': {
        name: 'Breadth-First Search (BFS)',
        timeComplexity: 'O(V+E)',
        spaceComplexity: 'O(V)',
        description: 'Explores all nodes at the current depth level before moving to nodes at the next depth level. Uses a queue to keep track of vertices to visit. Guarantees finding the shortest path in unweighted graphs.',
        useCases: [
            'Finding shortest path in unweighted graphs',
            'Level-order tree traversal',
            'Social network analysis',
            'Web crawling'
        ],
        pseudoCode: `from collections import deque

def bfs(graph, start):
    queue = deque([start])
    visited = {start}
    while queue:
        node = queue.popleft()
        print(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)`
    },
    'dijkstra': {
        name: "Dijkstra's Algorithm",
        timeComplexity: 'O((V+E) log V)',
        spaceComplexity: 'O(V)',
        description: 'Finds the shortest path from a source vertex to all other vertices in a weighted graph. Uses a priority queue to always process the vertex with the smallest known distance.',
        useCases: [
            'GPS navigation systems',
            'Network routing protocols',
            'Finding shortest paths in weighted graphs',
            'Social network analysis'
        ],
        pseudoCode: `import heapq

def dijkstra(graph, start):
    dist = {v: float('inf') for v in graph}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist`
    },
    'topological-sort': {
        name: 'Topological Sort',
        timeComplexity: 'O(V+E)',
        spaceComplexity: 'O(V)',
        description: 'Produces a linear ordering of vertices in a directed acyclic graph (DAG) such that for every directed edge (u, v), vertex u comes before v in the ordering.',
        useCases: [
            'Task scheduling',
            'Build systems (dependency resolution)',
            'Course prerequisites',
            'Event ordering'
        ],
        pseudoCode: `def topological_sort(graph):
    in_degree = {v: 0 for v in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1
    queue = [v for v in graph if in_degree[v] == 0]
    result = []
    while queue:
        u = queue.pop(0)
        result.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return result`
    },
    'bellman-ford': {
        name: "Bellman-Ford Algorithm",
        timeComplexity: 'O(VE)',
        spaceComplexity: 'O(V)',
        description: 'Finds shortest paths from a source vertex to all other vertices in a weighted graph. Unlike Dijkstra\'s, it can handle negative edge weights and detect negative cycles. Uses relaxation over V-1 iterations.',
        useCases: [
            'Graphs with negative edge weights',
            'Detecting negative cycles',
            'Distance-vector routing protocols',
            'Arbitrage detection in currency exchange'
        ],
        pseudoCode: `def bellman_ford(graph, start):
    dist = {v: float('inf') for v in graph}
    dist[start] = 0
    for _ in range(len(graph) - 1):
        for u in graph:
            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
    for u in graph:
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                return None  # Negative cycle
    return dist`
    },
    'floyd-warshall': {
        name: 'Floyd-Warshall Algorithm',
        timeComplexity: 'O(V³)',
        spaceComplexity: 'O(V²)',
        description: 'Finds shortest paths between all pairs of vertices in a weighted graph using dynamic programming. Works with negative edges (but not negative cycles). Uses a 3D table to store intermediate results.',
        useCases: [
            'All-pairs shortest path problems',
            'Transitive closure of graphs',
            'Network routing tables',
            'Social network analysis'
        ],
        pseudoCode: `def floyd_warshall(graph):
    n = len(graph)
    dist = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = w
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist`
    },
    'kruskal': {
        name: "Kruskal's Algorithm",
        timeComplexity: 'O(E log E)',
        spaceComplexity: 'O(V)',
        description: 'Finds a minimum spanning tree (MST) by sorting all edges by weight and adding them one by one if they don\'t form a cycle. Uses union-find data structure to efficiently check for cycles.',
        useCases: [
            'Finding minimum spanning trees',
            'Network design',
            'Clustering algorithms',
            'Approximation algorithms for TSP'
        ],
        pseudoCode: `def kruskal(edges, n):
    edges.sort(key=lambda x: x[2])
    parent = list(range(n))
    mst = []
    for u, v, w in edges:
        if find(parent, u) != find(parent, v):
            mst.append((u, v, w))
            union(parent, u, v)
    return mst

def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, x, y):
    parent[find(parent, x)] = find(parent, y)`
    },
    'prim': {
        name: "Prim's Algorithm",
        timeComplexity: 'O(E log V)',
        spaceComplexity: 'O(V)',
        description: 'Finds a minimum spanning tree (MST) by starting from an arbitrary vertex and repeatedly adding the minimum-weight edge that connects a vertex in the MST to a vertex outside it. Uses a priority queue.',
        useCases: [
            'Finding minimum spanning trees',
            'Network design',
            'Clustering problems',
            'When graph is dense'
        ],
        pseudoCode: `import heapq

def prim(graph, start):
    mst = []
    visited = {start}
    pq = [(w, start, v) for v, w in graph[start]]
    heapq.heapify(pq)
    while pq and len(visited) < len(graph):
        w, u, v = heapq.heappop(pq)
        if v not in visited:
            visited.add(v)
            mst.append((u, v, w))
            for neighbor, weight in graph[v]:
                if neighbor not in visited:
                    heapq.heappush(pq, (weight, v, neighbor))
    return mst`
    },
    'bst-operations': {
        name: 'BST Operations',
        timeComplexity: 'O(log n) avg, O(n) worst',
        spaceComplexity: 'O(log n) avg, O(n) worst',
        description: 'Binary Search Tree operations (search, insert, delete) work by comparing the target value with the root. If smaller, go left; if larger, go right. Continues recursively until the target is found or a null node is reached.',
        useCases: [
            'Dynamic data structures',
            'Database indexing',
            'Symbol tables',
            'Priority queues'
        ],
        pseudoCode: `def bst_search(root, key):
    if root is None or root.val == key:
        return root
    if root.val < key:
        return bst_search(root.right, key)
    return bst_search(root.left, key)

def bst_insert(root, key):
    if root is None:
        return Node(key)
    if key < root.val:
        root.left = bst_insert(root.left, key)
    else:
        root.right = bst_insert(root.right, key)
    return root`
    },
    'tree-traversal': {
        name: 'Tree Traversal',
        timeComplexity: 'O(n)',
        spaceComplexity: 'O(h)',
        description: 'Visits all nodes in a tree in a specific order. Common orders: inorder (left-root-right), preorder (root-left-right), and postorder (left-right-root). h represents the height of the tree.',
        useCases: [
            'Expression tree evaluation',
            'File system traversal',
            'Tree serialization',
            'Binary tree operations'
        ],
        pseudoCode: `def inorder(root):
    if root:
        inorder(root.left)
        print(root.val)
        inorder(root.right)

def preorder(root):
    if root:
        print(root.val)
        preorder(root.left)
        preorder(root.right)

def postorder(root):
    if root:
        postorder(root.left)
        postorder(root.right)
        print(root.val)`
    },
    'trie-operations': {
        name: 'Trie Operations',
        timeComplexity: 'O(m)',
        spaceComplexity: 'O(ALPHABET_SIZE × m × N)',
        description: 'A tree-like data structure for storing strings. Search, insert, and delete operations take O(m) time where m is the length of the key. Each node represents a character, and paths from root to leaf represent strings.',
        useCases: [
            'Autocomplete systems',
            'Spell checkers',
            'IP routing (longest prefix matching)',
            'Search engines and text indexing'
        ],
        pseudoCode: `class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

def trie_insert(root, word):
    node = root
    for char in word:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
    node.is_end = True

def trie_search(root, word):
    node = root
    for char in word:
        if char not in node.children:
            return False
        node = node.children[char]
    return node.is_end`
    },
    'hash-table': {
        name: 'Hash Table Lookup',
        timeComplexity: 'O(1) avg, O(n) worst',
        spaceComplexity: 'O(n)',
        description: 'Uses a hash function to map keys to array indices, allowing for average O(1) lookup time. Collisions are handled using techniques like chaining or open addressing.',
        useCases: [
            'Fast key-value lookups',
            'Caching',
            'Database indexing',
            'Counting frequencies'
        ],
        pseudoCode: `class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def hash(self, key):
        return hash(key) % self.size
    
    def insert(self, key, value):
        idx = self.hash(key)
        self.table[idx].append((key, value))
    
    def get(self, key):
        idx = self.hash(key)
        for k, v in self.table[idx]:
            if k == key:
                return v
        return None`
    },
    'binary-heap-insert-remove': {
        name: 'Binary Heap Insert/Remove',
        timeComplexity: 'O(log n)',
        spaceComplexity: 'O(1)',
        description: 'Maintains heap property by bubbling up (insert) or bubbling down (remove). Insert adds to the end and bubbles up; remove extracts root and bubbles down the last element.',
        useCases: [
            'Priority queues',
            'Heap sort',
            'Graph algorithms (Dijkstra, Prim)',
            'Event scheduling'
        ],
        pseudoCode: `def heap_insert(heap, val):
    heap.append(val)
    i = len(heap) - 1
    while i > 0 and heap[(i - 1) // 2] < heap[i]:
        heap[i], heap[(i - 1) // 2] = heap[(i - 1) // 2], heap[i]
        i = (i - 1) // 2

def heap_extract_max(heap):
    if not heap:
        return None
    max_val = heap[0]
    heap[0] = heap[-1]
    heap.pop()
    heapify_down(heap, 0)
    return max_val`
    },
    'binary-heap-build': {
        name: 'Binary Heap Build',
        timeComplexity: 'O(n)',
        spaceComplexity: 'O(1)',
        description: 'Builds a heap from an unsorted array efficiently by starting from the last non-leaf node and heapifying downward. More efficient than inserting elements one by one.',
        useCases: [
            'Initializing priority queues',
            'Heap sort initialization',
            'Batch heap construction'
        ],
        pseudoCode: `def build_heap(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify_down(arr, i)

def heapify_down(arr, i):
    n = len(arr)
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify_down(arr, largest)`
    },
    'kmp': {
        name: 'KMP Algorithm',
        timeComplexity: 'O(n+m)',
        spaceComplexity: 'O(m)',
        description: 'Knuth-Morris-Pratt algorithm for pattern matching in strings. Preprocesses the pattern to create a failure function (LPS array) that allows skipping characters when a mismatch occurs, avoiding unnecessary comparisons.',
        useCases: [
            'String searching and pattern matching',
            'Text editors and search functions',
            'DNA sequence matching',
            'Plagiarism detection'
        ],
        pseudoCode: `def kmp_search(text, pattern):
    lps = build_lps(pattern)
    i = j = 0
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
        if j == len(pattern):
            return i - j
        elif i < len(text) and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1

def build_lps(pattern):
    lps = [0] * len(pattern)
    length = 0
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps`
    },
    'rabin-karp': {
        name: 'Rabin-Karp Algorithm',
        timeComplexity: 'O(n+m) avg, O(nm) worst',
        spaceComplexity: 'O(1)',
        description: 'Pattern matching algorithm using rolling hash. Computes hash values for the pattern and all substrings of the text. When hashes match, verifies with actual string comparison. Average case is excellent, but worst case degrades.',
        useCases: [
            'Multiple pattern searching',
            'Plagiarism detection',
            'Finding duplicate substrings',
            'When average performance matters more than worst case'
        ],
        pseudoCode: `def rabin_karp(text, pattern):
    n, m = len(text), len(pattern)
    if m > n:
        return -1
    base = 256
    mod = 101
    pattern_hash = 0
    text_hash = 0
    h = 1
    for i in range(m - 1):
        h = (h * base) % mod
    for i in range(m):
        pattern_hash = (base * pattern_hash + ord(pattern[i])) % mod
        text_hash = (base * text_hash + ord(text[i])) % mod
    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            if text[i:i+m] == pattern:
                return i
        if i < n - m:
            text_hash = (base * (text_hash - ord(text[i]) * h) + ord(text[i+m])) % mod
    return -1`
    },
    'lcs': {
        name: 'Longest Common Subsequence',
        timeComplexity: 'O(nm)',
        spaceComplexity: 'O(nm)',
        description: 'Finds the longest subsequence (not necessarily contiguous) common to two sequences using dynamic programming. Builds a 2D table where each cell represents the LCS length of prefixes.',
        useCases: [
            'DNA sequence comparison',
            'Version control systems (diff algorithms)',
            'Text similarity analysis',
            'Bioinformatics'
        ],
        pseudoCode: `def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]`
    },
    'edit-distance': {
        name: 'Edit Distance (Levenshtein)',
        timeComplexity: 'O(nm)',
        spaceComplexity: 'O(min(n,m))',
        description: 'Calculates the minimum number of single-character edits (insertions, deletions, substitutions) needed to transform one string into another. Uses dynamic programming with space optimization.',
        useCases: [
            'Spell checkers and autocorrect',
            'DNA sequence alignment',
            'Fuzzy string matching',
            'Natural language processing'
        ],
        pseudoCode: `def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    # Ensure s1 is shorter to minimize space
    if m > n:
        s1, s2 = s2, s1
        m, n = n, m
    # Use only two rows for space optimization
    prev = [j for j in range(n + 1)]
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev, curr = curr, prev
    return prev[n]`
    }
};

// Modal functions
function openDescriptionModal(algorithmKey) {
    const algorithm = algorithmSummaries[algorithmKey];
    if (!algorithm) return;
    
    const modal = document.getElementById('algorithmModal');
    const modalContent = document.getElementById('modalContent');
    
    let useCasesHtml = '';
    algorithm.useCases.forEach(useCase => {
        useCasesHtml += `<li class="text-gray-300">${useCase}</li>`;
    });
    
    modalContent.innerHTML = `
        <h2 class="text-3xl font-bold mb-6 text-cyan-400">${algorithm.name}</h2>
        <div class="space-y-4 mb-6">
            <div class="flex items-center gap-4">
                <span class="text-gray-400">Time Complexity:</span>
                <span class="text-cyan-400 font-semibold text-lg">${algorithm.timeComplexity}</span>
            </div>
            <div class="flex items-center gap-4">
                <span class="text-gray-400">Space Complexity:</span>
                <span class="text-emerald-400 font-semibold text-lg">${algorithm.spaceComplexity}</span>
            </div>
        </div>
        <div class="mb-6">
            <h3 class="text-xl font-semibold mb-3 text-cyan-300">How it works:</h3>
            <p class="text-gray-300 leading-relaxed">${algorithm.description}</p>
        </div>
        <div>
            <h3 class="text-xl font-semibold mb-3 text-cyan-300">Use cases:</h3>
            <ul class="list-disc list-inside space-y-2">
                ${useCasesHtml}
            </ul>
        </div>
    `;
    
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function openPseudoCodeModal(algorithmKey) {
    const algorithm = algorithmSummaries[algorithmKey];
    if (!algorithm || !algorithm.pseudoCode) return;
    
    const modal = document.getElementById('algorithmModal');
    const modalContent = document.getElementById('modalContent');
    
    // Escape HTML in pseudo code
    const escapedCode = algorithm.pseudoCode
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    
    modalContent.innerHTML = `
        <h2 class="text-3xl font-bold mb-6 text-cyan-400">${algorithm.name} - Pseudo Code</h2>
        <div class="mb-4">
            <span class="text-gray-400 text-sm">Python</span>
        </div>
        <pre class="bg-[#0a0a15] border border-cyan-500/30 rounded-lg p-4 overflow-x-auto"><code class="text-cyan-300 text-sm font-mono whitespace-pre">${escapedCode}</code></pre>
    `;
    
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    const modal = document.getElementById('algorithmModal');
    modal.classList.remove('active');
    document.body.style.overflow = '';
}

// Chart.js configuration and complexity visualizer

let complexityChart = null;
const maxN = 1000;

// Complexity calculation functions
const calculateO1 = (n) => 1;
const calculateLogN = (n) => Math.log2(n || 1);
const calculateN = (n) => n;
const calculateNLogN = (n) => n * Math.log2(n || 1);
const calculateN2 = (n) => n * n;

// Complexity metadata
const complexities = {
    'o1': {
        name: 'O(1)',
        calculate: calculateO1,
        color: 'rgb(34, 197, 94)', // green
        description: 'Constant time - operations take the same time regardless of input size. Perfect for hash table lookups, array indexing, and simple arithmetic.'
    },
    'logn': {
        name: 'O(log n)',
        calculate: calculateLogN,
        color: 'rgb(6, 182, 212)', // cyan
        description: 'Logarithmic time - each step eliminates half the problem space. Scales beautifully: for 1M elements, only ~20 operations needed. Found in binary search, balanced BST operations, and divide-and-conquer algorithms.'
    },
    'n': {
        name: 'O(n)',
        calculate: calculateN,
        color: 'rgb(251, 191, 36)', // yellow
        description: 'Linear time - time grows proportionally with input size. One pass through the data. Common in linear search, array traversal, and single-loop algorithms.'
    },
    'nlogn': {
        name: 'O(n log n)',
        calculate: calculateNLogN,
        color: 'rgb(59, 130, 246)', // blue
        description: 'Linearithmic time - combines linear and logarithmic. Often the best possible for comparison-based sorting. Merge sort and heap sort achieve this optimal bound.'
    },
    'n2': {
        name: 'O(n²)',
        calculate: calculateN2,
        color: 'rgb(248, 113, 113)', // red
        description: 'Quadratic time - nested loops over the input. Time grows quadratically with input size. Bubble sort, insertion sort, and naive matrix multiplication exhibit this complexity.'
    }
};

// Generate data points for chart
function generateDataPoints(complexityKey, maxN) {
    const complexity = complexities[complexityKey];
    const dataPoints = [];
    const step = Math.max(1, Math.floor(maxN / 100)); // ~100 data points
    
    for (let n = 1; n <= maxN; n += step) {
        dataPoints.push({
            x: n,
            y: complexity.calculate(n)
        });
    }
    
    return dataPoints;
}

// Initialize Chart.js
function initializeChart() {
    const ctx = document.getElementById('complexityChart');
    if (!ctx) return;
    
    const datasets = [];
    const selectedComplexities = getSelectedComplexities();
    
    selectedComplexities.forEach(key => {
        const complexity = complexities[key];
        const dataPoints = generateDataPoints(key, maxN);
        
        datasets.push({
            label: complexity.name,
            data: dataPoints,
            borderColor: complexity.color,
            backgroundColor: complexity.color + '20',
            borderWidth: 2,
            fill: false,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 4
        });
    });
    
    complexityChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#e5e7eb',
                        font: {
                            family: "'JetBrains Mono', monospace",
                            size: 12
                        },
                        usePointStyle: true,
                        padding: 15
                    }
                },
                tooltip: {
                    backgroundColor: '#1e1e1e',
                    titleColor: '#e5e7eb',
                    bodyColor: '#e5e7eb',
                    borderColor: '#374151',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'Input Size (n)',
                        color: '#9ca3af',
                        font: {
                            family: "'JetBrains Mono', monospace",
                            size: 14
                        }
                    },
                    ticks: {
                        color: '#6b7280',
                        font: {
                            family: "'JetBrains Mono', monospace"
                        }
                    },
                    grid: {
                        color: '#374151'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Operations / Time Complexity',
                        color: '#9ca3af',
                        font: {
                            family: "'JetBrains Mono', monospace",
                            size: 14
                        }
                    },
                    ticks: {
                        color: '#6b7280',
                        font: {
                            family: "'JetBrains Mono', monospace"
                        }
                    },
                    grid: {
                        color: '#374151'
                    }
                }
            },
            animation: {
                duration: 300
            }
        }
    });
}

// Get selected complexities from checkboxes
function getSelectedComplexities() {
    const checkboxes = {
        'o1': document.getElementById('complexity-o1'),
        'logn': document.getElementById('complexity-logn'),
        'n': document.getElementById('complexity-n'),
        'nlogn': document.getElementById('complexity-nlogn'),
        'n2': document.getElementById('complexity-n2')
    };
    
    return Object.keys(checkboxes).filter(key => checkboxes[key]?.checked);
}

// Update chart based on selected complexities
function updateChart() {
    if (!complexityChart) return;
    
    const selectedComplexities = getSelectedComplexities();
    
    // Update datasets
    complexityChart.data.datasets = selectedComplexities.map(key => {
        const complexity = complexities[key];
        const dataPoints = generateDataPoints(key, maxN);
        
        return {
            label: complexity.name,
            data: dataPoints,
            borderColor: complexity.color,
            backgroundColor: complexity.color + '20',
            borderWidth: 2,
            fill: false,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 4
        };
    });
    
    complexityChart.update('active');
    updateExplanation();
}

// Update explanation box based on selected complexities
function updateExplanation() {
    const explanationBox = document.getElementById('explanation-content');
    if (!explanationBox) return;
    
    const selectedComplexities = getSelectedComplexities();
    
    if (selectedComplexities.length === 0) {
        explanationBox.innerHTML = '<p class="text-gray-300">Select at least one complexity to see explanations.</p>';
        return;
    }
    
    let html = '';
    
    // Special emphasis if O(log n) is selected
    const hasLogN = selectedComplexities.includes('logn');
    if (hasLogN && selectedComplexities.length > 1) {
        html += '<div class="mb-6 p-4 bg-gradient-to-br from-cyan-400/10 to-blue-400/10 border border-cyan-400/40 rounded-lg">';
        html += '<p class="text-cyan-300 font-semibold mb-2">🌟 Why O(log n) Stands Out:</p>';
        html += '<p class="text-gray-200">O(log n) represents the gold standard for search and tree operations. ';
        html += 'Notice how it grows much slower than O(n) and dramatically slower than O(n²). ';
        html += 'This is why binary search is preferred over linear search, and why balanced trees are fundamental to efficient data structures.</p>';
        html += '</div>';
    }
    
    // Add explanations for each selected complexity
    selectedComplexities.forEach(key => {
        const complexity = complexities[key];
        html += `<div class="mb-4">`;
        html += `<p class="mb-2"><span class="font-semibold text-cyan-400">${complexity.name}:</span> <span class="text-gray-200">${complexity.description}</span></p>`;
        html += `</div>`;
    });
    
    // Add comparison note if multiple selected
    if (selectedComplexities.length > 1) {
        html += '<div class="mt-6 p-4 bg-gradient-to-br from-[#1e293b]/80 to-[#0f172a]/80 rounded-lg border border-cyan-400/20">';
        html += '<p class="text-gray-200 text-sm">💡 <strong class="text-cyan-400">Tip:</strong> Compare how these complexities scale. ';
        html += 'As n grows, the difference between O(log n) and O(n²) becomes enormous. ';
        html += 'This visualization helps build intuition for why algorithm choice matters at scale.</p>';
        html += '</div>';
    }
    
    explanationBox.innerHTML = html;
}

// Update chart based on slider value (show current point)
function updateChartForN(n) {
    if (!complexityChart) return;
    
    // Update the chart to highlight the current n value
    // We can add a vertical line or update tooltips, but for simplicity,
    // we'll just ensure the chart shows the full range
    complexityChart.update('none');
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Initialize chart
    initializeChart();
    updateExplanation();
    
    // Checkbox event listeners
    const checkboxes = ['complexity-o1', 'complexity-logn', 'complexity-n', 'complexity-nlogn', 'complexity-n2'];
    checkboxes.forEach(id => {
        const checkbox = document.getElementById(id);
        if (checkbox) {
            checkbox.addEventListener('change', updateChart);
        }
    });
    
    // Set default n value display
    const nValue = document.getElementById('n-value');
    if (nValue) {
        nValue.textContent = '1000';
    }
    
    // Algorithm card click listeners removed - now using buttons
    
    // Close modal when clicking outside
    const modal = document.getElementById('algorithmModal');
    if (modal) {
        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                closeModal();
            }
        });
    }
    
    // Close modal on Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeModal();
        }
    });
});

// Toggle section visibility for showing more algorithms
function toggleSection(sectionId) {
    const grid = document.getElementById(sectionId + '-grid');
    const button = document.getElementById(sectionId + '-btn');
    
    if (!grid || !button) return;
    
    const hiddenAlgorithms = grid.querySelectorAll('.algorithm-hidden');
    if (hiddenAlgorithms.length === 0) return;
    
    // Check if any hidden algorithm is currently visible
    const firstHidden = hiddenAlgorithms[0];
    const isCurrentlyVisible = window.getComputedStyle(firstHidden).display !== 'none';
    
    if (isCurrentlyVisible) {
        // Collapse: hide all hidden algorithms
        hiddenAlgorithms.forEach(alg => {
            alg.style.display = 'none';
        });
        button.textContent = 'Show More';
    } else {
        // Expand: show all hidden algorithms
        hiddenAlgorithms.forEach(alg => {
            alg.style.display = 'block';
        });
        button.textContent = 'Show Less';
    }
}

