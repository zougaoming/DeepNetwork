class Vertex:
    def __init__(self,key):
        self.id = key
        self.connectedTo = {}

    #从这个顶点添加一个连接到另一个
    def addNeighbor(self,nbr,weight = 0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + 'connectedTo' + str([x.id for x in self.connectedTo])

    #返回邻接表中的所有的项点
    def getConnections(self):
        return  self.connectedTo

    def getId(self):
        return self.id

    #返回从这个顶点到作为参数顶点的边的权重
    def getweight(self,nbr):
        return  self.connectedTo[nbr]
class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return  newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return  self.vertList[n].getConnections()
        else:
            return  None

    def __contains__(self, n):
        return  n in self.vertList

    def addEdge(self,f,t,const = 0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not  in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t],const)

    def getVertices(self):
        return  self.vertList.keys()

    def __iter__(self):
        return  iter(self.vertList.values())


from queue import Queue
import numpy as np
def compute_gradients(target_op):
    ''' Backpropagation implementation computing gradient of target operation wrt
        all the other connected nodes.

    :param target_op: The target operation whose gradient wrt other nodes would
                      be computed.
    :type target_op: Any operation type.

    :return grad_table: A table containing node objects and gradients.
    :type grad_table: dict.
    '''
    # A dict containing a mapping between node and gradient value of target_op wrt the node's output.
    # NOTE: It is the gradient wrt the node's OUTPUT NOT input.
    grad_table = {}

    # The gradient wrt target_op itself is 1.
    grad_table[target_op] = np.ones_like(target_op.output_value)

    # Perform a breadth-first search staring from the target_op in graph.
    # Queue for node traverasl.
    queue = Queue()
    queue.put(target_op)

    # Set for visited nodes.
    visited = set()
    visited.add(target_op)

    while not queue.empty():
        node = queue.get()
        print('')
        print(node)
        # Compute gradient wrt the node's output.
        if node != target_op:
            pass

        # Put adjecent nodes to queue.
        if hasattr(node, 'input_nodes'):
            for input_node in node.input_nodes:
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table