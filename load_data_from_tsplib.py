

# Load data from TSPLib file
# return dimension, nodes
# nodes is a list like [[x,y],[x,y],...]
def loadDataFormTSPLibFile(fileName):
    """
    Load data from TSPLib file.
    """
    # Open file
    file = open(fileName, "r")

    # Read file
    fileContent = file.readlines()

    # Close file
    file.close()

    # Get dimension
    dimension = int(fileContent[3].split(":")[1])

    # Get nodes
    nodes = []
    for i in range(6, 6 + dimension):
        nodes.append(
            [float(fileContent[i].split()[1]), float(fileContent[i].split()[2])])

    # Return data
    return dimension, nodes