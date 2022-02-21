
def load_skeleton_from_obj_file(obj_file):
    vertices = []
    vertices.append([])
    
    edges = []

    f = open(obj_file)
    lines = f.readlines()
    for line in lines:
        line_data = line.split(' ')
        if line_data[0] == 'v':
            x = float(line_data[1])
            y = float(line_data[2])
            z = float(line_data[3])
            vertices.append([x, y, z])
        
        elif line_data[0] == 'l':
            v1 = int(line_data[1])
            v2 = int(line_data[2])

            edges.append([v1, v2])

    return vertices, edges


    
    


