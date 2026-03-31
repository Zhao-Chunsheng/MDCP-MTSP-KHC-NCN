# merge the tours of sub_tsps into a big tour



import numpy as np


# calculate the euclidean distance of the two points.
def distance(point1,point2):
    return np.linalg.norm(point1-point2)


def tour_merging(clusters, strategy="farestInsertion"):
    #if clusters has no more than 3 cluster, then do not sort.
    sorted_clusters=[]

    if len(clusters) <= 3:
        sorted_clusters=clusters
    else:
        cluster_centers=[]
        for c in clusters:
            cluster_centers.append(c["center"])
        merge_index=farthest_insertion_2d_tsp(cluster_centers)
        for i in range(len(clusters)):
            sorted_clusters.append(clusters[merge_index[i]])
    
    while len(sorted_clusters) > 1:
        # merge the first two clusters.
        new_cluster=merge_two_cluster_tours(sorted_clusters[0],sorted_clusters[1])

        # remove the first two clusters
        sorted_clusters.pop(0)
        sorted_clusters.pop(0)
        # insert the new cluster at index 0
        sorted_clusters.insert(0,new_cluster)

    return sorted_clusters[0]["sorted_tsp_data"]



def merge_sub_tsps(clusters, strategy="farestInsertion"):
    strategy=strategy.lower()
    if strategy=="farestinsertion" or strategy=="farest_insertion":
        # Tour merging procedure
        merge_index=merge_sub_tsps_by_farthest_insertion(clusters)
        # merge_index indicates the visiting order of the cluseters.
    
    sorted_clusters=[]
    for i in range(len(clusters)):
        sorted_clusters.append(clusters[merge_index[i]])
        
    print("merge_index:",merge_index)

    while len(sorted_clusters) > 1:
        # merge the first two clusters.
        new_cluster=merge_two_cluster_tours(sorted_clusters[0],sorted_clusters[1])
        # remove the first two clusters
        sorted_clusters.pop(0)
        sorted_clusters.pop(0)
        # insert the new cluster at index 0
        sorted_clusters.insert(0,new_cluster)

    return sorted_clusters[0]["sorted_tsp_data"]



def find_nearest_inter_tour_nodes(tour1,tour2):
    distances=np.zeros((len(tour1),len(tour2)))
    for i in range(len(tour1)):
        for j in range(len(tour2)):
            distances[i,j]=distance(tour1[i],tour2[j])
    
    min_distance=float("inf")
    tour1_conn_point_index=-1
    tour2_conn_point_index=-1
    for i in range(len(tour1)):
        for j in range(len(tour2)):
            if distances[i,j] < min_distance:
                min_distance=distances[i,j]
                tour1_conn_point_index=i
                tour2_conn_point_index=j
    return tour1_conn_point_index,tour2_conn_point_index


def calculate_four_costs(tour1_conn_point_index,tour2_conn_point_index,tour1_points,tour2_points, types):
    if tour1_conn_point_index==0:
        tour1_conn_point_pre_index=len(tour1_points)-1
        tour1_conn_point_next_index=1
    else:
        tour1_conn_point_pre_index=tour1_conn_point_index-1
        tour1_conn_point_next_index=(tour1_conn_point_index+1)%len(tour1_points)

    if tour2_conn_point_index==0:
        tour2_conn_point_pre_index=len(tour2_points)-1
        tour2_conn_point_next_index=1
    else:
        tour2_conn_point_pre_index=tour2_conn_point_index-1
        tour2_conn_point_next_index=(tour2_conn_point_index+1)%len(tour2_points)
    
    conn_points_distance=distance(tour1_points[tour1_conn_point_index],tour2_points[tour2_conn_point_index])
    # p1--p distance; prev--p distance
    tour1_pre_to_tour1_conn_point_dist=distance(tour1_points[tour1_conn_point_pre_index],tour1_points[tour1_conn_point_index])
    # q1--q distance; prev--q distance
    tour2_pre_to_tour2_conn_point_dist=distance(tour2_points[tour2_conn_point_pre_index],tour2_points[tour2_conn_point_index])
    # p2--p distance; next--p distance
    tour1_next_to_tour1_conn_point_dist=distance(tour1_points[tour1_conn_point_next_index],tour1_points[tour1_conn_point_index])
    # q2--q distance; next--q distance
    tour2_next_to_tour2_conn_point_dist=distance(tour2_points[tour2_conn_point_next_index],tour2_points[tour2_conn_point_index])

    # calcuate four costs of merging two tours by the four options.
    # first, next-next
    another_new_edge_distance=distance(tour1_points[tour1_conn_point_next_index],tour2_points[tour2_conn_point_next_index]) 
    distance_increase_01=conn_points_distance+another_new_edge_distance-tour1_next_to_tour1_conn_point_dist-tour2_next_to_tour2_conn_point_dist

    # second, prev-prev
    another_new_edge_distance=distance(tour1_points[tour1_conn_point_pre_index],tour2_points[tour2_conn_point_pre_index])
    distance_increase_02=conn_points_distance+another_new_edge_distance-tour1_pre_to_tour1_conn_point_dist-tour2_pre_to_tour2_conn_point_dist

    # third prev-next
    another_new_edge_distance=distance(tour1_points[tour1_conn_point_pre_index],tour2_points[tour2_conn_point_next_index])
    distance_increase_03=conn_points_distance+another_new_edge_distance-tour1_pre_to_tour1_conn_point_dist-tour2_next_to_tour2_conn_point_dist

    # fourth next-prev
    another_new_edge_distance=distance(tour1_points[tour1_conn_point_next_index],tour2_points[tour2_conn_point_pre_index])
    distance_increase_04=conn_points_distance+another_new_edge_distance-tour1_next_to_tour1_conn_point_dist-tour2_pre_to_tour2_conn_point_dist
    
    costs=[distance_increase_01,distance_increase_02,distance_increase_03,distance_increase_04]

    # next-next, prev-prev, prev-next, next-prev
    return costs




# two tours only contains the points coordincate, and already sort according to the visiting order.
def merge_two_cluster_tours(cluster1,cluster2):
   

    tour1_points=cluster1["sorted_tsp_data"]
    tour2_points=cluster2["sorted_tsp_data"]

   
    original_point_count=len(tour1_points)+len(tour2_points)

    tour1_conn_point_index,tour2_conn_point_index=find_nearest_inter_tour_nodes(tour1_points,tour2_points)
   
    # find the index of the pre point and the next point of tour1_conn_point_index in tour1_tour    
    if tour1_conn_point_index==0:
        tour1_conn_point_pre_index=len(tour1_points)-1
        tour1_conn_point_next_index=1   
    else:
        tour1_conn_point_pre_index=tour1_conn_point_index-1
        tour1_conn_point_next_index=(tour1_conn_point_index+1)%len(tour1_points)

    # find the index of the pre point and the next point of tour2_conn_point_index in tour2_tour
    if tour2_conn_point_index==0:
        tour2_conn_point_pre_index=len(tour2_points)-1
        tour2_conn_point_next_index=1
    else:
        tour2_conn_point_pre_index=tour2_conn_point_index-1
        tour2_conn_point_next_index=(tour2_conn_point_index+1)%len(tour2_points)

    # merge two tours by two nodes, there are four options: next-next, prev-prev, prev-next, next-prev.
    types=["next-next","prev-prev","prev-next","next-prev"]

    costs=calculate_four_costs(tour1_conn_point_index,tour2_conn_point_index,tour1_points,tour2_points, types)

    
    # find the minimum distance increase of costs.
    min_distance_increase=min(costs)
    min_distance_increase_index=costs.index(min_distance_increase)
    

    # merge the cluster1["ori_tsp_data"] and cluster2["ori_tsp_data"]
    ori_tsp_data=np.concatenate((cluster1["ori_tsp_data"],cluster2["ori_tsp_data"]),axis=0)

    new_cluster={}
    new_cluster["ori_tsp_data"]=ori_tsp_data

    new_tour_points=[]

    ## merge_by_edges(cluster1, cluster2, tour1_conn_point_index,tour2_conn_point_index, bestType)
    if min_distance_increase_index==0:
        # next-next connection
        new_tour_points.append(tour1_points[tour1_conn_point_index])
        new_tour_points.append(tour2_points[tour2_conn_point_index])

        if tour2_conn_point_index!=0:
            # append tour2 from q1 to head (index=0)
            for i in range(tour2_conn_point_pre_index,-1,-1):
                new_tour_points.append(tour2_points[i])

            # append tour2 from tail to q2
            for i in range(len(tour2_points)-1,tour2_conn_point_index,-1):
                new_tour_points.append(tour2_points[i])
        else:
            # append tour2 from q1 to head (index=0)
            for i in range(tour2_conn_point_pre_index,tour2_conn_point_index,-1):
                new_tour_points.append(tour2_points[i])

        if tour1_conn_point_index !=len(tour1_points)-1:
            # append tour1 from p2 to head
            for i in range(tour1_conn_point_next_index,len(tour1_points)):
                new_tour_points.append(tour1_points[i])
            # append tour1 from head to p2
            for i in range(0,tour1_conn_point_index):
                new_tour_points.append(tour1_points[i])
        else:
            # append tour1 from p2 to head
            for i in range(tour1_conn_point_next_index,tour1_conn_point_index):
                new_tour_points.append(tour1_points[i])

    elif min_distance_increase_index==1:
        # prev-prev connection

        new_tour_points.append(tour1_points[tour1_conn_point_index])
        new_tour_points.append(tour2_points[tour2_conn_point_index])
        if tour2_conn_point_index!=len(tour2_points)-1:
            # append tour2 from q1 to tail
            for i in range(tour2_conn_point_next_index,len(tour2_points)):
                new_tour_points.append(tour2_points[i])
            # append tour2 from head to q1
            for i in range(0,tour2_conn_point_index):
                new_tour_points.append(tour2_points[i])
        else:
            # append tour2 from q1 to tail
            for i in range(tour2_conn_point_next_index,tour2_conn_point_index):
                new_tour_points.append(tour2_points[i])

        if tour1_conn_point_index !=0:
            # append tour1 from p1 to head
            for i in range(tour1_conn_point_pre_index,-1,-1):
                new_tour_points.append(tour1_points[i])
            # append tour1 from head to p
            for i in range(len(tour1_points)-1,tour1_conn_point_index,-1):
                new_tour_points.append(tour1_points[i])
        else:
            # append tour1 from p1 to head
            for i in range(tour1_conn_point_pre_index,tour1_conn_point_index,-1):
                new_tour_points.append(tour1_points[i])


    elif min_distance_increase_index==2:
        # prev-next connection
        new_tour_points.append(tour1_points[tour1_conn_point_index])
        new_tour_points.append(tour2_points[tour2_conn_point_index])

        if tour2_conn_point_index!=0:
            # append tour2 from q2 to head(index=0, include 0)
            for i in range(tour2_conn_point_pre_index,-1,-1):
                new_tour_points.append(tour2_points[i])
            # append tour2 from tail to q2,indluce q2
            for i in range(len(tour2_points)-1,tour2_conn_point_index,-1):
                new_tour_points.append(tour2_points[i])
        else:
            # append tour2 from q2 to head(index=0, include 0)
            for i in range(tour2_conn_point_pre_index,tour2_conn_point_index,-1):
                new_tour_points.append(tour2_points[i])

        if tour1_conn_point_index !=0:
            # append tour1 from p1 to head (index =0, include head)
            for i in range(tour1_conn_point_pre_index,-1,-1):
                new_tour_points.append(tour1_points[i])
            # append tour1 from tail to p2, include p2
            for i in range(len(tour1_points)-1,tour1_conn_point_index,-1):
                new_tour_points.append(tour1_points[i])
        else:
            # append tour1 from p1 to head (index =0, include head)
            for i in range(tour1_conn_point_pre_index,tour1_conn_point_index,-1):
                new_tour_points.append(tour1_points[i]) 
    else:
        # next-prev connection
        new_tour_points.append(tour1_points[tour1_conn_point_index])
        new_tour_points.append(tour2_points[tour2_conn_point_index])

        if tour2_conn_point_index!=len(tour2_points)-1:
            # append tour2 from q2 to tail (include tail)
            for i in range(tour2_conn_point_next_index,len(tour2_points)):
                new_tour_points.append(tour2_points[i])
            # append tour2 from head(index=0) to q1
            for i in range(0,tour2_conn_point_index):
                new_tour_points.append(tour2_points[i])
        else:
            # append tour2 from q2 to tail (include tail)
            for i in range(tour2_conn_point_next_index,tour2_conn_point_index):
                new_tour_points.append(tour2_points[i])
        
        if tour1_conn_point_index !=len(tour1_points)-1:
            # append tour1 from p2 to tail (include tail)
            for i in range(tour1_conn_point_next_index,len(tour1_points)):
                new_tour_points.append(tour1_points[i])

            # append tour1 from head(index=0) to p
            for i in range(0,tour1_conn_point_index):
                new_tour_points.append(tour1_points[i])
        else:
            # append tour1 from p2 to tail (include tail)
            for i in range(tour1_conn_point_next_index,tour1_conn_point_index):
                new_tour_points.append(tour1_points[i])
    
    new_cluster["sorted_tsp_data"]=np.array(new_tour_points)
    # calculate the center of the new cluster, according to the ori_tsp_data
    new_cluster["center"]=np.mean(ori_tsp_data,axis=0)

    if original_point_count != len(new_tour_points):
        print("!!! Error occured when merging two clusters.")
        print("!!! The number of points in the two clusters is {}, but the number of points in the merged cluster is {}.".format(original_point_count,len(new_tour_points)))
        return None


    return new_cluster



# param
#   clusters: a list, each element is a cluster, each cluster is a dict, containing the following keys: points, center, tour, duration
def merge_sub_tsps_by_farthest_insertion(clusters):
    # fetch the center of each cluster, and form a list
    cluster_centers=[]
    for c in clusters:
        cluster_centers.append(c["center"])
    
    tour_index=farthest_insertion_2d_tsp(cluster_centers)
    return tour_index

def farthest_insertion_2d_tsp(points):
    n=len(points)

    distances=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            distances[i,j]=np.linalg.norm(np.array(points[i])-np.array(points[j]))

    # initialize the tour with two arbitrary points
    tour=[0,1]

    while len(tour) < n:
        max_distance = 0
        farthest_point = -1

        # Find the farthest point from the current tour
        for i in range(n):
            if i not in tour:
                for j in tour:
                    d = distances[i, j]
                    if d > max_distance:
                        max_distance = d
                        farthest_point = i

        # Find the edge (i, j) in the current tour with the maximum distance
        min_edge_distance = float('inf')
        min_edge_index = -1
        for i in range(len(tour)):
            j = (i + 1)%len(tour)
            d = distances[tour[i], farthest_point] + distances[farthest_point, tour[j]] - distances[tour[i], tour[j]]
            if d < min_edge_distance:
                min_edge_distance = d
                min_edge_index = i

        # Insert the farthest point into the tour
        tour.insert((min_edge_index + 1)%len(tour), farthest_point)

    return tour


