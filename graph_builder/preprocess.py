import sys
import os
import time

import svgpathtools
from svgpathtools import *
import numpy as np

import matplotlib.pyplot as plt
import pdb

file_name = sys.argv[1]
minEdgeLength = 0.2

paths, attributes, svg_attributes = svg2paths2(file_name)

# 1 
polys = []
isSeg = []  # if the path is a segment or a closed polygon
startEnds = []   # n x 4, start and end points of polygons
for path in paths:
    points = []
    nLines = 0

    # while not(type(path[0]) is svgpathtools.path.Line):
    #     path = path[1:] + path[0:1]
    startEnds.append([path.start.real, path.start.imag, path.end.real, path.end.imag])
    
    if not path.isclosed():
        isSeg.append(True)
        continue
        
    isSeg.append(False)

    for curve in path:
        points.append(np.array([curve.start.real, curve.start.imag]))
        
    if len(points) <=2: 
        continue

    points = np.array(points)

    vecs = [points[(i+1)%(len(points))] - points[i] for i in range(len(points))]
    if np.cross(vecs[0], vecs[1]) < 0:  # reorient
        points = points[::-1]

    polys.append(points)

same = lambda p,ps: (((p - ps)**2).sum(1) < 0.1)

ps = np.array(polys[0][0:1]).reshape(-1, 2)
ipolys = []
for poly in polys:
    ipoly = []
    for p in poly:
        ps = np.append(ps, [p], 0)
        ip = np.where(same(p,ps))[0][0]
        ipoly.append(ip)
    ipolys.append(ipoly)

# 2
newPolys = []
for poly in polys:
    newPoly = []
    ivToRemove = set()
    for i in range(len(poly)):
        j = (i+1) % len(poly)
        l = np.sqrt(((poly[i] - poly[j])**2).sum())
        if l < minEdgeLength:
            if same(poly[i], ps).sum() == 1:
                ivToRemove.add(i)
                print(poly[i])
            elif same(poly[j], ps).sum() == 1:
                ivToRemove.add(j)
                print(poly[j])
    for i in range(len(poly)):
        if not i in ivToRemove:
            newPoly.append(poly[i])

    assert(len(newPoly) > 2)   # increase minEdgeLength
    newPolys.append(newPoly)

polys = newPolys

same = lambda p,ps: (((p - ps)**2).sum(1) < 0.1)

ps = np.array(polys[0][0:1]).reshape(-1, 2)     # all points
ipolys = []     # indices of polygon points
for poly in polys:
    ipoly = []
    for p in poly:
        if same(p, ps).sum()>0:
            assert(same(p, ps).sum()==1)
        else:
            ps = np.append(ps, [p], 0)
        ip = np.where(same(p,ps))[0][0]
        ipoly.append(ip)
    ipolys.append(ipoly)

def showPoly(ipoly):
    for i in range(len(ipoly)):
        j = (i+1) % len(ipoly)
        plt.arrow(ps[ipoly[i]][0], ps[ipoly[i]][1], ps[ipoly[j]][0]-ps[ipoly[i]][0], ps[ipoly[j]][1]-ps[ipoly[i]][1], width=2, fc='k', ec='k')
    plt.show()

def addOrGetEdge(edge, edges):
    edge = np.array(edge)
    existed = False
    if len(edges) == 0:
        edges = edge.reshape(1, 2)
        iEdge = edges.shape[0] - 1
    else:
        nSameEdges = (edge == edges).prod(1) + (edge[::-1] == edges).prod(1)
        if nSameEdges.sum() == 1:
            iEdge = np.where(nSameEdges == 1)[0][0]
            existed = True
        else:
            assert(nSameEdges.sum() == 0)
            edges = np.append(edges,[edge], 0)
            iEdge = edges.shape[0] - 1
    return iEdge, edges, existed

es = np.array([], dtype=np.int32)
fs = np.array([], dtype=np.int32)
k_edges = []
is_facet = []
is_boundary_edges = []
is_boundary_nodes = [1] * len(ps)
target_angles = []
i_handle = len(ps) // 2

def addEdge(edge, k_edge, facet, boundary_edge, target_angle, edges, kes, fcs, bes, tas):
    edges.append(edge)
    kes.append(k_edge)
    fcs.append(facet)
    bes.append(boundary_edge)
    tas.append(target_angle)


for ipoly in ipolys:
    faces = []
    edges = []
    kes = []
    fcs = []
    bes = []
    tas = []

    if len(ipoly) > 4:
        pCenter = ps[ipoly].mean(0)
        ps = np.append(ps, [pCenter], 0)
        is_boundary_nodes.append(0)
        iCenter = len(ps) - 1

        for i, ip in enumerate(ipoly):
            ipNext = ipoly[(i + 1) % len(ipoly)]
            faces.append([ip, ipNext, iCenter])
            addEdge([ip, ipNext], k_edge=1, facet=0, boundary_edge=0 if i%2==0 else 1, target_angle=np.pi, 
                edges=edges, kes=kes, fcs=fcs, bes=bes, tas=tas)
            addEdge([ipNext, iCenter], k_edge=1, facet=1, boundary_edge=0, target_angle=np.pi, 
                edges=edges, kes=kes, fcs=fcs, bes=bes, tas=tas)
            addEdge([iCenter, ip], k_edge=1, facet=1, boundary_edge=0, target_angle=0, 
                edges=edges, kes=kes, fcs=fcs, bes=bes, tas=tas)
    else:
        if len(ipoly) == 4:
            faces.append(ipoly[0:3])
            faces.append([ipoly[2], ipoly[3], ipoly[0]])
            for i in range(4):
                j = (i+1) % 4
                addEdge([ipoly[i], ipoly[j]], k_edge=1, facet=0, boundary_edge=0 if i%2==0 else 1, target_angle=np.pi, 
                    edges=edges, kes=kes, fcs=fcs, bes=bes, tas=tas)
            addEdge([ipoly[0], ipoly[2]], k_edge=1, facet=1, boundary_edge=0, target_angle=0, 
                    edges=edges, kes=kes, fcs=fcs, bes=bes, tas=tas)
        else:   # len(ipoly) == 3:
            assert(len(ipoly) == 3)
            faces.append(ipoly[0:3])
            for i in range(3):
                j = (i+1) % 3
                addEdge([ipoly[i], ipoly[j]], k_edge=1, facet=0, boundary_edge=0 if i%2==0 else 1, target_angle=np.pi, 
                    edges=edges, kes=kes, fcs=fcs, bes=bes, tas=tas)

    if len(fs) == 0:
        fs = np.array(faces[0]).reshape(-1, 3)
        if len(faces)>1:
            fs = np.append(fs, faces[1:], 0)
    else:
        fs = np.append(fs, faces, 0)

    for i, edge in enumerate(edges):
        _, es, existed = addOrGetEdge(edge, es)
        if not existed:
            k_edges.append(kes[i])
            is_facet.append(fcs[i])
            is_boundary_edges.append(bes[i])
            target_angles.append(tas[i])

is_boundary_edges = []
for e in es:
    nPointsInFace = (e[0] == fs).sum(1) + (e[1] == fs).sum(1)       # number of edges points within a face {0, 1, 2}, 2 means edge is the face edge
    nAdjacentFaces = (nPointsInFace == 2).sum()
    if nAdjacentFaces == 1:
        is_boundary_edges.append(1)
    else:
        assert(nAdjacentFaces == 2)
        is_boundary_edges.append(0)

for i in range(len(target_angles)):
    if is_boundary_edges[i] or is_facet[i]:
        target_angles[i] = 0
    else:
        target_angles[i] = np.pi

startEnds = np.array(startEnds)
iSegs = []
for se in startEnds:
    assert(same(se[:2], ps[:, :2]).sum() == 1)
    assert (same(se[2:], ps[:, :2]).sum() == 1)
    i0 = np.where(same(se[:2], ps[:, :2]))[0][0]
    i1 = np.where(same(se[2:], ps[:, :2]))[0][0]
    iSegs.append([i0, i1])
iSegs = np.array(iSegs, dtype=np.int64)
isSeg = np.array(isSeg, dtype=bool)

for i, e in enumerate(es):
    same0 = same(e, iSegs)
    same1 = same(e[::-1], iSegs)
    if same0.sum() + same1.sum() < 1:
        continue
    
    if same0.sum() >= 1:
        iSeg = np.where(same0 * isSeg)[0][0]
    else:
        iSeg = np.where(same1 * isSeg)[0][0]
    
    if attributes[iSeg]['class'] == "st1":
        pass
    if attributes[iSeg]['class'] == "st2":
        target_angles[i] *= -1
    
    
# plot
for i, e in enumerate(es):
    p0 = ps[e[0]]
    p1 = ps[e[1]]
    if is_boundary_edges[i]:
        c = "r"
    elif is_facet[i]:
        c = "g"
    else:
        c = "b"
    plt.arrow(p0[0], p0[1], p1[0]-p0[0], p1[1]-p0[1], width=0, fc=c, ec=c)

ps = np.append(ps, np.zeros(len(ps)).reshape(-1, 1), 1)

for i, p in enumerate(ps):
    plt.text(p[0], p[1], str(i))

plt.show()


# output
k_edges = np.array(k_edges)
is_facet = np.array(is_facet)
is_boundary_edges = np.array(is_boundary_edges)
is_boundary_nodes = np.array(is_boundary_nodes)
target_angles = np.array(target_angles)
i_handle = len(ps) // 2

dirname = os.path.dirname(file_name)
newDirName = "./data/pattern/"
basename = os.path.basename(file_name)
newBasename = ".".join(basename.split('.')[:-1])
newFileName = os.path.join(newDirName, newBasename)

np.savez(newFileName,
         nodes=ps,
         edges=es,
         faces=fs,
         k_edges=k_edges,
         is_facet=is_facet,
         is_boundary_edges=is_boundary_edges,
         is_boundary_nodes=is_boundary_nodes,
         target_angles=target_angles,
         i_handle=i_handle)














