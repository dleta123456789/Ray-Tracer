# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 02:15:41 2021

@author: hamza
"""

from PIL import Image
from math import * 
import numpy as np

PI  = 3.14159265
INF = 9999999999
#-----------------------------------------------------------------------

class Color:
    BLACK = (0,0,0)
    RED   = (245,23,32)
    BLUE  = (46,139,192)
    GREEN = (24,165,88)
    PINK  = (250,38,160)
    GOLD  = (248,210,16)
    TEAL  = (43,124,133)
    WHITE = (255,255,255)
    CHROMA= (0,255,0)
    
#-----------------------------------------------------------------------

def dot(a, b):
    return a.x*b.x + a.y*b.y + a.z*b.z

def normalize(a):
    mag = a.magnitude()
    return Vector(a.x/mag,a.y/mag,a.z/mag)

def reflected(vector, axis):
    return vector - axis * vector.dot(axis) * 2



def refract(ray,normal,object_refraction_index,space_refraction_index):
    """
    vector is the ray
    axis is the normal
    refraction_index is self-explanatory
    
    """
    cos_theta = min(dot(ray,normal),1.0)
    index_div= (object_refraction_index/space_refraction_index)
    ray_perp=(ray+ normal*cos_theta)*index_div
    ray_perp_square= ray_perp*ray_perp
    ray_perp_subtracted= Vector(1.0,1.0,1.0)-ray_perp_square
    absoulute= Vector(abs(ray_perp_subtracted.x),abs(ray_perp_subtracted.y),abs(ray_perp_subtracted.z))
    square_root= Vector(sqrt(absoulute.x),sqrt(absoulute.y),sqrt(absoulute.z))
    negative_val_x= -(square_root.x)
    negative_val_y= -(square_root.y)
    negative_val_z= -(square_root.z)
    negative_square_root=Vector(negative_val_x,negative_val_y,negative_val_z)
    result=normal*negative_square_root
    return ray_perp+result
    

def SSAA(bitmap,width,height):
    Old_Bitmap=bitmap
    New_image = Image.new(mode = "RGB", size=(width//2, height//2), color=Color.WHITE)
    New_Bitmap=New_image.load()
    print("Height=",height,"Width=",width)
    for y in range(0,height,2):
        for x in range(0,width,2):
            """
            print("i,j=",i,j)
            Get RGB data for the 4 pixels
            R is first index
            G is second
            B is third index
            """
            R1,G1,B1=Old_Bitmap[x,y]
            R2,G2,B2=Old_Bitmap[x+1,y]
            R3,G3,B3=Old_Bitmap[x,y+1]
            R4,G4,B4=Old_Bitmap[x+1,y+1]
            """Average the color"""
            R=R1+R2+R3+R4
            R=int(R//4)
            G=G1+G2+G3+G4
            G=int(G//4)
            B=B1+B2+B3+B4
            B=int(B//4)
            
            """Store results in the new image"""
            New_Bitmap[x//2,y//2]=(R,G,B)
            #New_Bitmap.putpixel((i//2,j//2),(R,G,B))
    return New_image
    
class Vector:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
        self.w = 1

    def toString(self):
        return "S:["+"{:.9f}".format(self.x)+", "+"{:.9f}".format(self.y)+", "+"{:.9f}".format(self.z)+"]"

    def dot(self, b):
        return self.x*b.x + self.y*b.y + self.z*b.z
    
    def cross(self, b):
        return Vector(self.y*b.z-self.z*b.y, self.z*b.x-self.x*b.z, self.x*b.y-self.y*b.x)
        
    def magnitude(self):
        return sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    def normalize(self):
        mag = self.magnitude()

        if self.x==0:
            X=0
        else:
            X=self.x/mag
        if self.y ==0 :
            Y=0
        else:
            Y=self.y/mag
        if self.z == 0 :
            Z=0
        else:
            Z=self.z/mag
        return Vector(X, Y, Z)

    # Provide "overridden methods via the "__operation__" notation; allows you to do, for example, a+b, a-b, a*b
    def __add__(self, b):
        return Vector(self.x + b.x, self.y+b.y, self.z+b.z)

    def __sub__(self, b):
        return Vector(self.x-b.x, self.y-b.y, self.z-b.z)

    def __mul__(self, b):
        if type(b) == float or type(b) == int:
            return Vector(self.x*b, self.y*b, self.z*b)
        elif type(b) == Vector:
            return Vector(self.x*b.x, self.y*b.y, self.z*b.z)
        else:
            print(type(b))
            assert False


class Material:
    def __init__(self, ambient, diffuse, specular, shininess = 0, reflection = 0, refractiveIndex = -1):
        self.ambient    = ambient               # Vector(r,g,b)
        self.diffuse    = diffuse               # Vector(r,g,b)
        self.specular   = specular              # Vector(r,g,b)
        self.shininess  = shininess             # Integer between 0 and 100
        self.reflection = reflection            # Real number between 0 and 1.0
        self.refractiveIndex = refractiveIndex  # Real number. -1 means it is a solid object


class Intersectable:
    def intersect(self, ray):
        assert False, "Error: You must override this method"

class Sphere(Intersectable):
    def __init__(self,center, radius, material):
        self.center   = center
        self.radius   = radius
        self.material = material
        
    def intersect(self, ray):
        OC = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * ray.direction.dot(OC)
        c = OC.dot(OC) - self.radius*self.radius
        descriminant = b*b - 4*a*c
        if(descriminant > 0):
            t1 = (-b + sqrt(descriminant))/2
            t2 = (-b - sqrt(descriminant))/2
            if t1 > 0 and t2 > 0:
                dist = min(t1, t2)
                point = ray.pointAtParameter(dist)
                normal = (point - self.center).normalize()
                return Intersection(point, dist, normal, self)
        return Intersection(None, INF, None, None)
        
                
    def normal(self, b):
        return (b - self.center).normalize()


class Ray:
    def __init__(self, origin, direction):
        self.origin    = origin
        self.direction = direction

    def pointAtParameter(self, t):
        return self.origin + self.direction*t
    
    def toString(self):
        return "R->{ "+ self.origin.toString() + " -- " + self.direction.toString() +" }"
       
class Intersection:
	def __init__(self, point, distance, normal, object):
		self.point    = point
		self.distance = distance
		self.normal   = normal
		self.object   = object
		
		
class Screen:
    def __init__(self,  left, top, right, bottom, z):
        self.left   = left
        self.top    = top
        self.right  = right
        self.bottom = bottom
        self.z      = z
        
        
class Light:
    def __init__(self, position, material):
        self.position = position
        self.material = material
        
                    
class Triangle(Intersectable):
    def __init__(self, verts, normals, material):
        self.verts     = verts     # tuple of 3 vertices
        self.normals   = normals   # tuple of 3 normals
        self.material  = material
        self.name      = "triangle"
    
    def intersect(self, ray):
        A = self.verts[0]
        B = self.verts[1]
        C = self.verts[2]
        
        # Compute normal Vector:
        #n = ((B-A).cross(C-A)).normalize()
        n = self.normals[0]
        d = ray.direction
        P = ray.origin
        
        nd = n.dot(d)
        if nd == 0: # Ray is parallel to the triangle, intersection at INFINITY
            return Intersection(None, INF, None, None)            

        t = (n.dot(A) - n.dot(P)) * (1.0/(nd))
        if t < 0:
            return Intersection(None, INF, None, None)
            
        Q = ray.pointAtParameter(t)

        if ((B-A).cross(Q-A)).dot(n) < 0:
            return Intersection(None, INF, None, None)
        if ((C-B).cross(Q-B)).dot(n) < 0:
            return Intersection(None, INF, None, None)
        if ((A-C).cross(Q-C)).dot(n) < 0:
            return Intersection(None, INF, None, None)
        
        #Compute bericentric normal:
        alpha = ( ((C-B).cross(Q-B)).dot(n) ) / ( ((B-A).cross(C-A)).dot(n) )
        beta  = ( ((A-C).cross(Q-C)).dot(n) ) / ( ((B-A).cross(C-A)).dot(n) )
        gamma = ( ((B-A).cross(Q-A)).dot(n) ) / ( ((B-A).cross(C-A)).dot(n) )
        
        normal = self.normals[0]
        #normal = (self.normals[0] * alpha + self.normals[1] * beta + self.normals[2] * gamma).normalize()
        dist = t
        point = Q
        
        return Intersection(point, dist, normal, self)
    

class Mesh(Intersectable):
    def __init__(self, material):
        self.material = material
        self.vertices = []      # each vertex is a Vector3D
        self.normals  = []      # each normal is a Vector3D
        self.faces    = []      # each face is a list of 3 pairs [ (v, vn), (v, vn), (v, vn) ]
        self.position = Vector(0,0,0)
        self.scale    = Vector(1,1,1)
    
        
    def load(self, fileName):
        fo = open(fileName, "r+")
        
        lines = fo.readlines()
        
        for i in range(len(lines)):
            prefix = lines[i][0:2]
            if prefix == 'v ':
                v = lines[i].split()
                self.vertices.append(Vector(float(v[1]), float(v[2]), float(v[3])))
            elif prefix == 'vn':
                vn = lines[i].split()
                self.normals.append(Vector(float(vn[1]), float(vn[2]), float(vn[3])))
            elif prefix == 'f ':
                f = lines[i].split()
                triangle = []

                for j in range(1,4):
                    vindex = int(f[j].split('/')[0]) - 1
                    nindex = int(f[j].split('/')[2]) - 1
                    pair = (vindex, nindex)
                    triangle.append(pair)

                self.faces.append(triangle)
                
        print(fileName + " loaded...")
        fo.close()
        
    def intersect(self, ray):
        
        #Get Nearest Intersection:
        minHit = Intersection(None, INF, None, None)

        for f in self.faces:
            v1 = self.vertices[f[0][0]] * self.scale + self.position
            n1 = self.normals[f[0][1]]
            v2 = self.vertices[f[1][0]] * self.scale + self.position
            n2 = self.normals[f[1][1]]
            v3 = self.vertices[f[2][0]] * self.scale + self.position
            n3 = self.normals[f[2][1]]
            
            trig = Triangle((v1, v2, v3), (n1, n2, n3), self.material)
            
            hit = trig.intersect(ray)
            
            if(hit.point != None):
                if(hit.distance < minHit.distance):
                    minHit = hit
                
        return minHit
    
    def printData(self):
        for v in self.vertices:
            print("v ", v.x, " ", v.y, " ", v.z)
        for vn in self.normals:
            print("vn ", vn.x, " ", vn.y, " ", vn.z)
        for f in self.faces:
            print("f ", str(f[0][0]+1)+"/"+str(f[0][1]+1), " ", str(f[1][0]+1)+"/"+str(f[1][1]+1), " ",str(f[2][0]+1)+"/"+str(f[2][1]+1))
    

        
def MakeCube(mat = None):
    '''
    This function will create a cube with the bottom left point being the anchor point.\n
    x - The horizontal axis of bottom left corner of the cube\n
    y - The vertical axis of bottom left corner of the cube\n
    width - How wide the cube should be (0.0 - 9.9)\n
    height - How tall the cube should be (0.0 - 9.9)\n
    mat - Material of the object. A Material object.
    '''
    if mat == None:
        mat = Material(Vector(0.1, 0, 0),Vector(1,0.462745,0),Vector(1,1,1), 100, 0.5, -1)

    m = Mesh(mat)
    m.vertices = [
        Vector(1.000000, -1.000000, -1.000000),
        Vector(1.000000, -1.000000, 1.000000),
        Vector(-1.000000, -1.000000, 1.000000),
        Vector(-1.000000, -1.000000, -1.000000),
        Vector(1.000000, 1.000000, -0.999999),
        Vector(0.999999, 1.000000, 1.000001),
        Vector(-1.000000, 1.000000, 1.000000),
        Vector(-1.000000, 1.000000, -1.000000)
    ]

    m.normals = [
        Vector(0.000000, -1.000000, 0.000000),
        Vector(0.000000, 1.000000, 0.000000),
        Vector(1.000000, 0.000000, 0.000000),
        Vector(-0.000000, 0.000000, 1.000000),
        Vector(-1.000000, -0.000000, -0.000000),
        Vector(0.000000, 0.000000, -1.000000)
    ]

    m.faces = [
        [(1,0), (2,0), (3,0)],
        [(7,1), (6,1), (5,1)],
        [(4,2), (5,2), (1,2)],
        [(5,3), (6,3), (2,3)],
        [(2,4), (6,4), (7,4)],
        [(0,5), (3,5), (7,5)],
        [(0,0), (1,0), (3,0)],
        [(4,1), (7,1), (5,1)],
        [(0,2), (4,2), (1,2)],
        [(1,3), (5,3), (2,3)],
        [(3,4), (2,4), (7,4)],
        [(4,5), (0,5), (7,5)]
    ]

    return m

def MakeSquare(x, y, z, width, height, objects, mat = None, perspective = False, transformation = None):
    '''
    This function will create a 2D square on the screen with bottom left point as its anchor point.\n
    x - The horizontal axis of bottom left corner of the cube\n
    y - The vertical axis of bottom left corner of the cube\n
    width - How wide the cube should be (0.0 - 9.9)\n
    height - How tall the cube should be (0.0 - 9.9)\n
    objects - The objects array which contains all the objects to be rendered on the screen.\n
    mat - Material of the object. A Material object.
    '''
    
    if mat == None:
        mat = Material(Vector(0.1, 0, 0),Vector(1,0.462745,0),Vector(1,1,1), 100, 0.5, -1)

    if not perspective:
        t1 = Triangle((Vector(x+width,y+height,z), Vector(x,y,z), Vector(x+width,y,z)), # Vertices
                    (Vector(0, 0, 1), Vector(0, 0, 1), Vector(0, 0, 1)), # Normals
                    mat)
        
        t2 = Triangle((Vector(x+width, y+height, z), Vector(x,y+height,z), Vector(x,y,z)), # Vertices
                    (Vector(0, 0, 1), Vector(0, 0, -1), Vector(0, 0, 1)), # Normals
                    mat)
    
    else:
        t1 = Triangle((Vector(x+width,y+height+.3,z), Vector(x,y,z), Vector(x+width,y+.3,z)), # Vertices
                    (Vector(0, 0, 1), Vector(0, 0, 1), Vector(0, 0, 1)), # Normals
                    mat)
        
        t2 = Triangle((Vector(x+width, y+height+.3, z), Vector(x,y+height,z), Vector(x,y,z)), # Vertices
                    (Vector(0, 0, 1), Vector(0, 0, -1), Vector(0, 0, 1)), # Normals
                    mat)



    if transformation != None:
        # t1.verts = (transformation.Transform(t1.verts[0]), t1.verts[1], t1.verts[2])
        # t2.verts = (transformation.Transform(t2.verts[0]), transformation.Transform(t2.verts[1]), t2.verts[2])
        t1 = transformation.Transform(t1)
        t2 = transformation.Transform(t2)
    



    objects.append(t1)
    objects.append(t2)

    return objects

class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix

    def __mul__(self, b):
        if len(self.matrix[0]) == len(b.matrix):
            m = [[0]*len(b.matrix[0]) for i in range(len(self.matrix))]

            # iterate through rows of X
            for i in range(len(self.matrix)):
                # iterate through columns of Y
                for j in range(len(b.matrix[0])):
                    # iterate through rows of Y
                    for k in range(len(b.matrix)):
                        m[i][j] += self.matrix[i][k] * b.matrix[k][j]
        
        return Matrix(m)

    def ToString(self):
        for i in self.matrix:
            print(i)
            
class TransformationMatrices:
    def __init__(self, rotation, scaling, translation):
        self.scaling = scaling
        self.rotation = rotation
        self.translation = translation


    def Transform(self, vertex):
        temp = []
        for i in range(len(vertex.verts)):
            v_matrix = Matrix([[vertex.verts[i].x], [vertex.verts[i].y], [vertex.verts[i].z]])
            
            sr = self.scaling * self.rotation
            srt = sr * self.translation
            srtv = srt * v_matrix

            temp.append(srtv)
        
        new_vert = (
            Vector(temp[0].matrix[0][0], temp[0].matrix[1][0], temp[0].matrix[2][0]),
            Vector(temp[1].matrix[0][0], temp[1].matrix[1][0], temp[1].matrix[2][0]),
            Vector(temp[2].matrix[0][0], temp[2].matrix[1][0], temp[2].matrix[2][0])
        )
        
        vertex.verts = new_vert
        return vertex
