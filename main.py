'''
Computer Graphics - CSCS 453
Project

Group members:
Ans Naveed          21-10567
Hamza Muhammad      21-10059
'''


import sys
from PIL import Image
from math import * 
from util import *
import time


width   = 400
height  = 300

BOUNCES = 4

#-----------------------------------------------------------------------

def GetNearestIntersection(objects, ray):
    nearestObj = None
    minHit = Intersection(None, INF, None, None)
    
    for obj in objects:
        hit = obj.intersect(ray)
        if hit.distance < minHit.distance:
            nearestObj = obj
            minHit = hit
            
    return minHit


def Blinn(light, hit, eye):
    ka = hit.object.material.ambient
    kd = hit.object.material.diffuse
    ks = hit.object.material.specular
    la = light.material.ambient
    ld = light.material.diffuse
    ls = light.material.specular
    L  = (light.position - hit.point).normalize()
    N  = hit.normal
    V  = (eye - hit.point).normalize()
    a  = hit.object.material.shininess
    
    A = ka*la
    D = kd*ld*L.dot(N)
    if a==0:
         S = ks*ls*pow(abs(N.dot((L+V).normalize())) ,0)
    else:
         S = ks*ls*pow(abs(N.dot((L+V).normalize())) ,a/4)
 
    p = A + D + S
    
    return p

#Glas_Mat=Material(ambinet,diffuse,specular,shiness,reflection,)
#-----------------------------------------------------------------------

def main():
    global bitmap
    start_time = time.process_time()
    eye = Vector(0,0,1)
    ratio = float(width) / height
    screen = Screen(-1, 1 / ratio, 1, -1 / ratio, 0)
    
    
    
    objects = []
    lights= [] 
   #--------------------------------------------------------------------------
    #Copy code from here 
    #Initializing different material objects
    metal_mat = Material(Vector(0,0,0), Vector(0.972, 0.960, 0.915), Vector(0.5,.5,.5), 10, 0.9, -1)
    glass_mat = Material(Vector(0.0, 0.0, 0.0), Vector(0.588235, 0.670588, 0.929412),Vector(0.5, 0.5, 0.5), 96, 1, -1)
    wood_mat = Material(Vector(0, 0, 0), Vector(0.5, 0.5, 0.5), Vector(0, 0, 0), 1, 1, -1)
    road_mat = Material(Vector(0,0,0), Vector(.277,.277,.277), Vector(0.3,0.3,0.3), 25)


    # Initializing transformation matrix
    theta = -90 * pi/180 #-45 * pi/180
    if theta < 0: # clockwise
        rotation = Matrix(
            [
                [cos(theta), -sin(theta), 0],
                [sin(theta), cos(theta), 0],
                [0, 0, 1]
            ]
        )

    else: # counter clockwise
        rotation = Matrix(
            [
                [cos(theta), sin(theta), 0],
                [-sin(theta), cos(theta), 0],
                [0, 0, 1]
            ]
        )


    sx = 1
    sy = 1
    scale = Matrix(
        [
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ]
    )
    
    tx = -3
    ty = 2.5
    translation = Matrix(
        [
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ]
    )
    
    tm = TransformationMatrices(rotation, scale, translation)

    


    #Top left building
    m = MakeCube(metal_mat)
    m.position=Vector(-1.6,-.5,-3)
    m.scale=Vector(0.6,1.4,0.2)
    objects.append(m)

    objects = MakeSquare(-2.15, -1.5, -2.799, 0.25, 2, objects, glass_mat)
    objects = MakeSquare(-1.85, -1, -2.799, 0.44, 1, objects, wood_mat)
    objects = MakeSquare(-1.35, -1.5, -2.799, 0.25, 2, objects, glass_mat)
      

    # Bottom left building
    m = MakeCube(metal_mat)
    m.position=Vector(-1,-.4,-.5)
    m.scale=Vector(0.3,0.5,0.1)
    objects.append(m)
    del(m)

    objects = MakeSquare(-1.2, -0.2, -.399, 0.15, 0.25, objects, glass_mat, transformation=tm)
    objects = MakeSquare(-1.2, -0.6, -.399, 0.15, 0.25, objects, glass_mat)
    objects = MakeSquare(-0.98, -0.2, -.399, 0.15, 0.25, objects, glass_mat)
    objects = MakeSquare(-0.98, -0.6, -.399, 0.15, 0.25, objects, glass_mat)
    


    #sphere Building
    s1 = Sphere(Vector(1.8,0,-3.3),  1.3, Material(Vector(0.0, 0.0, 0.0), Vector(0.588235, 0.670588, 0.929412),Vector(0.5, 0.5, 0.5), 96, 10, -1))
    objects.append(s1)

    #Pyramid Building 
    Building=Mesh(Material(Vector(1,0,0), Vector(1, 0, 0), Vector(1,1,1), 10, .1, -1))
    Building.load("building.obj")
    Building.position=Vector(0.5,-0.3,0)
    Building.scale=Vector(0.2,0.2,0.2)
    objects.append(Building)

    #Clouds
    Cloud1=Mesh(Material(Vector(0.453,0.468,0.5), Vector(0.453, 0.453, 0.453), Vector(0.453,0.453,0.453), 10, 0.9, -1))
    Cloud1.load(r"cloud.obj")
    Cloud1.position=Vector(0.1,0.2,0.1)
    Cloud1.scale=Vector(0.1,0.1,0.1)
    objects.append(Cloud1)
    
    Cloud2=Mesh(Material(Vector(0.453,0.468,0.5), Vector(1, 1, 1), Vector(1,1,1), 1, 0.9, -1))
    Cloud2.load(r"cloud.obj")
    Cloud2.position=Vector(-0.7,0.3,0.1)
    Cloud2.scale=Vector(0.1,0.1,0.1)
    objects.append(Cloud2)
    
    # Stand
    Sphere1= Sphere(Vector(0,-0.1,-2), 0.1, Material(Vector(1,0,0), Vector(1, 1, 1), Vector(1,1,1), 100, 0.9, -1))
    objects.append(Sphere1)
    Post=Mesh(Material(Vector(1,0,0), Vector(1, 1, 1), Vector(1,1,1), 100, 0.9, -1))
    Post.load(r"signpost.obj")
    Post.position=Vector(0.0,-0.5,-2)
    Post.scale=Vector(1,1,1)
    objects.append(Post)
    # refraction sphere
    s2 = Sphere(Vector( 0,0.2,0),0.4,      Material(Vector(0,0,0),Vector(0,1,0),Vector(1,1,1), 100, 0.2, 1.5))
    objects.append(s2)
    
    objects.append(Sphere(Vector(0,-9000,0),  9000-0.7, road_mat))

    
    #objects = MakeSquare(0, 0, -1.3, 1, 1, objects, road_mat, transformation = tm)

    lights.append( Light(Vector(-5,5,5), Material(Vector(1,1,1),Vector(1,1,1),Vector(1,1,1), -1, -1)) )
    lights.append( Light(Vector(5,5,5), Material(Vector(1,1,1),Vector(1,1,1),Vector(1,1,1), -1, -1)) )

    
    for frame in range(1):
    
        img = Image.new(mode = "RGB", size=(width, height), color=Color.BLACK)
        bitmap = img.load() # create the pixel data

        lights[0].position.x += 10/300
        # s3.center.y -= 1/1000
        # s2.center.x -= 1/800
        # t1.verts[1].x -= 1/900
        # t1.verts[1].z -= 1/2000

#--------------------------------------------------------------
#---                    YOUR CODE HERE                      ---
#--------------------------------------------------------------
        sys.setrecursionlimit(100000)
        
        #breakpoint()
        
        deltaX = (screen.right - screen.left)/(width-1)
        deltaY = (screen.top - screen.bottom)/(height-1)

        for y in range(height):
            for x in range(width):
                pixel = Vector(screen.left+x*deltaX, screen.top-y*deltaY, screen.z)    
    
                bitmap[x,y] = Color.BLACK
                color = Vector(.529,.808,.922)    
                # color = Vector(0,0,0)   
                
                newEye = eye
                reflection = 1
                """ Assume that the object at start"""
                refraction_index= 1
                direction = (pixel - newEye).normalize()
                for light in lights:
                    for k in range(BOUNCES):
                        
                        pixelRay = Ray(newEye, direction) 
                        
                        hit = GetNearestIntersection(objects, pixelRay)
                        if hit.point == None:
                            break
                        
                        """space refraction is the index of the medium outside objects"""
                        space_refraction_index= hit.object.material.refractiveIndex
                        
                        shadowed = False
                        shiftedPoint = hit.point + (hit.normal * 1e-5)
                        shadowRay = Ray(shiftedPoint, (light.position - shiftedPoint).normalize())
                        result = GetNearestIntersection(objects, shadowRay)
    
                        if result.point != None:
                            color=Vector(0,0,0)
                            break
                        
                        if (hit.object.material.refractiveIndex==-1):
                            color = Vector(0,0,0)
                            color += Blinn(light, hit, newEye) * reflection
                            reflection *= hit.object.material.reflection
            
                            # new ray origin and direction
                            newEye = shiftedPoint
                            direction = reflected(direction, hit.normal)
                        else:
                            #color += Blinn(light, hit, newEye) * refraction_index
                            refraction_index= hit.object.material.refractiveIndex
                            
                            #newEye= shiftedPoint
                            direction= refract(direction,hit.normal,refraction_index,space_refraction_index)
                            
                            

                bitmap[x,y] = (int(color.x*256), int(color.y*256), int(color.z*256))

            if y % 20 == 0:
                print("progress: %d %%" % ((y+1)/height*100.0))
#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------
        """Apply AA here"""
        img=SSAA(bitmap,width,height)
        print(str((time.process_time() - start_time)/60) + " minutes to render the image")
        # img.show()
        img.save("pic1 transformation.png")
        # img.save("fig" + f'{frame:06}' + ".png")
        print("Saving ---> fig" + f'{frame:06}' + ".png")
        # img.close()

main()
