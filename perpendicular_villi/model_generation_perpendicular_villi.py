import numpy as np
import matplotlib.pyplot as plt
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax2
from OCC.Core.GC import GC_MakeCircle
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace)
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCC.Core.Geom import Geom_Ellipse
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Extend.DataExchange import write_step_file
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
import os
import scipy.ndimage as ndi
from skimage.morphology import medial_axis

i = 1 # The ID of the sample
b = 0.156 #ellipse semi-minor axis (mm) 
a = 2 * b #aspect ratio of 2
height = 0.9 #height of the elliptical cylinder (mm)

output_dir = f"geometries_b{b:.3f}mm/{i}/Geometry"
os.makedirs(output_dir, exist_ok=True)

r1 = 0.075 #radius of the immature villus
r2 = 0.05 #radius of the mature villus
gap_shape = 5.14  #shape parameter of the gamma probability density function
gap_scale = 0.0148  # scale parameter of the gamma probability density function
min_clearance = 0.001 # in mm

z_slice = 0.6 # in mm
y_check = 0.095 # in mm

if abs(y_check/b) <= 1:
    x_wall = a * np.sqrt(1 - (y_check/b)**2)
    x_wall_candidates = np.array([-x_wall , x_wall])
else:
    raise ValueError("y_check exceeds ellipse boundary.")


#defining the location of the first villus   
for attempt in range(1000):
    gap1 = np.random.gamma(gap_shape, gap_scale)
    target_dist = gap1 + r1     
    x_wall = np.random.choice(x_wall_candidates)    
    x1 = x_wall - np.sign(x_wall)*target_dist    
    x1 = x_wall - np.sign(x_wall)*target_dist
    dist_to_wall = np.min(np.abs(x1 - x_wall_candidates))   
    if dist_to_wall >= r1 + min_clearance:
        break
else:
    raise RuntimeError("Failed to place the first villus, please run again.")    


#defining the location of the second villus 
for attempt in range(1000):
    gap2 = np.random.gamma(gap_shape, gap_scale)
    center_dist = gap2 + r1 + r2
    if x1 > 0:
        x2 = x1 - center_dist
    else:
        x2 = x1 + center_dist    
    dist_to_wall = np.min(np.abs(x2 - x_wall_candidates))
    if dist_to_wall >= r2 + min_clearance:    
        if abs(x2 - x1) >= r1 + r2 + min_clearance:
            break
else:
    raise RuntimeError("Failed to place the second villus, please run again.")    
    
#defining villi path points    
y_path = np.linspace(-b, b, 10)
x_v1 = np.full_like(y_path, x1) 
x_v2 = np.array([x2, x2, x2-0.005, x2, x2+0.005, x2, x2-0.005, x2, x2+0.001, x2])   
z_v1 = np.full_like(y_path, z_slice)
z_v2 = np.full_like(y_path, z_slice)

# function for lofting the villus
def loft_solid(path_pts , radius):
    wires = []
    for pt in path_pts:
        ax = gp_Ax2(pt, gp_Dir(0, 1, 0))  # y-axis direction
        circ = GC_MakeCircle(ax, radius).Value()
        edge = BRepBuilderAPI_MakeEdge(circ).Edge()
        wire = BRepBuilderAPI_MakeWire(edge).Wire()
        wires.append(wire)
    loft = BRepOffsetAPI_ThruSections(True , False)
    for wire in wires:
        loft.AddWire(wire)
    loft.Build()
    return loft.Shape()    
    
path_v1 = [gp_Pnt(x, y, z) for x, y, z in zip(x_v1, y_path, z_v1)]
solid_v1 = loft_solid(path_v1, r1)

path_v2 = [gp_Pnt(x, y, z) for x, y, z in zip(x_v2, y_path, z_v2)]
solid_v2 = loft_solid(path_v2, r2)

#function for building the elliptical cylinder 
def make_elliptical_cylinder(a, b, height):
    ax2 = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    ellipse = Geom_Ellipse(ax2, a, b)
    edge = BRepBuilderAPI_MakeEdge(ellipse).Edge()
    wire = BRepBuilderAPI_MakeWire(edge).Wire()
    face = BRepBuilderAPI_MakeFace(wire).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(0,0, height)).Shape()    

elliptical_cylinder = make_elliptical_cylinder(a, b, height)

#the fixed 4 cylinders (two located above and two below the villi considered for WSS analysis) 
r_fixed = 0.075
x_fixed_positions = [-0.135, 0.135]
z_fixed_levels = [0.375+0.035, 0.825-0.035 ]
fixed_cylinders = []
for zc in z_fixed_levels:
    for xc in x_fixed_positions:
        center = gp_Pnt(xc, -b, zc)
        cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(center, gp_Dir(0, 1 , 0)), r_fixed, 2*b).Shape()
        fixed_cylinders.append(cyl)
fixed_combined = fixed_cylinders[0]
for fc in fixed_cylinders[1:]:
    fixed_combined = BRepAlgoAPI_Fuse(fixed_combined,fc).Shape()
elliptical_with_fixed_cut = BRepAlgoAPI_Cut(elliptical_cylinder, fixed_combined).Shape()  #subtracting the fixed cylinders   
    
fused_villi = BRepAlgoAPI_Fuse(solid_v1,solid_v2).Shape()
cut_result = BRepAlgoAPI_Cut(elliptical_with_fixed_cut, fused_villi).Shape() #subtracting the villi

write_step_file(fused_villi, os.path.join(output_dir, f"villi{i}.step"))   
write_step_file(cut_result, os.path.join(output_dir, f"subtraction{i}.step"))

# distance map and skeleton 
# xy plane plots
res = 0.001
xs = np.arange(-a, a + res, res)
ys = np.arange(-b, b + res, res)
X_xy, Y_xy = np.meshgrid(xs, ys, indexing='ij')
inside_ellipse_xy = (X_xy/a)**2 + (Y_xy/b)**2 <= 1
outside_villus1_xy = ~((np.abs(X_xy-x1) <= r1) & (np.abs(Y_xy) <= b))
outside_villus2_xy = ~((np.abs(X_xy-x2) <= r2) & (np.abs(Y_xy) <= b))
mask_xy = inside_ellipse_xy & outside_villus1_xy & outside_villus2_xy
  
edt_pixels_xy = ndi.distance_transform_edt(mask_xy)
edt_xy = edt_pixels_xy * res

skel_medial_xy, dist_pix_medial_xy = medial_axis(mask_xy, return_distance=True)
skel_dist_mm_medial_xy = dist_pix_medial_xy[skel_medial_xy] * res
mean_diameter_xy = 2 * np.mean(skel_dist_mm_medial_xy)
print(f"mean pore diameter in xy plane: {mean_diameter_xy:.6f} mm")

#distance map plot (x-y plane)
plt.figure(figsize=(6, 5))
plt.imshow(np.ma.masked_where(~mask_xy, edt_xy).T, origin='lower', extent=[-a, a, -b, b], cmap='viridis')
plt.colorbar(label="distance to nearest boundary [mm]")
plt.title("distance map (x-y plane at z = 0.6)")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"distance_map_xy_{i}.png"), dpi=300)
plt.show()

#distance map + medial axis skeleton plot (x-y plane)
plt.figure(figsize=(6, 5))
plt.imshow(np.ma.masked_where(~mask_xy, edt_xy).T, origin='lower', extent=[-a, a, -b, b], cmap='viridis')
plt.colorbar(label="distance to nearest boundary [mm]")
plt.contour(xs, ys, skel_medial_xy.T.astype(float), levels=[0.5], colors='red', linewidths=0.8)
plt.title("distance map + medial axis skeleton (x-y plane at z = 0.6)")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"distance_map_plus_medial_skeleton_xy_{i}.png"), dpi=300)
plt.show()


# xz plane plots
xs = np.arange(-a, a + res, res)
zs = np.arange(z_fixed_levels[0] + r_fixed, z_fixed_levels[1] - r_fixed + res , res)
X_xz, Z_xz = np.meshgrid(xs, zs, indexing='ij')
inside_domain_xz = np.abs(X_xz) <= a 
inside_domain_xz[0,:] = False 
inside_domain_xz[-1,:] = False  
idx_z_min = np.argmin(np.abs(zs - (z_fixed_levels[0] + r_fixed)))
idx_z_max = np.argmin(np.abs(zs - (z_fixed_levels[1] - r_fixed)))
inside_domain_xz[:,idx_z_min] = False  
inside_domain_xz[:,idx_z_max] = False   
outside_villus1_xz = ((X_xz-x1)**2 + (Z_xz-z_slice)**2) > r1**2
outside_villus2_xz = ((X_xz-x2)**2 + (Z_xz-z_slice)**2) > r2**2  
outside_fixed_cylinders_xz = np.ones_like(inside_domain_xz, dtype=bool)
for zc in z_fixed_levels:
    if (zc + r_fixed < zs[0]) or (zc -r_fixed > zs[-1]):
        continue     
    for xc in x_fixed_positions:
        fixed_mask_xz = ((X_xz-xc)**2 + (Z_xz - zc)**2) <= r_fixed**2
        outside_fixed_cylinders_xz &= ~fixed_mask_xz  

mask_xz = inside_domain_xz & outside_villus1_xz & outside_villus2_xz & outside_fixed_cylinders_xz
        
edt_pixels_xz = ndi.distance_transform_edt(mask_xz)
edt_xz = edt_pixels_xz * res 
       
skel_medial_xz, dist_pix_medial_xz = medial_axis(mask_xz, return_distance=True)
skel_dist_mm_medial_xz = dist_pix_medial_xz[skel_medial_xz] * res
mean_diameter_xz = 2 * np.mean(skel_dist_mm_medial_xz)
print(f"mean pore diameter in xz plane: {mean_diameter_xz:.6f} mm")

#distance map (x-z plane)
plt.figure(figsize=(6, 5))
plt.imshow(np.ma.masked_where(~mask_xz, edt_xz).T, origin='lower', extent=[-a, a , zs[0], zs[-1]], cmap='viridis')
plt.colorbar(label="distance to nearest boundary [mm]")
plt.title("distance Map (x-z plane at y = 0)")
plt.xlabel("x [mm]")
plt.ylabel("z [mm]")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"distance_map_xz_{i}.png"), dpi=300)
plt.show()

#distance map + medial axis skeleton plot (x-z plane)
plt.figure(figsize=(6, 5))
plt.imshow(np.ma.masked_where(~mask_xz, edt_xz).T, origin='lower', extent=[-a, a , zs[0], zs[-1]], cmap='viridis')
plt.colorbar(label="distance to nearest boundary [mm]")
plt.contour(xs, zs, skel_medial_xz.T.astype(float), levels=[0.5], colors='red', linewidths=0.8)
plt.title("distance map + medial axis skeleton (x-z plane at y = 0)")
plt.xlabel("x [mm]")
plt.ylabel("z [mm]")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"distance_map_plus_medial_skeleton_xz_{i}.png"), dpi=300)
plt.show()
      
