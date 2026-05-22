import numpy as np
import matplotlib.pyplot as plt
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax1, gp_Ax2, gp_Trsf
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
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
import os
import scipy.ndimage as ndi
from skimage.morphology import medial_axis

i = 1 # The ID of the sample
b = 0.128 # ellipse semi-minor axis (mm) 
b_eff = b * np.sqrt(2) # b of the elliptical cross-section parallel to the villi axes
a = 2 * b_eff #aspect ratio of 2 with respect to b_eff
height = 1.166 #height of the elliptical cylinder (mm)

output_dir = f"geometries_b{b:.3f}mm/{i}/Geometry"
os.makedirs(output_dir, exist_ok=True)

r1 = 0.075 #radius of the immature villus
r2 = 0.05 #radius of the mature villus
gap_shape = 5.14   #shape parameter of the gamma probability density function
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
y_path = np.linspace(-2*b, 2*b, 10)
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

#rotating the elliptical cylinder around x-axis with the pivot at (0, 0, 0.6)
pivot_point = gp_Pnt(0, 0, 0.6)
rotation_axis = gp_Ax1(pivot_point, gp_Dir(1, 0, 0))  # x-axis
rotation = gp_Trsf()
rotation.SetRotation(rotation_axis, np.radians(45))  #45 degrees

# applying the transformation
rotated_cylinder = BRepBuilderAPI_Transform(elliptical_cylinder, rotation, True).Shape()
elliptical_cylinder = rotated_cylinder  

#the fixed 4 cylinders (two located above and two below the villi considered for WSS analysis)
r_fixed = 0.075
x_fixed_positions = [-0.135, 0.135]
z_fixed_levels = [0.440, 0.760]
fixed_cylinders = []
for zc in z_fixed_levels:
    for xc in x_fixed_positions:
        center = gp_Pnt(xc, -8*b, zc)
        cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(center, gp_Dir(0, 1 , 0)), r_fixed, 16*b).Shape()
        fixed_cylinders.append(cyl)
fixed_combined = fixed_cylinders[0]
for fc in fixed_cylinders[1:]:
    fixed_combined = BRepAlgoAPI_Fuse(fixed_combined,fc).Shape()
elliptical_with_fixed_cut = BRepAlgoAPI_Cut(elliptical_cylinder, fixed_combined).Shape() #subtracting the fixed cylinders     
    
fused_villi = BRepAlgoAPI_Fuse(solid_v1,solid_v2).Shape()
cut_result = BRepAlgoAPI_Cut(elliptical_with_fixed_cut, fused_villi).Shape()  #subtracting the villi    

#rotating the cut_result for -45 deg so that the axis of the elliptical cylinder becomes along z direction 
pivot_point = gp_Pnt(0, 0, 0.6)
rotation_axis = gp_Ax1(pivot_point, gp_Dir(1, 0, 0))  # x-axis through pivot

inv_rotation = gp_Trsf()
inv_rotation.SetRotation(rotation_axis , np.radians(-45.0))  

fused_villi_aligned = BRepBuilderAPI_Transform(fused_villi, inv_rotation, True).Shape()
cut_result_aligned = BRepBuilderAPI_Transform(cut_result, inv_rotation, True).Shape()

write_step_file(fused_villi_aligned, os.path.join(output_dir, f"villi{i}.step"))
write_step_file(cut_result_aligned, os.path.join(output_dir, f"subtraction{i}.step"))

# distance map and skeleton
# xy plane plots 
res = 0.001
xs = np.arange(-a, a + res, res)
ys = np.arange(-b, b + res, res)
X, Y = np.meshgrid(xs, ys, indexing='ij')
inside_ellipse_xy = (X/a)**2 + (Y/b)**2 <= 1
sin45 = np.sqrt(2)/2
outside_villus1_xy = (X-x1)**2 + (Y*sin45)**2 > r1**2
outside_villus2_xy = (X-x2)**2 + (Y*sin45)**2 > r2**2
mask_xy_aligned = inside_ellipse_xy & outside_villus1_xy & outside_villus2_xy

edt_pixels_xy_aligned = ndi.distance_transform_edt(mask_xy_aligned)
edt_xy_aligned = edt_pixels_xy_aligned * res

skel_medial_xy_aligned, dist_pix_medial_xy_aligned = medial_axis(mask_xy_aligned, return_distance=True)
skel_dist_mm_medial_xy_aligned = dist_pix_medial_xy_aligned[skel_medial_xy_aligned] * res
mean_diameter_xy_aligned = 2 * np.mean(skel_dist_mm_medial_xy_aligned)
print(f"mean pore diameter in xy plane: {mean_diameter_xy_aligned:.6f} mm")

# distance map plot (x-y plane)
plt.figure(figsize=(6, 5))
plt.imshow(np.ma.masked_where(~mask_xy_aligned, edt_xy_aligned).T, origin='lower', extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap='viridis')
plt.colorbar(label="distance to nearest boundary [mm]")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"distance_map_xy_aligned_{i}.png"), dpi=300)
plt.show()

# distance map + medial axis skeleton plot (x-y plane)
plt.figure(figsize=(6, 5))
plt.imshow(np.ma.masked_where(~mask_xy_aligned, edt_xy_aligned).T, origin='lower', extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap='viridis')
plt.colorbar(label="distance to nearest boundary [mm]")
plt.contour(xs, ys, skel_medial_xy_aligned.T.astype(float), levels=[0.5], colors='red', linewidths=0.8)
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"distance_map_plus_medial_xy_aligned_{i}.png"), dpi=300)
plt.show()


# xz plane plots

cos45 = np.sqrt(2)/2
zprime_min = z_fixed_levels[0] + r_fixed
zprime_max = z_fixed_levels[1] - r_fixed
z_min = 0.6 + (zprime_min - 0.6)/cos45
z_max = 0.6 + (zprime_max - 0.6)/cos45

xs = np.arange(-a, a + res, res)
zs = np.arange(z_min, z_max + res, res)
X, Z = np.meshgrid(xs, zs, indexing='ij')
inside_domain_xz = (np.abs(X) <= a)
inside_domain_xz[0,:] = False
inside_domain_xz[-1,:] = False
inside_domain_xz[:,0] = False
inside_domain_xz[:,-1] = False
z_prime = 0.6 + (Z - 0.6) * cos45
outside_villus1_xz = (X-x1)**2 + (z_prime-0.6)**2 > r1**2
outside_villus2_xz = (X-x2)**2 + (z_prime-0.6)**2 > r2**2

outside_fixed_cylinders_xz = np.ones_like(inside_domain_xz, dtype=bool)
for zc in z_fixed_levels:
    for xc in x_fixed_positions:
        fixed_mask_xz = (X-xc)**2 + (z_prime - zc)**2 <= r_fixed**2
        outside_fixed_cylinders_xz &= ~fixed_mask_xz
mask_xz_aligned = inside_domain_xz & outside_villus1_xz & outside_villus2_xz & outside_fixed_cylinders_xz


edt_pixels_xz_aligned = ndi.distance_transform_edt(mask_xz_aligned)
edt_xz_aligned = edt_pixels_xz_aligned * res

skel_medial_xz_aligned, dist_pix_medial_xz_aligned = medial_axis(mask_xz_aligned, return_distance=True)
skel_dist_mm_medial_xz_aligned = dist_pix_medial_xz_aligned[skel_medial_xz_aligned] * res
mean_diameter_xz_aligned = 2 * np.mean(skel_dist_mm_medial_xz_aligned)
print(f"mean pore diameter in xz plane: {mean_diameter_xz_aligned:.6f} mm")

#distance map plot (x-z plane)
plt.figure(figsize=(6, 5))
plt.imshow(np.ma.masked_where(~mask_xz_aligned, edt_xz_aligned).T, origin='lower', extent=[xs[0], xs[-1] , zs[0], zs[-1]], cmap='viridis')
plt.colorbar(label="distance to nearest boundary [mm]")
plt.xlabel("x [mm]")
plt.ylabel("z [mm]")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"distance_map_xz_aligned_{i}.png"), dpi=300)
plt.show()

#distance map + medial axis skeleton plot (x-z plane)
plt.figure(figsize=(6, 5))
plt.imshow(np.ma.masked_where(~mask_xz_aligned, edt_xz_aligned).T, origin='lower', extent=[xs[0], xs[-1] , zs[0], zs[-1]], cmap='viridis')
plt.colorbar(label="distance to nearest boundary [mm]")
plt.contour(xs, zs, skel_medial_xz_aligned.T.astype(float), levels=[0.5], colors='red', linewidths=0.8)
plt.xlabel("x [mm]")
plt.ylabel("z [mm]")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"distance_map_plus_medial_xz_aligned_{i}.png"), dpi=300)
plt.show()
    
