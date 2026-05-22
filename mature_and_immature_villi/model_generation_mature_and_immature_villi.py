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
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.Geom import Geom_Ellipse
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Extend.DataExchange import write_step_file
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from scipy.interpolate import splprep, splev, interp1d
import os
import scipy.ndimage as ndi
from skimage.morphology import medial_axis


b = 0.135 #ellipse semi-minor axis (mm) 
i = 1    # The ID of the sample
output_dir = f"geometries_b{b:.3f}mm/{i}/Geometry"
os.makedirs(output_dir, exist_ok=True)

a = 2 * b #ellipse semi-major axis, aspect ratio has been taken to be 2 (explained in the paper) 
height = 1.16  #height of the elliptical cylinder (mm)
z_slice = 0.6  # in mm      
gap_shape = 5.14  #shape parameter of the gamma probability density function
gap_scale = 0.0148 # scale parameter of the gamma probability density function
min_clearance = 0.01 # in mm


# defining the z coordinate (along the elliptical tube height) of the villi path points and the corresponding villi radius in mm
z_main = np.array([-0.01, 0.11, 0.19, 0.29, 0.4, 0.45, 0.5, 0.55, 0.6, 0.66, 0.72, 0.77, 0.85, 0.91, 0.96, 1.01, 1.08])
r_main = np.array([0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.07, 0.06, 0.055, 0.04])

z_branch = np.array([0.29, 0.38, 0.46, 0.50, 0.60, 0.65, 0.70, 0.80, 0.87, 0.96, 1.02, 1.11])
r_branch = np.array([0.035, 0.045, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.045, 0.045, 0.045])

idx_main = np.where(np.isclose(z_main, z_slice))[0][0]
r_main_at_z = r_main[idx_main]
idx_branch = np.where(np.isclose(z_branch, z_slice))[0][0]
r_branch_at_z = r_branch[idx_branch]

phi = np.linspace(0, 2*np.pi, 360, endpoint=False)
ellipse_points = np.column_stack((a*np.cos(phi), b*np.sin(phi)))

def is_clear_from_ellipse(center , radius):
    dists = np.linalg.norm(ellipse_points - center, axis=1)
    return np.all(dists - radius >= min_clearance)

#defining the location of the main villus center at z_slice
for attempt in range(1000):
    gap1 = np.random.gamma(gap_shape, gap_scale)
    theta = np.random.uniform(0, 2*np.pi)
    wall_pt = np.array([a*np.cos(theta), b*np.sin(theta)])
    normal = np.array([wall_pt[0] / a**2, wall_pt[1] / b**2])
    normal /= np.linalg.norm(normal)
    center1 = wall_pt - (gap1 + r_main_at_z) * normal
    if (center1[0]/a)**2 + (center1[1]/b)**2 <= 1 and is_clear_from_ellipse(center1, r_main_at_z):
        x_rand_m , y_rand_m = center1
        break
else:
    raise RuntimeError("Failed to place the first villus, please run again.")

#defining the location of the branch villus center at z_slice
gap2 = np.random.gamma(gap_shape, gap_scale)
dist2 = r_main_at_z + r_branch_at_z + gap2
for attempt in range(1000):
    theta2 = np.random.uniform(0, 2*np.pi)
    delta = dist2 * np.array([np.cos(theta2), np.sin(theta2)])
    center2 = np.array([x_rand_m , y_rand_m]) + delta
    if (center2[0]/a)**2 + (center2[1]/b)**2 <= 1 and is_clear_from_ellipse(center2, r_branch_at_z):
        x_rand_b , y_rand_b = center2
        break
else:
    raise RuntimeError("Failed to place the second villus, please run again.")

# No path smoothing with s = 0 and the greater the s, the stronger would be the path smoothing effect 
def smooth_path(points, s=0.0, num=100): 
    coords = np.array([[p.X(), p.Y(), p.Z()] for p in points]).T
    tck, _ = splprep(coords, s=s)
    new_points = splev(np.linspace(0,1, num), tck)
    return [gp_Pnt(x, y, z) for x, y, z in zip(*new_points)]

#villus radius interpolation function
def interpolate_radii(radii , num):
    x = np.arange(len(radii))
    f = interp1d(x, radii, kind='cubic')
    return f(np.linspace(0, len(radii)-1, num))

# function for lofting the villus
def loft_solid(path_pts , radii):
    wires = []
    for i, pt in enumerate(path_pts):
        if i == 0:
            T = gp_Vec(path_pts[1], pt)
        elif i == len(path_pts) - 1:
            T = gp_Vec(path_pts[-2], pt)
        else:
            T = gp_Vec(path_pts[i-1], path_pts[i+1])
        ax = gp_Ax2(pt, gp_Dir(T))
        circ = GC_MakeCircle(ax, radii[i]).Value()
        edge = BRepBuilderAPI_MakeEdge(circ).Edge()
        wire = BRepBuilderAPI_MakeWire(edge).Wire()
        wires.append(wire)
    loft = BRepOffsetAPI_ThruSections(True , False)
    for wire in wires:
        loft.AddWire(wire)
    loft.Build()
    return loft.Shape()

# function for building fillet at the bottom of the villus
def fillet_outlet(solid, last_pt, radius):
    face_exp = TopExp_Explorer(solid, TopAbs_FACE)
    min_dist = float('inf')
    target_face = None
    while face_exp.More():
        face = face_exp.Current()
        surf = BRep_Tool.Surface(face)
        umin, umax, vmin, vmax = surf.Bounds()
        mid = surf.Value((umin+umax) / 2, (vmin+vmax) / 2)
        d = mid.Distance(last_pt)
        if d < min_dist:
            min_dist, target_face = d , face
        face_exp.Next()
    fillet = BRepFilletAPI_MakeFillet(solid)
    edge_exp = TopExp_Explorer(target_face, TopAbs_EDGE)
    while edge_exp.More():
        fillet.Add(radius , edge_exp.Current())
        edge_exp.Next()
    fillet.Build()
    return fillet.Shape()

#defining the rest of the main villus path points x and y coordinates based on the villus randomly obtained location at z = z_slice
x_main = np.array([x_rand_m + d for d in [0.07, 0.07, 0.06, 0.06, 0.06, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, -0.005, -0.005, -0.01 , -0.01, -0.015, -0.01]])
y_main = np.full_like(x_main, y_rand_m)

#constructing the main villus
main_path = [gp_Pnt(x, y, z) for x, y, z in zip(x_main, y_main, z_main)]
main_path_smooth = smooth_path(main_path, s=0.001 , num=60)
main_radii_smooth = interpolate_radii(r_main, 60)
main_solid = loft_solid(main_path_smooth, main_radii_smooth)
main_fillet = fillet_outlet(main_solid, main_path_smooth[-1], radius=0.037)

#defining the rest of the branch villus path points x and y coordinates based on the villus randomly obtained location at z = z_slice
x_ms = x_main[3]
x_branch = np.array([x_ms, 0.75 * x_ms + 0.25 * x_rand_b, 0.5 * x_ms + 0.5 * x_rand_b, 0.25 * x_ms + 0.75 * x_rand_b, x_rand_b, x_rand_b + 0.01, x_rand_b, x_rand_b + 0.01,
    x_rand_b, x_rand_b + 0.015, x_rand_b + 0.01, x_rand_b + 0.015])
y_branch = np.array([y_rand_m , 0.75 * y_rand_m + 0.25 * y_rand_b, 0.5 * y_rand_m + 0.5 * y_rand_b, 0.25 * y_rand_m + 0.75 * y_rand_b] + [y_rand_b] * 8)

#constructing the branch villus
branch_path = [gp_Pnt(x, y, z) for x, y, z in zip(x_branch, y_branch, z_branch)]
branch_path_smooth = smooth_path(branch_path, s=0.0 , num=60)
branch_radii_smooth = interpolate_radii(r_branch, 60)
branch_solid = loft_solid(branch_path_smooth, branch_radii_smooth)
branch_fillet = fillet_outlet(branch_solid, branch_path_smooth[-1], radius=0.037)

#for the junction fillet
def get_edges(shape):
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    edges = []
    while exp.More():
        edges.append(exp.Current())
        exp.Next()
    return edges

def edge_exists(edge, edge_list):
    return any(edge.IsSame(e) or edge.IsEqual(e) for e in edge_list)

def get_edge_midpoint(edge):
    curve = BRepAdaptor_Curve(edge)
    u1, u2 = curve.FirstParameter() , curve.LastParameter()
    return curve.Value(0.5 * (u1+u2))

def smooth_junction(main_fillet, branch_fillet, fused_shape, z_min=0.25, z_max=0.55 , radius=0.002):
    edges_main = get_edges(main_fillet)
    edges_branch = get_edges(branch_fillet)
    edges_combined = get_edges(fused_shape)
    junction_edges = []
    for edge in edges_combined:
        if not edge_exists(edge, edges_main+ edges_branch):
            midpoint = get_edge_midpoint(edge)
            if z_min <= midpoint.Z()<= z_max:
                junction_edges.append(edge)
    fillet = BRepFilletAPI_MakeFillet(fused_shape)
    for edge in junction_edges:
        fillet.Add(radius,edge)
    fillet.Build()
    return fillet.Shape()

#fusing and smoothing the junction
fused = BRepAlgoAPI_Fuse(main_fillet, branch_fillet).Shape()
smoothed = smooth_junction(main_fillet, branch_fillet , fused, z_min=0.25, z_max=0.55, radius=0.02)

write_step_file(smoothed, os.path.join(output_dir, f"villi{i}.step"))

#function for building the elliptical cylinder 
def make_elliptical_cylinder(a, b, height):
    ax2 = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    ellipse = Geom_Ellipse(ax2, a, b)
    edge = BRepBuilderAPI_MakeEdge(ellipse).Edge()
    wire = BRepBuilderAPI_MakeWire(edge).Wire()
    face = BRepBuilderAPI_MakeFace(wire).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(0,0, height)).Shape()

elliptical_cylinder = make_elliptical_cylinder(a, b, height)
cut_result = BRepAlgoAPI_Cut(elliptical_cylinder, smoothed).Shape() #subtracting the villi from elliptical cylinder

write_step_file(cut_result, os.path.join(output_dir, f"subtraction{i}.step"))

#distance map and skeleton at z = 0.6
res = 0.001
xs = np.arange(-a, a + res, res)
ys = np.arange(-b, b + res, res)
X, Y = np.meshgrid(xs, ys, indexing='ij')
inside_ellipse = (X/a)**2 + (Y/b)**2 < 1
outside_main   = (X-x_rand_m)**2 + (Y-y_rand_m)**2 >= r_main_at_z**2
outside_branch = (X-x_rand_b)**2 + (Y-y_rand_b)**2 >= r_branch_at_z**2
mask = inside_ellipse & outside_main & outside_branch

edt_pixels = ndi.distance_transform_edt(mask)
edt = edt_pixels * res

skel_medial, dist_pix_medial = medial_axis(mask, return_distance=True)
skel_dist_mm_medial = dist_pix_medial[skel_medial] * res
mean_diameter = 2 * np.mean(skel_dist_mm_medial)
print(f"mean pore diameter at z = {z_slice}: {mean_diameter:.6f} mm")

#distance map plot
plt.figure(figsize=(6, 5))
plt.imshow(np.ma.masked_where(~mask, edt).T, origin='lower', extent=[-a, a, -b, b], cmap='viridis')
plt.colorbar(label="distance to nearest boundary [mm]")
plt.title(f"distance map at z = {z_slice}")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"distance_map_z0.6_{i}.png"), dpi=300)
plt.show()

#distance map + medial axis skeleton plot
plt.figure(figsize=(6, 5))
plt.imshow(np.ma.masked_where(~mask, edt).T, origin='lower', extent=[-a, a, -b, b], cmap='viridis')
plt.colorbar(label="distance to nearest boundary [mm]")
plt.contour(xs, ys, skel_medial.T.astype(float), levels=[0.5], colors='red', linewidths=0.8)
plt.title(f"distance map + medial axis skeleton at z = {z_slice}")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"distance_map_plus_medial_skeleton_z0.6_{i}.png"), dpi=300)
plt.show()
