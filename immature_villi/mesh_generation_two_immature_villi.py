import gmsh
import os
import meshio
import subprocess
import vtk

work_dir = os.getcwd()
b = 0.144
a = 2 * b
outer_height = 1.13
epsilon = 1e-4

for i in range(1, 26):
    step_file = os.path.join(work_dir, f"geometries_b{b:.3f}mm/{i}/Geometry", f"subtraction{i}.step")
    output_subdir = os.path.join(work_dir, f"geometries_b{b:.3f}mm/{i}")
    os.makedirs(output_subdir , exist_ok=True)
    output_vtk = os.path.join(output_subdir, "mesh_volume.vtk")
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.open(step_file)
    gmsh.model.occ.synchronize()
    
    inlet_tag = None
    outlet_tag = None
    candidates = []

    #identifying the inlet and outlet surfaces    
    for dim, tag in gmsh.model.getEntities(2):
        x, y, z = gmsh.model.occ.getCenterOfMass(dim,tag)
        if abs(z - outer_height) < epsilon:
            inlet_tag = tag
        elif abs(z) < epsilon:
            outlet_tag = tag
        else:
            candidates.append(tag)
    
    periphery_tags = []
    villi_tags = []

    #separating the remaining wall surfaces into periphery and villi    
    for tag in candidates:
        is_periphery = False
        for u, v in [(0.25,0.25), (0.5,0.25), (0.25,0.5)]:
            try:
                x, y, _ = gmsh.model.getValue(2, tag, [u,v])
                val = (x**2)/(a**2) + (y**2)/(b**2)
                if val > (1-epsilon):
                    is_periphery = True
                    break
            except:
                continue
        if is_periphery:
            periphery_tags.append(tag)
        else:
            villi_tags.append(tag)
            
    #setting a global mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),0.015)
    #refining mesh around villi
    for surf_tag in villi_tags:
        curves = gmsh.model.getBoundary([(2, surf_tag)], combined=False, oriented=False)
        for c in curves:
            pts = gmsh.model.getBoundary([c], combined=False, oriented=False)
            gmsh.model.mesh.setSize(pts , 0.00275)
    
    #applying distance-based mesh refinement around villi
    villi_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(villi_field, "FacesList", villi_tags)    
    thresh_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh_field,"InField", villi_field)
    gmsh.model.mesh.field.setNumber(thresh_field,"SizeMin", 0.00275)
    gmsh.model.mesh.field.setNumber(thresh_field,"SizeMax", 0.015)
    gmsh.model.mesh.field.setNumber(thresh_field,"DistMin",0.0)
    gmsh.model.mesh.field.setNumber(thresh_field,"DistMax", 0.02)
    gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)

    #generating and saving the 3D mesh    
    gmsh.model.mesh.generate(3)
    gmsh.write(output_vtk)
    gmsh.finalize()   
    print(f"\nMesh saved to: {output_vtk}")
    
    #remeshing the Gmsh-generated mesh with TetGen (because the direct Gmsh mesh output is not compatible with svFSI solver)
    mesh = meshio.read(output_vtk)
    ele_path = os.path.join(output_subdir, "mesh_volume.ele")
    meshio.write(ele_path, mesh, file_format="tetgen")
    tetgen_cmd = ["tetgen", "-k", ele_path]
    try:
        subprocess.run(tetgen_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("TetGen failed:", e)
    
    #converting .1.vtk to .vtu with added GlobalNodeIDs and GlobalElementIDs
    vtk1_path = os.path.join(output_subdir, "mesh_volume.1.vtk")
    mesh_complete_dir = os.path.join(output_subdir, "mesh-complete")
    vtu1_path = os.path.join(mesh_complete_dir, "mesh.1.vtu")
    os.makedirs(mesh_complete_dir, exist_ok=True)
    
    if os.path.exists(vtk1_path):
        reader = vtk.vtkDataSetReader()
        reader.SetFileName(vtk1_path)
        reader.Update()
        data = reader.GetOutput()
    
        num_points = data.GetNumberOfPoints()
        num_cells = data.GetNumberOfCells()
    
        global_node_ids = vtk.vtkIntArray()
        global_node_ids.SetName('GlobalNodeID')
        global_node_ids.SetNumberOfComponents(1)
        global_node_ids.SetNumberOfTuples(num_points)
        for j in range(num_points):
            global_node_ids.SetTuple1(j, j+1)
        data.GetPointData().AddArray(global_node_ids)
    
        global_elem_ids = vtk.vtkIntArray()
        global_elem_ids.SetName('GlobalElementID')
        global_elem_ids.SetNumberOfComponents(1)
        global_elem_ids.SetNumberOfTuples(num_cells)
        for j in range(num_cells):
            global_elem_ids.SetTuple1(j, j+1)
        data.GetCellData().AddArray(global_elem_ids)
    
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(data)
        writer.SetFileName(vtu1_path)
        writer.Write()
    
        #extracting and saving inlet, outlet, periphery wall, and villi wall surfaces from the final vtu mesh   
        nodes_region = vtk.vtkDoubleArray()
        nodes_region.SetName('nodes_region')
        nodes_region.SetNumberOfComponents(1)
        nodes_region.SetNumberOfTuples(num_points)
    
        for j in range(num_points):
            z = data.GetPoints().GetPoint(j)[2]
            if z >= outer_height - epsilon:
                nodes_region.SetTuple1(j, 0.0)
            elif z <= epsilon:
                nodes_region.SetTuple1(j, 3.0)
            else:
                nodes_region.SetTuple1(j, 9.0)
        data.GetPointData().AddArray(nodes_region)
    
        surface = vtk.vtkGeometryFilter()
        surface.SetInputData(data)
        surface.Update()
        surface_data = surface.GetOutput()
        surface_data.GetPointData().SetActiveScalars('nodes_region')
    
        def extract_and_save_region(surface_data, name, lower, upper, invert=False):
            t = vtk.vtkThreshold()
            t.SetInputData(surface_data)
            t.SetLowerThreshold(lower)
            t.SetUpperThreshold(upper)
            t.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
            if invert:
                t.InvertOn()
            t.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,'nodes_region')
            t.Update()   
            t_surf = vtk.vtkDataSetSurfaceFilter()
            t_surf.SetInputData(t.GetOutput())
            t_surf.Update()
    
            out = vtk.vtkXMLPolyDataWriter()
            out.SetInputData(t_surf.GetOutput())
            out.SetFileName(os.path.join(mesh_complete_dir, name))
            out.Write()
            print(f"{name} saved")
    
        extract_and_save_region(surface_data,'inlet.vtp', -0.5, 0.5)
        extract_and_save_region(surface_data,'outlet.vtp', 2.5, 3.5)
        extract_and_save_region(surface_data,'walls_combined.vtp', -0.5, 3.5, invert=True)
    
        walls_path = os.path.join(mesh_complete_dir,"walls_combined.vtp")
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(walls_path)
        reader.Update()
        walls = reader.GetOutput()
    
        points = walls.GetPoints()
        n_points = walls.GetNumberOfPoints()
    
        wall_type = vtk.vtkDoubleArray()
        wall_type.SetName('wall_type')
        wall_type.SetNumberOfComponents(1)
        wall_type.SetNumberOfTuples(n_points)
    
        per_count = vill_count = 0
        for j in range(n_points):
            x, y, _ = points.GetPoint(j)
            val = (x**2)/(a**2) + (y**2)/(b**2)
            if val > (1-epsilon):
                wall_type.SetTuple1(j, 1.0)  # periphery
                per_count += 1
            else:
                wall_type.SetTuple1(j, 2.0)  # villi
                vill_count += 1
    
        walls.GetPointData().AddArray(wall_type)
        walls.GetPointData().SetActiveScalars('wall_type')
    
        def extract_wall_type_region(walls_data, name, lower, upper):
            t = vtk.vtkThreshold()
            t.SetInputData(walls_data)
            t.SetLowerThreshold(lower)
            t.SetUpperThreshold(upper)
            t.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
            t.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,'wall_type')
            t.Update()
    
            surf = vtk.vtkDataSetSurfaceFilter()
            surf.SetInputData(t.GetOutput())
            surf.Update()
    
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetInputData(surf.GetOutput())
            writer.SetFileName(os.path.join(mesh_complete_dir, name))
            writer.Write()
            print(f"{name} saved")
    
        extract_wall_type_region(walls,'wall_periphery.vtp', 0.5, 1.5)
        extract_wall_type_region(walls,'wall_villi.vtp', 1.5, 2.5)
    else:
        print("TetGen .1.vtk output was not found for conversion to .vtu")
