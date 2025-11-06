import pymeshlab
import argparse
import subprocess
import sys, os
import numpy as np
from plyfile import PlyData, PlyElement
import cv2
from shutil import rmtree
import time

def blender_to_meshlab_ply(header, vertices, faces):
    np_faces = np.array(faces)
    texcoords = np.array(vertices)[:, 3:5]
    used_uvs = texcoords[np_faces[:, 1:]]
    used_uvs = np.reshape(used_uvs, (used_uvs.shape[0], 6))
    nb = 6 * np.ones((used_uvs.shape[0], 1)) # 6 coordinates

    used_uvs = np.hstack((nb, used_uvs))
    out_faces = np.hstack((np_faces, used_uvs))
    header.append('property list uchar float texcoord')

    # print(f"header {header}")
    return header, out_faces

def convert_ply_to_meshlab(in_ply_path, out_ply_path):
    plydata = PlyData.read(in_ply_path)
    vertices = plydata['vertex']
    s = np.array(vertices['s'])
    t = np.array(vertices['t'])
    s = np.reshape(s, (s.shape[0], 1))  
    t = np.reshape(t, (t.shape[0], 1))
    uvs = np.hstack((s, t))
    vertex_indices = np.stack(plydata['face']['vertex_indices'])
    ordered_uvs = uvs[vertex_indices]
    ordered_uvs = np.reshape(ordered_uvs, (ordered_uvs.shape[0], 6))

    dtype = [('vertex_indices', 'i4', (3,)), ('texcoord', 'f4', (6,))]
    elements = np.empty(vertex_indices.shape[0], dtype=dtype)
    elements['vertex_indices'] = vertex_indices
    elements['texcoord'] = ordered_uvs
    updated_faces = PlyElement.describe(elements, 'face')

    PlyData([vertices, updated_faces]).write(out_ply_path)

def write_ply(filename, header, vertices, faces):
    with open(filename, 'w') as file:
        # Write header
        for line in header:
            file.write(f"{line}\n")
        file.write("end_header\n")

        # Write vertices
        for vertex in vertices:
            file.write(" ".join(map(str, vertex)) + "\n")

        # Write faces
        for face in faces:
            indices = " ".join(map(str, face[:5].astype(np.uint32)))
            values = " ".join(map(str, face[5:])) + "\n"
            file.write(indices + " " + values)

def setup_tmp_files(svbrdf_dir, output_dir, method):
    
    os.makedirs(os.path.join(output_dir, "tmp", method), exist_ok=True)

    albedo_str = ""
    roughness_str = ""
    metallic_str = ""
    for image_name in [img for img in os.listdir(svbrdf_dir) if img.endswith('_svbrdf.png')]:
        
        albedo_path = os.path.abspath(os.path.join(output_dir, "tmp", method, 'albedo_' + image_name))
        roughness_path = os.path.abspath(os.path.join(output_dir, "tmp", method, 'roughness_' + image_name))
        metallic_path = os.path.abspath(os.path.join(output_dir, "tmp", method, 'metallic_' + image_name))
        
        stacked_maps = cv2.imread(os.path.join(svbrdf_dir, image_name), -1)
        offset = stacked_maps.shape[0]
        
        cv2.imwrite(albedo_path, stacked_maps[:, offset:2*offset, :])       # albedo
        cv2.imwrite(roughness_path, stacked_maps[:, 2*offset:3*offset, :])  # roughness
        cv2.imwrite(metallic_path, stacked_maps[:, 3*offset:, :])           # metallic

        albedo_str += albedo_path + '\n'
        roughness_str += roughness_path + '\n'
        metallic_str += metallic_path + '\n'
    
    with open(os.path.join(output_dir, "tmp", method, "albedo.txt"), "w") as file:
        file.write(albedo_str)
    with open(os.path.join(output_dir, "tmp", method, "roughness.txt"), "w") as file:
        file.write(roughness_str)
    with open(os.path.join(output_dir, "tmp", method, "metallic.txt"), "w") as file:
        file.write(metallic_str)

    return [os.path.join(output_dir, "tmp")]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--blender_exe', default="C:/Program Files/Blender Foundation/Blender 4.2/blender.exe") 
  
    parser.add_argument('--scene_dir', default="./data/bathroom/") 
    parser.add_argument('--svbrdf_dir', default="./output/bathroom") 
    parser.add_argument('--output_dir', default="./output/rustic_1200_output") 

    parser.add_argument('--meshlab_plyfile', default="bathroom_meshlab.ply") 
    parser.add_argument('--blender_plyfile', default="bathroom_blender.ply") 
    parser.add_argument('--bundler_file', default="bundler.out") 
    parser.add_argument('--method_name', default="UNet_HF") 

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    begin = time.time()

    converted_ply_path = os.path.abspath(os.path.join(args.scene_dir, args.meshlab_plyfile))
    bundler_cams = os.path.join(args.scene_dir, args.bundler_file).replace("\\", "/")
    ms = pymeshlab.MeshSet()
    texture_names = ["texture.png"]
    
    print(f"creating temporary files.")
    tmp_dirs = setup_tmp_files(args.svbrdf_dir, args.output_dir, args.method_name)

    method = args.method_name
    os.makedirs(os.path.join(args.output_dir, "textures", method), exist_ok=True)
    
    image_lists = [os.path.join("tmp", method, list_name) for list_name in os.listdir(os.path.join(args.output_dir, "tmp", method)) if list_name.endswith('.txt')]
    texture_names = [os.path.basename(tex_name).replace(".txt", ".png") for tex_name in image_lists]
    
    for image_list, texture_name in zip(image_lists, texture_names):
        ms.load_project([os.path.abspath(bundler_cams), os.path.abspath(os.path.join(args.output_dir, image_list))]) # load bundler
        ms.load_new_mesh(converted_ply_path)
        print("Computing colors from projected images.")
        ms.compute_color_and_texture_from_active_rasters_projection(texsize = 4096, deptheta = 0.010000, textname=texture_name)
        pymeshlab.pmeshlab.Mesh.texture(ms.current_mesh(), 0).save(os.path.join(args.output_dir, "textures", method, texture_name))
        print(f"Texture saved here > {os.path.join(args.output_dir, 'textures', method, texture_name)}.")
        ms.clear()

    print(f"cleaning temporary files.")
    for tmp_dir in tmp_dirs:
        rmtree(tmp_dir)

    #########
    # Generate scene.blend file that has the textured mesh.
    #########

    # GENERATE a material per method (each material has albedo metallic roughness)
    print("launching create_pbr_scene.py ....")

    if not os.path.exists(args.blender_exe):
        print("Could not find Blender executable at:", args.blender_exe)
        print("Exiting")
        exit(0)
    
    ver = os.path.dirname(args.blender_exe)[-3:]
    envmaps_dir = os.path.join(os.path.dirname(args.blender_exe), ver, "datafiles/studiolights/world")
    
    conversion_args = ["--background", "--python", "blender_scripts/create_pbr_scene.py", "--",
                        # "--blender_file", os.path.abspath(os.path.join(args.scene_dir, args.blender_file)), 
                        "--textures_dir", os.path.abspath(os.path.join(args.output_dir, 'textures')), 
                        "--output_dir", os.path.abspath(args.output_dir), 
                        "--method_name", method, 
                        "--blender_plyfile", os.path.abspath(os.path.join(args.scene_dir, args.blender_plyfile)), 
                        "--envmaps_dir", os.path.abspath(envmaps_dir), 
                        ]
    try:
        subprocess.run([args.blender_exe] + conversion_args, check=True) 
    except subprocess.CalledProcessError as e:
        print(f"Error executing create_pbr_scene.py: {e}")
        sys.exit(1)
        
    end = time.time()
    print(f"{end - begin} elapsed time")
