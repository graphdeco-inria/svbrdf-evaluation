import bpy
import argparse
import sys, os
import math

def parse_args():
    # We only want the args after '--'
    argv = sys.argv

    if "--" in argv:
        argv = argv[argv.index("--") + 1:]  # Arguments after '--'

    # Set up argparse to handle the arguments
    parser = argparse.ArgumentParser(description="Blender Script with Arguments")
    parser.add_argument('--textures_dir', default="")
    parser.add_argument('--output_dir', default="")
    parser.add_argument('--blender_plyfile', default="")
    parser.add_argument('--envmaps_dir', default="")
    parser.add_argument('--method_name', default="")
    # parser.add_argument('--scene_type', default="", choices=["bedroom", "kitchen", "livingroom", "bathroom"])

    # Parse the arguments and return them
    return parser.parse_args(argv)

def generate_material(mat_name):
    # Create a new material for the object
    material = bpy.data.materials.new(name=mat_name)
    material.use_nodes = True
    # mesh_object.active_material = material

    # Get the material's node tree
    nodes = material.node_tree.nodes
    # links = material.node_tree.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)
    
    return material

def set_textures(textures_dir, nodes, links):

    # Add a new Principled BSDF shader node
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    bsdf.label = 'bsdf'

    # Add a Material Output node
    material_output = nodes.new(type="ShaderNodeOutputMaterial")
    material_output.location = (300, 0)
    material_output.label = 'mat_output'

    # Connect the BSDF to the Material Output node
    links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    # Load the Albedo (Base Color) texture
    albedo_texture = nodes.new(type="ShaderNodeTexImage")
    albedo_texture.image = bpy.data.images.load(os.path.join(textures_dir, "albedo.png"))
    albedo_texture.location = (-300, 0)
    albedo_texture.label = 'albedo'
    links.new(albedo_texture.outputs['Color'], bsdf.inputs['Base Color'])
    
    # Metallic
    metallic_texture = nodes.new(type="ShaderNodeTexImage")
    metallic_texture.image = bpy.data.images.load(os.path.join(textures_dir, "metallic.png"))
    metallic_texture.location = (-300, -300)
    metallic_texture.label='metallic'
    # metallic_texture.image.colorspace_settings.name = 'Non-Color'
    links.new(metallic_texture.outputs['Color'], bsdf.inputs['Metallic'])

    # Roughness
    roughness_texture = nodes.new(type="ShaderNodeTexImage")
    roughness_texture.image = bpy.data.images.load(os.path.join(textures_dir, "roughness.png"))
    roughness_texture.location = (-300, -600)
    roughness_texture.label='roughness'
    # roughness_texture.image.colorspace_settings.name = 'Non-Color'
    sqrt_node = nodes.new(type="ShaderNodeMath")
    sqrt_node.location = (100, -600)
    sqrt_node.operation = "SQRT"
    sqrt_node.label='sqrt'

    links.new(roughness_texture.outputs['Color'], sqrt_node.inputs['Value'])
    links.new(sqrt_node.outputs['Value'], bsdf.inputs['Roughness'])


# Create a new world and set environment texture
def set_world_with_envmaps(envmap_paths):
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    
    # Get the node tree of the world shader
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)


    # Texture coordinates node
    tex_coord_node = nodes.new(type="ShaderNodeTexCoord")
    tex_coord_node.location = (-900, 0)

    # Add 3 Mapping nodes
    mapping_nodes = {}
    for i in range(3):
        name = os.path.basename(envmap_paths[i])[:-4]
        mapping_node = nodes.new(type="ShaderNodeMapping")
        mapping_node.location = (-600, i * 400)  # Offset vertically for each Mapping node
        mapping_nodes[name] = mapping_node

    # VERY SPECIFIC TO KITCHEN WORKING WELL ROTATIONS FOR ENVMAPS
    mapping_nodes["courtyard"].inputs["Rotation"].default_value = (0, 0, math.radians(-151))
    mapping_nodes["sunset"].inputs["Rotation"].default_value = (0, 0, math.radians(220.6))
    mapping_nodes["interior"].inputs["Rotation"].default_value = (0, 0, math.radians(242))


    offset = 0
    env_nodes = {}
    for map in envmap_paths:
        # Add Environment Texture node
        name = os.path.basename(map)[:-4]
        print(f"envmap label: {name}")
        env_tex_node = nodes.new(type="ShaderNodeTexEnvironment")
        env_tex_node.image = bpy.data.images.load(map) 
        env_tex_node.location = (-300, 300*offset)
        env_tex_node.label = name
        offset+=1
        env_nodes[name] = env_tex_node

    # Add Background node
    background_node = nodes.new(type="ShaderNodeBackground")
    background_node.location = (0, 0)

    # Connect Texture Coordinates node to all 3 Mapping nodes
    for key in mapping_nodes:
        mapping_node = mapping_nodes[key]
        links.new(tex_coord_node.outputs['Generated'], mapping_node.inputs['Vector'])
        links.new(mapping_node.outputs['Vector'], env_nodes[key].inputs["Vector"])

    # Add Output node
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    output_node.location = (300, 0)

    links.new(env_nodes["courtyard"].outputs['Color'], background_node.inputs['Color'])
    links.new(background_node.outputs['Background'], output_node.inputs['Surface'])

def set_camera(transform, focal_length):
    # Add a camera object
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    
    camera.location = transform['location']
    camera.rotation_euler = transform['rotation']

    # Set camera field of view (FOV)
    camera_data.lens = focal_length

    # Set the active camera
    bpy.context.scene.camera = camera
    
if __name__ == "__main__":
    args = parse_args()
    output_scene_path = os.path.join(args.output_dir, f"textured_{args.method_name}.blend")
    methods = os.listdir(args.textures_dir)
    
    # Clear existing mesh data (optional)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import the PLY mesh
    bpy.ops.wm.ply_import(filepath=args.blender_plyfile)
    mesh_object = bpy.context.selected_objects[0]  # Get the imported mesh
    mesh_object.name = "mesh"
    
    # mesh_name = os.path.basename(args.ply_file)[:-4]
    # mesh_object.rotation_euler[2] = math.radians(180) # messed up the first time, kitchen SHOULD be rotated 180deg
    
    # set shade smooth
    with bpy.context.temp_override(selected_editable_objects=[bpy.data.objects['mesh']]):
        bpy.ops.object.shade_smooth()

    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024

    materials = {}
    # for method in methods:
    method = args.method_name
    print(f"texture {method}")
    textures = os.path.join(args.textures_dir, method)
    material = generate_material(method)
    set_textures(os.path.abspath(textures), nodes=material.node_tree.nodes, links=material.node_tree.links)
    materials[method] = material
    mesh_object.data.materials.append(material)

    bpy.ops.wm.save_as_mainfile(filepath=output_scene_path)
    print(f"Blender scene saved here > {output_scene_path}.")