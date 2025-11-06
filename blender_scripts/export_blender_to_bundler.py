import bpy
import mathutils
import argparse
import sys, os

def parse_args():
    # We only want the args after '--'
    argv = sys.argv

    if "--" in argv:
        argv = argv[argv.index("--") + 1:]  # Arguments after '--'

    # Set up argparse to handle the arguments
    parser = argparse.ArgumentParser(description="Blender Script with Arguments")
    parser.add_argument('--blender_file', default="") 
    parser.add_argument('--output_dir', default="")

    # Parse the arguments and return them
    return parser.parse_args(argv)

def load_blend_file(filepath):
    """Load a .blend file."""
    # Ensure that the current file is saved if needed
    if bpy.data.is_dirty:
        print("Current file has unsaved changes. Consider saving before loading a new file.")
    
    # Load the .blend file
    bpy.ops.wm.open_mainfile(filepath=filepath)
    print(f"Loaded .blend file: {filepath}")

def get_cameras_params(width):
    all_cameras_params = []
    cameras = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']
    cameras_sorted = sorted(cameras, key=lambda cam : cam.name)
    print("found:", len(cameras_sorted), "cameras")
    for cam in cameras_sorted:
        # print(cam.name)
        # Get the camera data
        cam_data = cam.data

        fx = cam_data.lens * width / cam_data.sensor_width
        cam_rot = cam.matrix_world.to_3x3().transposed()

        T = mathutils.Vector(cam.location)
        T1 = -(cam_rot @ T)
        
        tx = T1[0]
        ty = T1[1]
        tz = T1[2]
        
        # Gather camera parameters
        camera_params = {
            "name": cam.name,
            "fx": fx,
            "rotation": cam_rot,
            "translation": (tx, ty, tz)
        }
        
        all_cameras_params.append([camera_params])
    
    return all_cameras_params  

def export_dataset(dirpath, cameras_params):
    output_dir = dirpath
    cameras_file = os.path.join(output_dir, 'bundler.out')

    os.makedirs(output_dir, exist_ok=True)

    with open(cameras_file, 'w') as f_cam:
        f_cam.write(f'# Bundle file v0.3\n')
        f_cam.write(f'{len(cameras_params)} 0\n')
        
        for params in cameras_params:
            t = params[0]["translation"]
            r0 = params[0]["rotation"][0]
            r1 = params[0]["rotation"][1]
            r2 = params[0]["rotation"][2]
            
            f_cam.write(f'{params[0]["fx"]} 0 0\n')
            f_cam.write(f'{r0.x} {r0.y} {r0.z}\n')
            f_cam.write(f'{r1.x} {r1.y} {r1.z}\n')
            f_cam.write(f'{r2.x} {r2.y} {r2.z}\n')
            f_cam.write(f'{t[0]} {t[1]} {t[2]}\n')

if __name__ == "__main__":
    args = parse_args()
    
    load_blend_file(args.blender_file)
    scene = bpy.context.scene
    width = scene.render.resolution_x    
    cameras_params = get_cameras_params(width)
    export_dataset(args.output_dir, cameras_params)
    print(f"dataset exported in bundler format here > {args.output_dir}.")