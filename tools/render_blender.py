import bpy
import os
import sys
import argparse
import math

def parse_args():
    # Args are after '--'
    args = []
    if '--' in sys.argv:
        args = sys.argv[sys.argv.index('--') + 1:]
    
    parser = argparse.ArgumentParser(description="Blender batch renderer for WaLa evaluation")
    parser.add_argument("--shapes_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--materials", type=str, default=None,
                        help="Liste de matériaux séparés par des virgules. Par défaut les 5. "
                             "Pour un dataset d'entraînement, 'lambertian_textured' suffit : "
                             "c'est celui où WaLa réussit le mieux (94.6%% de F1 sur la sphère), "
                             "donc celui qui donne une baseline image réellement fonctionnelle.")
    parser.add_argument("--suffix_material", action="store_true", default=False,
                        help="Suffixer le nom de fichier par le matériau. Sans ce drapeau et "
                             "avec un seul matériau, le fichier s'appelle <objet>.png.")
    return parser.parse_args(args)

def setup_scene():
    # Delete default collection objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Create camera
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # Camera position: angled front-top view
    cam_obj.location = (2.2, -2.2, 1.8)
    # Point at origin
    # Camera rotation using euler angles
    cam_obj.rotation_mode = 'XYZ'
    # Camera looks towards [0,0,0]. Let's compute euler angles to point at [0,0,0] from [2.2, -2.2, 1.8]
    # theta = atan2(y, x), phi = atan2(z, sqrt(x^2+y^2))
    # Roughly: rx = 60 deg, ry = 0, rz = 45 deg
    cam_obj.rotation_euler = (math.radians(60), 0, math.radians(45))
    
    # Create lights (studio rig)
    # 1. Key Light
    key_light_data = bpy.data.lights.new(name="Key_Light", type='POINT')
    key_light_data.energy = 250.0  # Point light energy in Watts
    key_obj = bpy.data.objects.new(name="Key_Light", object_data=key_light_data)
    key_obj.location = (3.0, -3.0, 4.0)
    bpy.context.scene.collection.objects.link(key_obj)
    
    # 2. Fill Light
    fill_light_data = bpy.data.lights.new(name="Fill_Light", type='POINT')
    fill_light_data.energy = 80.0
    fill_obj = bpy.data.objects.new(name="Fill_Light", object_data=fill_light_data)
    fill_obj.location = (-3.0, -2.0, 2.0)
    bpy.context.scene.collection.objects.link(fill_obj)
    
    # 3. Rim/Back Light
    back_light_data = bpy.data.lights.new(name="Back_Light", type='POINT')
    back_light_data.energy = 150.0
    back_obj = bpy.data.objects.new(name="Back_Light", object_data=back_light_data)
    back_obj.location = (-1.0, 4.0, 3.0)
    bpy.context.scene.collection.objects.link(back_obj)

    # Set render properties
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'  # Use Cycles for high-quality transparent glass/refraction
    scene.cycles.device = 'CPU'     # Safe default for background headless
    scene.cycles.samples = 64       # High enough for clean render with denoising, low enough for speed
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.film_transparent = False # We want white background
    
    # Background color to white
    world = scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        bg_node.inputs['Strength'].default_value = 1.0

def set_node_value(node, name, val):
    for socket in node.inputs:
        if name.lower() in socket.name.lower():
            socket.default_value = val
            return True
    return False

def create_lambertian_textured():
    mat = bpy.data.materials.new(name="Lambertian_Textured")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_bsdf.location = (0, 0)
    set_node_value(node_bsdf, "Roughness", 0.8)
    set_node_value(node_bsdf, "Metallic", 0.0)
    
    # Checkerboard texture
    node_checker = nodes.new(type='ShaderNodeTexChecker')
    node_checker.location = (-300, 0)
    node_checker.inputs['Scale'].default_value = 12.0
    node_checker.inputs['Color1'].default_value = (0.85, 0.15, 0.15, 1.0) # Red
    node_checker.inputs['Color2'].default_value = (0.15, 0.15, 0.85, 1.0) # Blue
    
    node_coord = nodes.new(type='ShaderNodeTexCoord')
    node_coord.location = (-500, 0)
    
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    node_out.location = (300, 0)
    
    links.new(node_coord.outputs['Generated'], node_checker.inputs['Vector'])
    links.new(node_checker.outputs['Color'], node_bsdf.inputs['Base Color'])
    links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])
    return mat

def create_uniform_matte():
    mat = bpy.data.materials.new(name="Uniform_Matte")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    set_node_value(node_bsdf, "Base Color", (0.7, 0.7, 0.7, 1.0))
    set_node_value(node_bsdf, "Roughness", 0.8)
    set_node_value(node_bsdf, "Metallic", 0.0)
    
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])
    return mat

def create_specular_metallic():
    mat = bpy.data.materials.new(name="Specular_Metallic")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    set_node_value(node_bsdf, "Base Color", (0.9, 0.9, 0.95, 1.0)) # Silver chrome
    set_node_value(node_bsdf, "Roughness", 0.05)
    set_node_value(node_bsdf, "Metallic", 1.0)
    
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])
    return mat

def create_transparent_glass():
    mat = bpy.data.materials.new(name="Transparent_Glass")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    set_node_value(node_bsdf, "Base Color", (1.0, 1.0, 1.0, 1.0))
    set_node_value(node_bsdf, "Roughness", 0.02)
    # Transmission Weight (Blender 4.0+) or Transmission (Blender 3.x)
    set_node_value(node_bsdf, "Transmission", 1.0)
    set_node_value(node_bsdf, "Transmission Weight", 1.0)
    set_node_value(node_bsdf, "IOR", 1.5)
    
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])
    return mat

def create_translucent_frosted():
    mat = bpy.data.materials.new(name="Translucent_Frosted")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    set_node_value(node_bsdf, "Base Color", (0.9, 0.95, 1.0, 1.0)) # Slightly bluish
    set_node_value(node_bsdf, "Roughness", 0.3)                    # High roughness for frosted glass
    set_node_value(node_bsdf, "Transmission", 0.9)                 # Faintly transparent
    set_node_value(node_bsdf, "Transmission Weight", 0.9)
    set_node_value(node_bsdf, "IOR", 1.45)
    # Subsurface weight for translucency SSS effect
    set_node_value(node_bsdf, "Subsurface Weight", 0.3)
    set_node_value(node_bsdf, "Subsurface Scale", 0.1)
    
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])
    return mat

def load_and_normalize_object(filepath):
    # Import OBJ
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=filepath)
    else:
        bpy.ops.import_scene.obj(filepath=filepath)
    
    # Get imported object (should be active or the only mesh)
    imported_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not imported_objs:
        raise ValueError(f"Failed to import mesh from {filepath}")
    
    # Combine if multiple meshes imported
    bpy.ops.object.select_all(action='DESELECT')
    for obj in imported_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported_objs[0]
    if len(imported_objs) > 1:
        bpy.ops.object.join()
        
    obj = bpy.context.view_layer.objects.active
    
    # Center object origin at geometry center
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)
    
    # Normalize dimensions (fit within a bounding box of size 1.3)
    dims = obj.dimensions
    max_dim = max(dims)
    if max_dim > 0:
        scale_factor = 1.3 / max_dim
        obj.scale = (scale_factor, scale_factor, scale_factor)
        
    # Apply transformation
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    return obj

def main():
    args = parse_args()
    shapes_dir = args.shapes_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    setup_scene()
    
    # Create materials
    materials = {
        "lambertian_textured": create_lambertian_textured(),
        "uniform_matte": create_uniform_matte(),
        "specular_metallic": create_specular_metallic(),
        "transparent_glass": create_transparent_glass(),
        "translucent_frosted": create_translucent_frosted(),
    }
    
    if args.materials:
        keep = [m.strip() for m in args.materials.split(",")]
        inconnus = [m for m in keep if m not in materials]
        if inconnus:
            raise SystemExit(f"[Blender] Matériaux inconnus : {inconnus}. "
                             f"Disponibles : {list(materials)}")
        materials = {k: v for k, v in materials.items() if k in keep}
    print(f"[Blender] Matériaux rendus : {list(materials)}")

    # Find shape OBJ files
    shape_files = sorted([f for f in os.listdir(shapes_dir) if f.endswith(".obj")])
    print(f"[Blender] Found {len(shape_files)} shapes to render.")
    
    for f in shape_files:
        shape_name = os.path.splitext(f)[0]
        filepath = os.path.join(shapes_dir, f)
        print(f"[Blender] Rendering shape: {shape_name}")
        
        # Load object
        obj = load_and_normalize_object(filepath)
        
        # Render for each material
        for mat_name, mat in materials.items():
            # Assign material. On vide d'abord tous les emplacements : remplacer
            # seulement le premier laisse en place les matériaux propres au maillage
            # quand il en porte plusieurs, et le rendu sort alors avec la texture
            # d'origine au lieu du damier. Mesuré sur les deux objets importés depuis
            # `poisson/` (001_chips_can, 076_timer) : luminance moyenne 97 et 103
            # contre 150 à 180 pour les 43 autres.
            obj.data.materials.clear()
            obj.data.materials.append(mat)
                
            # Render and save
            if len(materials) == 1 and not args.suffix_material:
                render_filepath = os.path.join(output_dir, f"{shape_name}.png")
            else:
                render_filepath = os.path.join(output_dir, f"{shape_name}_{mat_name}.png")
            bpy.context.scene.render.filepath = render_filepath
            bpy.ops.render.render(write_still=True)
            print(f"  → Saved {mat_name} render to {render_filepath}")
            
        # Delete object to clear scene for next shape
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.object.delete()

if __name__ == "__main__":
    main()
