# github.com/colinrizzman
# pip install tensorflow numpy pandas trimesh "pyglet<2"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import signal
import tensorflow as tf
import numpy as np
import pandas as pd

# hyperparameters
plyname = "model"
epochs = 33333
batchs = 512
units = 96
layers = 6
topo = 1

# graceful stopping
class _SigintFlag:
    def __init__(self): self.stop = False
    def __call__(self, signum, frame): self.stop = True
sigint_flag = _SigintFlag()
signal.signal(signal.SIGINT, sigint_flag)
class GracefulStop(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        if sigint_flag.stop: self.model.stop_training = True
    def on_epoch_end(self, epoch, logs=None):
        if sigint_flag.stop: self.model.stop_training = True

# parse ascii ply
def parse_ascii_ply(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    header = []
    vertex_data = []
    start_index = 0

    for i, line in enumerate(lines):
        header.append(line)
        if line.strip() == "end_header":
            start_index = i + 1
            break

    for line in lines[start_index:]:
        tokens = line.strip().split()
        if len(tokens) >= 10:  # x y z nx ny nz r g b a
            try:
                x, y, z = map(float, tokens[0:3])
                r, g, b = map(int, tokens[6:9])
                vertex_data.append([x, y, z, r, g, b])
            except ValueError:
                continue

    vertex_df = pd.DataFrame(vertex_data, columns=["x", "y", "z", "r", "g", "b"])
    return vertex_df, header

# load input ply
input_path = plyname+'.ply'
output_path = 'reconstructed_'+plyname+'_'+str(epochs)+'_'+str(batchs)+'_'+str(units)+'_'+str(layers)+'_'+str(topo)+'.ply'
vertex_df, ply_header = parse_ascii_ply(input_path)

# extract and normalise data
positions = vertex_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
colors = vertex_df[["r", "g", "b"]].to_numpy(dtype=np.float32) / 255.0
pos_mean = positions.mean(axis=0)
pos_std = positions.std(axis=0)
positions_norm = (positions - pos_mean) / pos_std

# construct neural network
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(3,)))
model.add(tf.keras.layers.Dense(units, activation='relu'))
if topo == 0:
    for x in range(layers): model.add(tf.keras.layers.Dense(units, activation='relu'))
elif topo == 1:
    for x in range(layers-1): model.add(tf.keras.layers.Dense(units, activation='relu'))
    model.add(tf.keras.layers.Dense(int(units/2), activation='relu'))
elif topo == 2:
    dunits = units
    for x in range(layers):
        model.add(tf.keras.layers.Dense(int(dunits), activation='relu'))
        dunits=dunits/2
model.add(tf.keras.layers.Dense(3))

# summary, compile, fit
model.summary()
model.compile(optimizer='adam', loss='mse')
model.fit(positions_norm, colors, epochs=epochs, batch_size=batchs, callbacks=[GracefulStop()])

# parse input ply
def parse_ply_fully(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header = []
    vertex_lines = []
    face_lines = []
    vertex_count = 0
    face_count = 0
    in_header = True

    for line in lines:
        if in_header:
            header.append(line)
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face"):
                face_count = int(line.split()[-1])
            elif line.strip() == "end_header":
                in_header = False
                continue
        else:
            break

    vertex_lines = lines[len(header):len(header)+vertex_count]
    face_lines = lines[len(header)+vertex_count:]
    return header, vertex_lines, face_lines

# export ply
def reconstruct_ply_with_preserved_format(input_path, output_path):
    header, vertex_lines, face_lines = parse_ply_fully(input_path)

    # extract positions
    positions = []
    for line in vertex_lines:
        tokens = line.strip().split()
        x, y, z = map(float, tokens[:3])
        positions.append([x, y, z])
    positions = np.array(positions, dtype=np.float32)

    # predict new colors
    positions_norm = (positions - pos_mean) / pos_std
    predicted_colors = model.predict(positions_norm)
    predicted_colors = np.clip(predicted_colors * 255.0, 0, 255).astype(np.uint8)

    # export new lines
    new_vertex_lines = []
    for original_line, new_rgb in zip(vertex_lines, predicted_colors):
        tokens = original_line.strip().split()
        tokens[6] = str(new_rgb[0])  # red
        tokens[7] = str(new_rgb[1])  # green
        tokens[8] = str(new_rgb[2])  # blue
        new_vertex_lines.append(" ".join(tokens) + "\n")

    # output to file
    with open(output_path, 'w') as out:
        out.writelines(header)
        out.writelines(new_vertex_lines)
        out.writelines(face_lines)

# do reconstruction
reconstruct_ply_with_preserved_format(input_path, output_path)

# display result
import trimesh
mesh1 = trimesh.load(input_path)
mesh2 = trimesh.load(output_path)
mesh2.apply_translation([0.53, 0, 0])
rotation_x = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])
mesh1.apply_transform(rotation_x)
mesh2.apply_transform(rotation_x)
scene = trimesh.Scene()
scene.add_geometry(mesh1)
scene.add_geometry(mesh2)
scene.show()
