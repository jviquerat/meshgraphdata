import os
import sys
import json

import meshio
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def read_meta_json(filename):
    """
    Read the meta json file for a dataset and return dict
    """
    with open(filename, 'r') as f:
        meta = json.load(f)

    return meta

def create_pvd(filename, timesteps, filenames):
    """
    Create a pvd file to represent a time series of data files
    """
    # Get the directory where the PVD file will be saved to make file paths relative
    output_dir = os.path.dirname(filename)

    # Start xml
    pvd_content = """<?xml version="1.0"?>
    <VTKFile type="Collection" version="1.0" byte_order="LittleEndian">
    <Collection>
    """

    # Add a dataSet entry for each timestep and filename
    for time, fname in zip(timesteps, filenames):
        # Make the file path relative to the PVD file's location
        relative_path = os.path.relpath(fname, output_dir)
        pvd_content += f'    <DataSet timestep="{time}" file="{relative_path}"/>\n'

    # Close xml
    pvd_content += """  </Collection>
    </VTKFile>
    """

    with open(filename, 'w') as f:
        f.write(pvd_content)

def tfrecord_to_vtu(data_file, meta, output_dir):
    """
    Read a tfrecord dataset and convert it to a set of separate vtu trajectories
    """

    def parse(proto):
        """
        Parse a trajectory from tf.Example
        Copied from the meshgraphnet repository
        """
        feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
        features = tf.io.parse_single_example(proto, feature_lists)
        out = {}
        for key, field in meta["features"].items():
            data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
            data = tf.reshape(data, field["shape"])
            if field.get("for_sim", False):
                out[key] = data
                continue
            if field["type"] == "static":
                data = tf.tile(data, [meta["trajectory_length"], 1, 1])
            elif field["type"] == "dynamic_varlen":
                length = tf.io.decode_raw(features["length_" + key].values, tf.int32)
                length = tf.reshape(length, [-1])
                data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
            elif field["type"] != "dynamic":
                raise ValueError("invalid data format")
            out[key] = data

        return out

    # Load dataset and map using dedicated function
    dataset = tf.data.TFRecordDataset(data_file)
    dataset = dataset.map(parse)

    # Process trajectories
    for i, trajectory in enumerate(dataset):
        print(f"# Processing trajectory {i}", end="\r")

        # Put trajectory in separate folder
        odir = os.path.join(output_dir, str(i))
        if not os.path.exists(odir):
            os.makedirs(odir)

        # Unroll the trajectory into frames using only the dynamic fields
        frames = tf.data.Dataset.from_tensor_slices(trajectory)
        timestamps = []
        filenames  = []
        findex = 0
        for frame in frames:
            # Coordinates
            if "world_pos" in meta["field_names"]:
                coords = frame["world_pos"].numpy()
            else:
                coords = frame["mesh_pos"].numpy()

            # Cells
            cells = frame["cells"].numpy()
            shape = meta["features"]["cells"]["shape"][1:]
            cell_type = "tetra" if shape[-1] == 4 else "triangle"
            cells = [(cell_type, cells)]

            # Combine all other data as point_data
            point_data = {}
            for key, tensor in frame.items():
                if key not in ["world_pos", "cells", "mesh_pos"]:
                    point_data[key] = tensor.numpy()

            # Write vtu file for current frame
            mesh = meshio.Mesh(
                points=coords,
                cells=cells,
                point_data=point_data
            )

            # Write
            output_filename = os.path.join(odir, f"{findex}.vtu")
            mesh.write(output_filename, binary=False)

            timestamps.append(findex)
            filenames.append(output_filename)
            findex += 1

        pvd_filename = os.path.join(odir, "trajectory.pvd")
        create_pvd(pvd_filename, timestamps, filenames)

#################################################
if __name__ == '__main__':

    args       = sys.argv
    data_dir   = args[1]
    data_split = args[2]
    output_dir = args[3]
    data_file  = os.path.join(data_dir, data_split + ".tfrecord")
    meta_file  = os.path.join(data_dir, "meta.json")
    print(f"# Data file: {data_file}")

    # Check inputs
    if not os.path.isfile(data_file):
        print("# Error: could not find data file")
        exit(0)

    if not os.path.isfile(meta_file):
        print("# Error: could not find meta file")
        exit(0)

    # Inspect meta file
    meta = read_meta_json(meta_file)
    print("# Found features:")
    for k in meta["features"]:
        print(f"#   - {k}")

    # Create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tfrecord_to_vtu(data_file, meta, output_dir)
