# meshgraphdata

This snippet converts the `.tfrecord` files from the meshgraphnet datasets to `.vtu` trajectories. 

## Usage

First, download a dataset using the `download.sh` file (copied from the meshgraphnet repository) following <a href="https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets#datasets">these steps</a>:

```
./download.sh <dataset_name> <local_folder_name>
```

Then, convert using the `data.py` file:

```
python3 data.py <dataset_folder> <split> <output_folder>
```

This should create the `output_folder`, and store the trajectories as `.vtu` files in separate folders within it. In each trajectory folder, it will write a `trajectory.pvd` file that can be open directly with paraview to visualize the trajectory. This is a quite long process.

## Examples

<img width="260" alt="gif" src="animations/plate.gif"> <img width="260" alt="gif" src="animations/flag.gif"> <img width="260" alt="gif" src="animations/sphere.gif">

## Issues

The metadata files for the `flag_dynamic_sizing` and `sphere_dynamic_sizing` datasets seem to have erroneous attributes for the `sizing_field` attributes. It is necessary to correct them by hand before converting them (issue opened <a href="https://github.com/google-deepmind/deepmind-research/issues/651">here</a>). To fix it, change this block:

```
"sizing_field": {
  "type": "dynamic_varlen",
  "shape": [
    -1,
     4
  ],
  "dtype": "<dtype: 'float32'>"
}
```

to this:

```
"sizing_field": {
  "type": "dynamic_varlen",
  "shape": [
    -1,
     3
  ],
  "dtype": "float32"
}
```
