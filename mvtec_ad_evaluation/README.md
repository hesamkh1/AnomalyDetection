# Evaluate your experiments on the MVTec AD dataset

The evaluation scripts can be used to assess the performance of a method on the
MVTec AD dataset. Given a directory with anomaly maps, the scripts compute
the area under the PRO curve for anomaly localization. Additionally, the area
under the ROC curve for anomaly classification is computed.

## Installation.
Our evaluation scripts require a python 3.7 installation as well as the
following packages:
- numpy
- Pillow
- scipy
- tabulate
- tifffile
- tqdm

### conda

For Linux, we provide an anaconda environment file. It can be used
to create a new conda environment with all required packages readily installed:

```
conda env create --name mad_eval_script --file=conda_environment.yml
conda activate mad_eval_script
```

### pip

```
pip install -r requirements.txt
```

## Evaluating a single experiment.
It requires an anomaly map to be present for each test sample in our dataset in
`.tif` or `.tiff` format. Anomaly maps must contain real-valued anomaly scores
and their size must match the one of the original dataset images. Anomaly maps
must all share the same base directory and adhere to the following folder
structure:
`<anomaly_maps_dir>/<object_name>/test/<defect_name>/<image_id>.tiff`

To evaluate a single experiment, the script `evaluate_experiment.py` can be
used. It requires the following user arguments:

- `dataset_base_dir`: Base directory that contains the MVTec AD dataset.
- `anomaly_maps_dir`: Base directory that contains the corresponding anomaly
  maps.

Optional parameters can be specified as follows:

- `output_dir`: Directory to store evaluation results as `.json` files.
- `pro_integration_limit`: Integration limit for the computation of the AU-PRO.
  Default: 0.3
- `evaluated_objects`: List of dataset object categories for which the
  computation should be performed.

A possible example call to this script would be:
```
  python evaluate_experiment.py --dataset_base_dir 'path/to/dataset/' \
                                --anomaly_maps_dir 'path/to/anomaly_maps/' \
                                --output_dir 'metrics/'
                                --pro_integration_limit 0.3
```

## Evaluate multiple experiments.
If more than a single experiment should be evaluated at once, the script
`evaluate_multiple_experiments.py` can be used. All directories with anomaly
maps can be specified in a `config.json` file with the following structure:
```
{
    "exp_base_dir": "/path/to/all/experiments/",
    "anomaly_maps_dirs": {
        "experiment_id_1": "eg/model_1/anomaly_maps/",
        "experiment_id_2": "eg/model_2/anomaly_maps/",
        "experiment_id_3": "eg/model_3/anomaly_maps/",
        "...": "..."
    }
}
```
- `exp_base_dir`: Base directory that contains all experimental results.
- `anomaly_maps_dirs`: Dictionary that contains an identifier for each evaluated
  experiment and the location of its anomaly maps relative to the
  `exp_base_dir`.

The evaluation can be run by calling `evaluate_multiple_experiments.py`,
providing the following user arguments:

- `dataset_base_dir`: Base directory that contains the MVTec AD dataset.
- `experiment_configs`: Path to the above `config.json` that contains all
  experiments to be evaluated.

Optional parameters can be specified as follows:

- `output_dir`: Directory to store evaluation results as `.json` file for each
  evaluated experiment.
- `pro_integration_limit`: Integration limit for the computation of the AU-PRO.

A possible example call to this script would be:
```
python evaluate_multiple_experiments.py \
  --dataset_base_dir 'path/to/dataset/' \
  --experiment_configs 'experiment_configs.json' \
  --output_dir 'metrics/' \
  --pro_integration_limit 0.3
```

## Visualize the evaluation results.
After running `evaluate_experiment.py` or `evaluate_multiple_experiments.py`,
the script `print_metrics.py` can be used to visualize all computed metrics in a
table. It requires only a single user argument:

- `metrics_folder`: The output directory specified in
  `evaluate_experiment.py` or `evaluate_multiple_experiments.py`.

# License
The license agreement for our evaluation code is found in the accompanying
`LICENSE.txt` file.

The version of this evaluation script is: 1.0
