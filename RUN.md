# Docker images
We prepared a Docker image that can be used to run the processing pipeline.
A Docker image can contain anything; in our case, it contains a Python installation with all required dependencies,
together with the most important part - the code that actually does image processing for the Imagine project. Make sure you have docker `Docker version 27.0.0` or later.

Think of a Docker image as a template with pre-defined behavior (the behavior is defined by the image creator). 
In order to obtain any Docker image, one must first build it. This is achieved by using Dockerfiles (blueprints/recipes),
simple text files with a specific syntax. One can be found in the same folder as this README.

Once the Dockerfile (the blueprint for the software) is created, the image can be built. The build process creates a piece of executable code.
However, since images can (vaguely) be considered as templates, it is natural to assume that one can make several instances 
of such an image - those are called containers. 

## Why use containers?
The purpose of containerization is to remove the need to install all the software that is required in order to run a 
specific piece of code - these software are called dependencies. In our case, that code does image processing, 
machine learning and visualizations. All dependencies are installed within the Docker image so that the user of the image
does not have to worry about that. 

The only software that one must install in order to create Docker containers for the Imagine project are these:
- git; to clone the Imagine project repository and obtain the source code
- docker; to build and run Docker images

Each image can also expose parameters, that the user of the image can specify during the creation of a container. In our 
case, an example of such parameter is the path to the image folder.

## Generating and running the containers
In order to use the containers, one must first build them locally. 
Container(s) should be build only the very first time or after changes have been made to the actual pipeline.

First, install Docker:
- Linux users (Ubuntu): https://docs.docker.com/engine/install/ubuntu/ 
- Windows users: https://docs.docker.com/desktop/install/windows-install/

Then, build the container (commands bellow assume you are running them from the same directory as this README):
```sh
docker compose build --no-cache
```

## Testing if everything works
For testing purposes, commands below use predefined paths to data. The paths are specified in `docker-compose.yml` - do not change them.
Run the following commands one after another and wait for each of them to finish. The testing data is available in `./examples/test_images` folder. 
All commands should run and produce results (in folder `./examples/test_images_results`) without errors:

```sh
1. docker compose run --rm imagine-test-generate-features
2. docker compose run --rm imagine-test-learning-benchmark
3. docker compose run --rm imagine-test-data-visualization
```


## Mounting volumes of data into the containers
The prepared Docker image is used to create containers that do the actual work. The input data needs to be accessible to the containers. We use Docker volumes to mount the data into the containers.
To simplify the usage of the Imagine Docker image, we prepared a Compose file (docker-compose.yml) which does all the required volume mounting for you.
All one needs to do is to provide environment variables with correct paths.

One should specify the environment variables in one of two ways:
1. For each experiment, edit the `.env` file and change the paths:
```sh
docker compose run --rm imagine <PARAMETERS>
```

2. Create a different env file for each experiment and pass the env file (specify the whole env file name, including the file extension) as a parameter to the docker compose command :
```sh
docker compose --env-file my-env-file run --rm imagine <PARAMETERS>
```
(note: if `my-env-file` is not an `.env` file, please use appropriate postfix (e.g., `.txt`))

Windows and Linux hosts differ in the way how they mount folders to docker volumes. 
The two (toy example) commands below showcase the slight difference between running Docker images on Windows and Unix hosts.

* IMAGINE_IMAGES=./my_folder/test_images
* IMAGINE_RESULTS=./my_folder/test_results

For Windows users:
* IMAGINE_IMAGES=/c/my_folder/test_images
* IMAGINE_RESULTS=/c/my_folder/test_results

The expected result is the same in both cases. The only thing that differs is the way that the input/output folders are mounted into the containers. 
It is important to note that Windows requires absolute paths to mount volumes whereas Linux does not. 
Therefore, in order to mount a volume from Windows folder `c:\my_folder\test_images`, one must write it as `/c/my_folder/test_images`.

## Using the Docker image
The image that we prepared (tagged jsi/imagine:latest) accepts several parameters. One of those parameters is called "task" which translates to "what we want to do" with the image.
Other parameters, that are required to run that specific task, are task-dependent. 

**Note: Keep in mind that the order of parameters matters!**

The following use-cases use the folders specified in the `.env` file.


### Task: Generate features
This task has the following parameters:
- the number of parallel threads to use: 4
- the desired name of the dataset that we will create: datafile.tsv
- the number of top features to visualize: 10
- the task: generate_features

```sh
docker compose run --rm imagine 4 datafile.tsv 10 generate_features
```

The results of this task will be in the folder, that is mapped to the `/imagine/results` within the container (`/c/my_folder/test_results` in this case).

### Task: Learning benchmark
Learning benchmark contains the gist of this software - a collection of machine learning algorithms that attampt to approximate the strain based on thousands of generated features. Current implementation is fully automated; by running the command below, you can simulate how well the algorithm learns to associate labels with feature space. The run includes the currently selected tree-based ensembles, as well as simple baselines (majority) that should be indicative of how well a naive approach would perform.

This task has the following parameters:
- ...
- ...
- the task: learning_benchmark

```sh
docker compose run --rm imagine 4 datafile.tsv 10 learning_benchmark
```

(i.e., we just replace "generate_features" with "learning_benchmark")

The process will start from the working folder (`/imagine/results`) and conduct the basic machine learning benchmark.

## How results look like/interpretation

In folder `results_example` you have an example results of an actual run (`bash run_docker_image_everything.sh`). The output of the final run (results folder) has the following structure:

```
├── analysis
│   ├── SUMMARYCustomAlgos.tsv_max.txt
│   ├── SUMMARYCustomAlgos.tsv_mean.txt
....
....
│   └── SUMMARYDiffGlobal.tsv_var.txt
├── classification.tsv
├── data.tsv
├── data.tsvintermediary_aggregated.tsv
├── feature_generator
│   ├── 04072023_s_Lm_st_L1323_p_C04_pos001_tm_24_ch_Syto9_z_21CustomAlgos.txt
....
....
│   └── 30052023_s_Lm_st_L634_p_D05_pos005_tm_24_ch_Syto9_z_21DiffGlobal.txt
├── rankings.tsv
├── raw
│   ├── CustomAlgos.tsv
│   └── DiffGlobal.tsv
```


File descriptions:

* `analysis` -> Aggregated statistics for each image, input for creating `data.tsv`
* `data.tsv` -> file containing final features and samples, input for learning
* `data.tsvintermediary_aggregated.tsv` -> Intermediary feature-related results, for inspection (Nika's request)
* `feature_generator` -> Features, generated for each image. Multiple files, used to create final dataset
* `rankings.tsv` -> outputs of feature ranking. For each feature, we compute the strength of its relation with target - bigger values imply higher association (more important features)
* `classification.tsv` -> Results of machine learning classification. Data is split into 10 parts, one is hidden, remaining are used to predict the label of that part (repeated for each part). `results` also contain labels where the algorith wasn't successful (for debugging)
* `raw` -> Raw outputs of feature generation, useful for inspection

## Task: Visualization

To run visualization, simply change the task to `data_visualization`, for example,

```sh
docker compose run --rm imagine 4 data.tsv 50 data_visualization
```

This will produce a folder called `tmp` in the results folder. The folder contains all visualizations of top n features, grouped by strains.

## Task: Inference

The inference task allows you to use previously trained models to generate predictions on new data. This task requires that you have already run the `learning_benchmark` task to train and save models.

```sh
docker compose run --rm imagine 4 data.tsv 10 inference
```

**Prerequisites:**
- Models must exist in the results folder (created by running `learning_benchmark`)
- Dataset file must exist (created by running `generate_features` or provided manually)

**What it does:**
- Loads trained models from the `models/` directory in your results folder
- Applies the same preprocessing pipeline used during training
- Generates predictions for all samples in the dataset
- Outputs results to `inference_results/` directory

**Output files:**
- `predictions.tsv` - Contains predictions from all models alongside true labels (if available)
- `probabilities/` - Directory containing prediction probabilities for each model
- `inference_summary.json` - Summary of the inference run including model performance metrics

**Example workflow:**
1. Run feature generation: `docker compose run --rm imagine 4 data.tsv 10 generate_features`
2. Train models: `docker compose run --rm imagine 4 data.tsv 10 learning_benchmark`
3. Generate predictions: `docker compose run --rm imagine 4 data.tsv 10 inference`

If you have new unlabeled data for prediction, simply replace the dataset file with your new data (ensure it has the same feature columns) and run the inference task.
