#!/bin/bash
########################################################################################################################
# run full pipe																											  #
# bash run_analysis.sh <image folder> <number of parallel jobs> <TSV dataset name> <visualize top N features> #
# 																														  #
# time bash run_analysis.sh ../datasets/3d_images_all 7 ../prepared_data/new_features_11_11_23.tsv

########################################################################################################################

if [ $# -lt 4 ]; then
  echo "Not enough input parameters"
  echo ""
  echo "Usage:"
  echo ""
  echo "If you are using a Docker container:"
  echo -e "=> docker run \t-v ./your/images:/imagine/images"
  echo -e "\t\t-v ./your/results:/imagine/results "
  echo -e "\t\t--rm"
  echo -e "\t\t-it jsi/imagine"
  echo -e "\t\t<nb parallel jobs (4)>"
  echo -e "\t\t<dataset name (datafile.tsv)>"
  echo -e "\t\t<top features to visualize (10)>"
  echo -e "\t\t<task (generate_features | generate_features_lite | learning_benchmark | learning_benchmark_save_models | data_visualization | inference)>"
  echo -e "\t\t[--all_learners (optional: enable all ML algorithms)]"
  echo
  echo
  echo "For inference task (environment variable mode):"
  echo "Set IMAGINE_INFERENCE_INPUTS and IMAGINE_INFERENCE_OUTPUTS in .env file"
  echo -e "=> docker compose run --rm imagine 4 - 10 inference"
  echo
  echo "For inference task (parameter mode):"
  echo -e "=> docker compose run --rm imagine 4 - 10 inference <models_path> <images_path> <output_path>"
  echo
  echo
  echo "If you are running the script directly:"
  echo "=> run_analysis.sh <image folder> <nb parallel jobs (4)> <dataset name (datafile.tsv)> <top features to visualize (10)> <results folder (./your/results)> <task>"
  exit
fi

INPUT_IMAGE_FOLDER="/imagine/images"
INPUT_PARALLELISM="$1"
INPUT_DATASETNAME="$2"
INPUT_NB_VISUALIZATION_FEATURES="$3"
LEARNING_TASK="$4"
OUTPUT_RESULTS_FOLDER='/imagine/results'

# Check for optional --all_learners flag in any position after the required parameters
ALL_LEARNERS_FLAG=""
for arg in "$@"; do
    if [ "$arg" = "--all_learners" ]; then
        ALL_LEARNERS_FLAG="--all_learners"
        echo "All learners flag enabled - will use all machine learning algorithms"
        break
    fi
done

# Handle special case for inference task
if [ "$LEARNING_TASK" = "inference" ]; then
	# Use environment variables if available, otherwise require parameters
	if [ -n "$IMAGINE_INFERENCE_INPUTS" ] && [ -n "$IMAGINE_INFERENCE_OUTPUTS" ]; then
		# Environment variable mode
		MODELS_PATH="$OUTPUT_RESULTS_FOLDER/models"
		IMAGES_PATH="$IMAGINE_INFERENCE_INPUTS"
		INFERENCE_OUTPUT_PATH="$IMAGINE_INFERENCE_OUTPUTS"
		echo "Using environment variables for inference:"
		echo "Models path: $MODELS_PATH (from IMAGINE_RESULTS/models)"
		echo "Images path: $IMAGES_PATH (from IMAGINE_INFERENCE_INPUTS)"
		echo "Output path: $INFERENCE_OUTPUT_PATH (from IMAGINE_INFERENCE_OUTPUTS)"
	else
		# Parameter mode (backward compatibility)
		# Filter out --all_learners flag from inference parameters
		INFERENCE_PARAMS=()
		for arg in "$@"; do
			if [ "$arg" != "--all_learners" ]; then
				INFERENCE_PARAMS+=("$arg")
			fi
		done
		
		if [ ${#INFERENCE_PARAMS[@]} -lt 7 ]; then
			echo "Inference task requires either environment variables or additional parameters:"
			echo ""
			echo "Environment variable mode:"
			echo "Set IMAGINE_INFERENCE_INPUTS and IMAGINE_INFERENCE_OUTPUTS in .env file"
			echo "=> docker compose run --rm imagine 4 - 10 inference"
			echo ""
			echo "Parameter mode:"
			echo "=> docker compose run --rm imagine 4 - 10 inference <models_path> <images_path> <output_path>"
			echo "=> docker compose run --rm imagine 4 - 10 inference <models_path> <images_path> <output_path> --all_learners"
			exit 1
		fi
		MODELS_PATH="${INFERENCE_PARAMS[4]}"
		IMAGES_PATH="${INFERENCE_PARAMS[5]}"  
		INFERENCE_OUTPUT_PATH="${INFERENCE_PARAMS[6]}"
		echo "Using parameters for inference:"
		echo "Models path: $MODELS_PATH"
		echo "Images path: $IMAGES_PATH"
		echo "Output path: $INFERENCE_OUTPUT_PATH"
	fi
fi

RESULTS_CREATE_DF="${OUTPUT_RESULTS_FOLDER}/${INPUT_DATASETNAME}"
RESULTS_RANKING_FILE="${OUTPUT_RESULTS_FOLDER}/rankings.tsv"
RESULTS_FOLDER_VISUALIZATIONS="${OUTPUT_RESULTS_FOLDER}/visualizations"
RESULTS_FOLDER_FEATURE_GENERATOR="${OUTPUT_RESULTS_FOLDER}/feature_generator"
RESULTS_FOLDER_RAW="${OUTPUT_RESULTS_FOLDER}/raw"
RESULTS_FOLDER_ANALYSIS="${OUTPUT_RESULTS_FOLDER}/analysis"
############################################ [ LEAVE IT ] ##############################################################

echo "Using the following parameters for input:"
echo "Input images folder: $INPUT_IMAGE_FOLDER"
echo "Results folder: $OUTPUT_RESULTS_FOLDER (if running Docker, map it to a volume to see the results)"
echo "Parallelism: $INPUT_PARALLELISM"
echo "Dataset name: $INPUT_DATASETNAME"
echo "Number of visualization features: $INPUT_NB_VISUALIZATION_FEATURES"
echo "Task: $LEARNING_TASK"


if [ $LEARNING_TASK = "generate_features" ]; then
	rm -rvf "${OUTPUT_RESULTS_FOLDER}/*"
	mkdir -p "${OUTPUT_RESULTS_FOLDER}"/{feature_generator,raw,analysis,visualizations}

	ls "${INPUT_IMAGE_FOLDER}"/*.tif | awk -v res=$RESULTS_FOLDER_FEATURE_GENERATOR '{print "python feature_generator.py --outfolder " res " --file "$1}' | parallel --progress --verbose -j"${INPUT_PARALLELISM}"

	# creating a dataset from images
	python create_joint_df.py "${RESULTS_FOLDER_FEATURE_GENERATOR}" "${RESULTS_FOLDER_RAW}"

	# Compute aggregated features
	python analysis.py "${RESULTS_FOLDER_RAW}" "${RESULTS_FOLDER_ANALYSIS}"

	# Create the final DF
	python create_final_df_from_results.py "${RESULTS_FOLDER_ANALYSIS}" "${RESULTS_CREATE_DF}"
fi

if [ $LEARNING_TASK = "generate_features_lite" ]; then
	rm -rvf "${OUTPUT_RESULTS_FOLDER}/*"
	mkdir -p "${OUTPUT_RESULTS_FOLDER}"/{feature_generator,raw,analysis,visualizations}

	ls "${INPUT_IMAGE_FOLDER}"/*.tif | awk -v res=$RESULTS_FOLDER_FEATURE_GENERATOR '{print "python feature_generator_lite.py --outfolder " res " --file "$1}' | parallel --progress --verbose -j"${INPUT_PARALLELISM}"

	# creating a dataset from images
	python create_joint_df.py "${RESULTS_FOLDER_FEATURE_GENERATOR}" "${RESULTS_FOLDER_RAW}"

	# Compute aggregated features
	python analysis.py "${RESULTS_FOLDER_RAW}" "${RESULTS_FOLDER_ANALYSIS}"

	# Create the final DF
	python create_final_df_from_results.py "${RESULTS_FOLDER_ANALYSIS}" "${RESULTS_CREATE_DF}"
fi

if [ $LEARNING_TASK = "learning_benchmark" ]; then
	# Check if datafile exists in working directory but not in results directory
	if [ ! -f "${RESULTS_CREATE_DF}" ] && [ -f "${INPUT_DATASETNAME}" ]; then
		echo "Copying ${INPUT_DATASETNAME} from working directory to ${RESULTS_CREATE_DF}"
		mkdir -p "${OUTPUT_RESULTS_FOLDER}"
		cp "${INPUT_DATASETNAME}" "${RESULTS_CREATE_DF}"
	fi
	# calculating feature rankings + intermediary frames etc.
	python feature_ranking_lite.py --parallelism "${INPUT_PARALLELISM}" --files "${RESULTS_CREATE_DF}" --fout "${RESULTS_RANKING_FILE}" ${ALL_LEARNERS_FLAG}
fi

if [ $LEARNING_TASK = "learning_benchmark_save_models" ]; then
	# Check if datafile exists in working directory but not in results directory
	if [ ! -f "${RESULTS_CREATE_DF}" ] && [ -f "${INPUT_DATASETNAME}" ]; then
		echo "Copying ${INPUT_DATASETNAME} from working directory to ${RESULTS_CREATE_DF}"
		mkdir -p "${OUTPUT_RESULTS_FOLDER}"
		cp "${INPUT_DATASETNAME}" "${RESULTS_CREATE_DF}"
	fi
	# calculating feature rankings + intermediary frames etc. + save models for inference
	python feature_ranking_lite.py --parallelism "${INPUT_PARALLELISM}" --files "${RESULTS_CREATE_DF}" --fout "${RESULTS_RANKING_FILE}" --save_models ${ALL_LEARNERS_FLAG}
fi

if [ $LEARNING_TASK = "data_visualization" ]; then
	# Check if datafile exists in working directory but not in results directory
	if [ ! -f "${RESULTS_CREATE_DF}" ] && [ -f "${INPUT_DATASETNAME}" ]; then
		echo "Copying ${INPUT_DATASETNAME} from working directory to ${RESULTS_CREATE_DF}"
		mkdir -p "${OUTPUT_RESULTS_FOLDER}"
		cp "${INPUT_DATASETNAME}" "${RESULTS_CREATE_DF}"
	fi
	# visualizations
	python ./visualizations/pipeline_visualizations.py --data "${RESULTS_CREATE_DF}" --rankings "${RESULTS_RANKING_FILE}" --fout "${RESULTS_FOLDER_VISUALIZATIONS}" --nbfeatures "${INPUT_NB_VISUALIZATION_FEATURES}"
fi

if [ $LEARNING_TASK = "reduce_layers" ]; then
  # halve number of layers in images
  NUM_LAYERS=${5:-21}
  for IMAGE in $INPUT_IMAGE_FOLDER/*.tif; do
    bash remove_layers.sh $IMAGE $NUM_LAYERS
  done
fi

if [ $LEARNING_TASK = "inference" ]; then
	# run inference on new images using pre-trained models
	echo "Running inference..."
	echo "Models: $MODELS_PATH"
	echo "Images: $IMAGES_PATH" 
	echo "Output: $INFERENCE_OUTPUT_PATH"
	
	python inference.py "$MODELS_PATH" "$IMAGES_PATH" "$INFERENCE_OUTPUT_PATH"
fi 
