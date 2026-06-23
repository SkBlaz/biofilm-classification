#!/bin/bash
set -o pipefail
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
  echo -e "\t\t<task (generate_features | learning_benchmark | learning_benchmark_save_models | inference)>"
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
	FEATURES_FILE=""
	# Use environment variables if available, otherwise require parameters
	if [ -n "$IMAGINE_INFERENCE_INPUTS" ] && [ -n "$IMAGINE_INFERENCE_OUTPUTS" ]; then
		# Environment variable mode
		MODELS_PATH="$OUTPUT_RESULTS_FOLDER/models"
		IMAGES_PATH="$IMAGINE_INFERENCE_INPUTS"
		INFERENCE_OUTPUT_PATH="$IMAGINE_INFERENCE_OUTPUTS"
		if [ "$INPUT_DATASETNAME" != "-" ]; then
			FEATURES_FILE="$INPUT_DATASETNAME"
			if [ ! -f "$FEATURES_FILE" ]; then
				features_basename=$(basename "$INPUT_DATASETNAME")
				if [ -n "$IMAGINE_INFERENCE_DATAFILE" ] && [ -f "$IMAGINE_INFERENCE_DATAFILE/$features_basename" ]; then
					FEATURES_FILE="$IMAGINE_INFERENCE_DATAFILE/$features_basename"
				elif [ -f "$OUTPUT_RESULTS_FOLDER/$features_basename" ]; then
					FEATURES_FILE="$OUTPUT_RESULTS_FOLDER/$features_basename"
				fi
			fi
		elif [ -n "$IMAGINE_INFERENCE_DATAFILE" ] && [ -f "$IMAGINE_INFERENCE_DATAFILE" ]; then
			FEATURES_FILE="$IMAGINE_INFERENCE_DATAFILE"
		fi
		echo "Using environment variables for inference:"
		echo "Models path: $MODELS_PATH (from IMAGINE_RESULTS/models)"
		echo "Images path: $IMAGES_PATH (from IMAGINE_INFERENCE_INPUTS)"
		echo "Output path: $INFERENCE_OUTPUT_PATH (from IMAGINE_INFERENCE_OUTPUTS)"
		if [ -n "$FEATURES_FILE" ]; then
			echo "Features file: $FEATURES_FILE"
		fi
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
		if [ ${#INFERENCE_PARAMS[@]} -ge 8 ]; then
			FEATURES_FILE="${INFERENCE_PARAMS[7]}"
		elif [ "$INPUT_DATASETNAME" != "-" ]; then
			FEATURES_FILE="$INPUT_DATASETNAME"
		fi
		echo "Using parameters for inference:"
		echo "Models path: $MODELS_PATH"
		echo "Images path: $IMAGES_PATH"
		echo "Output path: $INFERENCE_OUTPUT_PATH"
		if [ -n "$FEATURES_FILE" ]; then
			echo "Features file: $FEATURES_FILE"
		fi
	fi
fi

RESULTS_CREATE_DF="${OUTPUT_RESULTS_FOLDER}/${INPUT_DATASETNAME}"
RESULTS_RANKING_FILE="${OUTPUT_RESULTS_FOLDER}/rankings.tsv"
RESULTS_FOLDER_VISUALIZATIONS="${OUTPUT_RESULTS_FOLDER}/visualizations"
RESULTS_FOLDER_FEATURE_GENERATOR="${OUTPUT_RESULTS_FOLDER}/feature_generator"
RESULTS_FOLDER_RAW="${OUTPUT_RESULTS_FOLDER}/raw"
RESULTS_FOLDER_ANALYSIS="${OUTPUT_RESULTS_FOLDER}/analysis"
############################################ [ LEAVE IT ] ##############################################################

print_failure_hints() {
	local log_file="$1"

	if grep -qiE "out of memory|memoryerror|cannot allocate memory|killed" "$log_file"; then
		echo "Hint: The process may have run out of memory. Try reducing parallelism and/or using fewer learners."
	fi

	if grep -qiE "cannot create cross-validation splitter|minimum class count|error.*label|failed.*label|missing.*label|error.*target[_ ]col|failed.*target[_ ]col|at least [0-9]+ samples" "$log_file"; then
		echo "Hint: The dataset may not have enough supported labels/classes for the requested benchmark step."
	fi

	if grep -qiE "No such file or directory|FileNotFoundError" "$log_file"; then
		echo "Hint: One of the input/output paths does not exist or is not mounted correctly."
	fi
}

run_step() {
	local step_name="$1"
	shift

	local safe_step_name
	safe_step_name=$(echo "$step_name" | tr -c '[:alnum:]' '_')
	local tmp_dir
	tmp_dir="${TMPDIR:-/tmp}"
	local log_file
	log_file=$(mktemp "${tmp_dir}/pipeline_${safe_step_name}.XXXXXX") || {
		echo "ERROR: Could not create a temporary log file for step: ${step_name}"
		exit 1
	}
	echo "Running step: ${step_name}"

	"$@" >"$log_file" 2>&1
	local status=$?
	cat "$log_file"

	if [ "$status" -ne 0 ]; then
		echo ""
		echo "ERROR: Pipeline step failed: ${step_name}"
		echo "Exit code: ${status}"
		print_failure_hints "$log_file"
		echo "Please review the logs above for details."
		rm -f "$log_file"
		exit "$status"
	fi

	rm -f "$log_file"
}

validate_tif_inputs() {
	local image_dir="$1"
	shopt -s nullglob
	local tif_files=("${image_dir}"/*.tif)
	shopt -u nullglob
	if [ "${#tif_files[@]}" -eq 0 ]; then
		echo "ERROR: No .tif files found in ${image_dir}"
		return 1
	fi
	return 0
}

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

	run_step "validate input .tif files" validate_tif_inputs "${INPUT_IMAGE_FOLDER}"
	run_step "feature generation for all input images" parallel --halt now,fail=1 --verbose -j"${INPUT_PARALLELISM}" \
		python feature_generator.py --outfolder "${RESULTS_FOLDER_FEATURE_GENERATOR}" --file {} ::: "${INPUT_IMAGE_FOLDER}"/*.tif

	# creating a dataset from images
	run_step "create joint dataframe" python create_joint_df.py "${RESULTS_FOLDER_FEATURE_GENERATOR}" "${RESULTS_FOLDER_RAW}"

	# Compute aggregated features
	run_step "compute aggregated features" python analysis.py "${RESULTS_FOLDER_RAW}" "${RESULTS_FOLDER_ANALYSIS}"

	# Create the final DF
	run_step "create final dataframe" python create_final_df_from_results.py "${RESULTS_FOLDER_ANALYSIS}" "${RESULTS_CREATE_DF}"
fi

if [ $LEARNING_TASK = "learning_benchmark" ]; then
	# Check if datafile exists in working directory but not in results directory
	if [ ! -f "${RESULTS_CREATE_DF}" ] && [ -f "${INPUT_DATASETNAME}" ]; then
		echo "Copying ${INPUT_DATASETNAME} from working directory to ${RESULTS_CREATE_DF}"
		mkdir -p "${OUTPUT_RESULTS_FOLDER}"
		cp "${INPUT_DATASETNAME}" "${RESULTS_CREATE_DF}"
	fi
	# Ensure visualizations directory exists
	mkdir -p "${RESULTS_FOLDER_VISUALIZATIONS}"
	# calculating feature rankings + intermediary frames etc.
	run_step "run learning benchmark" python feature_ranking_lite.py --parallelism "${INPUT_PARALLELISM}" --files "${RESULTS_CREATE_DF}" --fout "${RESULTS_RANKING_FILE}" ${ALL_LEARNERS_FLAG}
	run_step "visualize benchmark results" python visualize_benchmark.py
fi

if [ $LEARNING_TASK = "learning_benchmark_save_models" ]; then
	# Check if datafile exists in working directory but not in results directory
	if [ ! -f "${RESULTS_CREATE_DF}" ] && [ -f "${INPUT_DATASETNAME}" ]; then
		echo "Copying ${INPUT_DATASETNAME} from working directory to ${RESULTS_CREATE_DF}"
		mkdir -p "${OUTPUT_RESULTS_FOLDER}"
		cp "${INPUT_DATASETNAME}" "${RESULTS_CREATE_DF}"
	fi
	# Ensure visualizations directory exists
	mkdir -p "${RESULTS_FOLDER_VISUALIZATIONS}"
	# calculating feature rankings + intermediary frames etc. + save models for inference
	run_step "run learning benchmark and save models" python feature_ranking_lite.py --parallelism "${INPUT_PARALLELISM}" --files "${RESULTS_CREATE_DF}" --fout "${RESULTS_RANKING_FILE}" --save_models ${ALL_LEARNERS_FLAG}
	run_step "visualize benchmark results" python visualize_benchmark.py
fi

if [ $LEARNING_TASK = "data_visualization" ]; then
	# Check if datafile exists in working directory but not in results directory
	if [ ! -f "${RESULTS_CREATE_DF}" ] && [ -f "${INPUT_DATASETNAME}" ]; then
		echo "Copying ${INPUT_DATASETNAME} from working directory to ${RESULTS_CREATE_DF}"
		mkdir -p "${OUTPUT_RESULTS_FOLDER}"
		cp "${INPUT_DATASETNAME}" "${RESULTS_CREATE_DF}"
	fi
	# Use rankings_label.tsv if rankings.tsv doesn't exist
	RANKINGS_FILE="${RESULTS_RANKING_FILE}"
	if [ ! -f "${RANKINGS_FILE}" ] && [ -f "${OUTPUT_RESULTS_FOLDER}/rankings_label.tsv" ]; then
		echo "Using rankings_label.tsv for visualization"
		RANKINGS_FILE="${OUTPUT_RESULTS_FOLDER}/rankings_label.tsv"
	fi
	# visualizations
	run_step "generate data visualizations" python ./visualizations/pipeline_visualizations.py --data "${RESULTS_CREATE_DF}" --rankings "${RANKINGS_FILE}" --fout "${RESULTS_FOLDER_VISUALIZATIONS}" --nbfeatures "${INPUT_NB_VISUALIZATION_FEATURES}"
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
	if [ -n "$FEATURES_FILE" ]; then
		echo "Features: $FEATURES_FILE"
		run_step "run inference pipeline" python inference.py "$MODELS_PATH" "$IMAGES_PATH" "$INFERENCE_OUTPUT_PATH" --features_file "$FEATURES_FILE"
	else
		run_step "run inference pipeline" python inference.py "$MODELS_PATH" "$IMAGES_PATH" "$INFERENCE_OUTPUT_PATH"
	fi
fi 
