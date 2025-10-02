FROM python:3.11-slim

WORKDIR /opt/imagine

RUN apt-get update && apt-get install -y --no-install-recommends \
    dos2unix libhdf5-dev graphviz locales curl git zip parallel imagemagick && \
	pip install --upgrade pip && \
	rm -rf /var/lib/apt/lists/*

# requirements
COPY src/requirements.docker.txt .

RUN pip install \
	--no-cache-dir \
	--trusted-host pypi.org \
	--trusted-host pypi.python.org \
	--trusted-host files.pythonhosted.org \
	-r requirements.docker.txt

# processing, ranking and util
COPY src/create_final_df_from_results.py \
	 src/feature_generator.py \
	 src/create_joint_df.py \
	 src/analysis.py \
	 src/run_analysis.sh \
	 src/feature_ranking_lite.py \
	 src/inference.py \
	 src/remove_layers.sh \
	 ./

# visualizations
COPY src/visualizations/pipeline_visualizations.py \
     src/visualizations/visualize.py \
	 ./visualizations/

RUN dos2unix run_analysis.sh && \
	dos2unix remove_layers.sh
	
# test data for CI
COPY datafile.tsv ./

ENTRYPOINT [ "bash", "run_analysis.sh" ]