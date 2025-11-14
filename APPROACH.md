# Algorithmic overview of MicroICS

There are three main computational components that constitute each run.

## Feature construction
Feature construction aims to facilitate the process of identifying geometric (and other!) features from a given 3D image. The main source file that contains the bulk of the logic is `src/feature_generator.py`. Features that are created broadly follow the following algorithm:
1. For each input feature (in parallel)
2. For each layer in image
3. Gather statistics about layer + push statistics about global info about image to a container
4. At the end, dump the statistics to a given file

Statistics aim to extract both density based aspects, as well as general (e.g., homogeneity) aspects of images. The goal of this part is to produce a comprehensive feature set fast, as it's meant to run in parallel for thousands of images on commodity hardware (e.g., Fiji-based flows for similar scope run out of memory). One of the novelties of this approach is automated threshold-based generation -- as we cannot be certain of appropriate thresholds when generating features, we lift the constraint and generate thousands of features corresponding to different e.g., intensity/volume thresholds. This, albeit resulting in many features, enables the algorithms to capture the relevant patterns while remaining general (same threshold for whole data set). Rough sets of features generated:

1. counts (based on intensity, thresholded)
2. Normalized counts (thresholded)
3. Intensity diffs (max - min)
4. Max intensity
5. Median intensity
6. Standard deviation of intensity
7. Mean intensity
8. minProp (new) - proportion of pixels that are below mean intensity
9. Normalized dispersion (std / avg. pixel intensity)
10. Normalized values for all above w.r.t. all layer values
11. BioVolume
12. SubstratumCoverage
13. Homogeneity
14. ThicknessThreshold(min_thr) - thickness, thresholded
15. RoughnessThreshold(min_thr) - roughness, thresholded
16. layerwise global diff - mean
17. layerwise global diff - max
18. layerwise global diff - min
19. Fractal dimension
20. Graycomatrix features (contrast, correlation, dissimilarity, energy)

## Machine learning
Machine learning takes as input the results of feature construction, and attempts to associate the plethora of generated features with the target of interest (strain in most cases). We ran comprehensive experiments with different approaches (`feature_ranking_lite.py`), and as default selected tree ensembles, as they offered good time-performance trade-off. This part of the flow enables insights into learnability of strain properties based on image-derived features. Main use case is of diagnostic nature -- for new images, the trained algorithm will be used to assess the strain/virulence/other properties. Results of machine learning are discussed next (example follows)

```
	tag	model	upsampling	n_components	fold	accuracy	test_set	thr_features
0	RESULT	RandomForestClassifier()	1	32	0	0.5074626865671642	L1323,L1323,L1323,L1323,L1323,L1323,L1323,L1323,L1323,L1823,L1823,L1823,L1823,L1823,L1823,L1823,L1823,L394,L394,L394,L394,L394,L634,L634,L634,L634,L634,L628,L628,L628,L628,L628,L628,L628,L628,L1764,L1764,L1764,L1764,L1764,L1764,L1764,L1764,L634,L634,L634,L634,19115,19115,19115,19115,19115,L455,L455,L455,L455,L455,L394,L394,L394,L455,L455,L455,19115,19115,19115,19115	True
1	RESULT	RandomForestClassifier()	1	32	1	0.7164179104477612	L1323,L1823,L1823,L1823,L1823,L1823,L1823,L1823,L628,L628,L1764,L1764,L634,L634,L634,L634,L634,L634,L1323,L1323,L1323,L1323,L1323,L1764,L1764,L1764,L1764,L1764,L628,L628,L628,L628,L628,L394,L394,L394,L394,L394,L394,L394,L455,L455,L455,L455,L455,L455,L455,L394,L1323,L1323,L1823,L1823,19115,19115,19115,19115,19115,19115,L628,L628,19115,19115,L455,L455,L1764,L634,L634	True
```

Explanation:

1. `tag_model` -> ignore, relevant for gathering by the approach
2. `upsampling` -> We found out that upsampling training data (over-copying it) helps, this is factor for that
3. `n_components` -> if <50, it means features were compressed by using Singular Value Decomposition prior to learning -- this is effectively like doing PCA but faster (without centering)
4. `fold` -> which iteration of learning (which part was used to be predicted on)
5. `accuracy` -> accuracy metric
6. `test_set` -> What were the labels of the test set for this fold
7. `thr_features` -> are we using thresholded features (True/False possible)

## Inference
Inference is the final computational component that applies trained machine learning models to new, unseen biofilm images. Once the feature construction and machine learning steps have been completed, inference enables real-time classification of images without requiring retraining. The trained models are used to predict strain, virulence, or other properties for newly acquired biofilm images, making the system practical for diagnostic applications. This component leverages the trained classifiers to efficiently process new images through the established feature extraction pipeline and produce predictions based on the learned patterns.