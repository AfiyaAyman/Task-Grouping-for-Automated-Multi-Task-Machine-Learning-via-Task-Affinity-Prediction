# Task Grouping for Automated Multi-Task Machine Learning via Task Affinity Prediction


**ABSTRACT**

When a number of similar tasks have to be learned simultaneously, multi-task
learning (MTL) models can attain significantly higher accuracy than single-task
learning (STL) models. However, the advantage of MTL depends on various factors,
such as the similarity of the tasks, the sizes of the datasets, and so on; in fact, some
tasks might not benefit from MTL and may even incur a loss of accuracy compared
to STL. Hence, the question arises: which tasks should be learned together?
Domain experts can attempt to group tasks together following intuition, experience,
and best practices, but manual grouping can be labor-intensive and far from optimal.
In this paper, we propose a novel automated approach for task grouping. First,
we study the affinity of tasks for MTL using four benchmark datasets that have
been used extensively in the MTL literature, focusing on neural network-based
MTL models. We identify inherent task features and STL characteristics that can
help us to predict whether a group of tasks should be learned together using MTL
or if they should be learned independently using STL. Building on this predictor,
we introduce a randomized search algorithm, which employs the predictor to
minimize the number of MTL trainings performed during the search for task
groups. We demonstrate on the four benchmark datasets that our predictor-driven
search approach can find better task groupings than existing baseline approaches.

Full Study: https://arxiv.org/pdf/2310.16241
--------------------------------------------------------------------------------------------------
[Requirements]:
--------------------------------------------------------------------------------------------------
Python 3.8+

install requirements in requirements.txt


[Directory Description]:
--------------------------------------------------------------------------------------------------
grouping_with_affinity_predictor: Contains scripts for Task-Affinity Predictor (Quick_Reject_Class.py), Randomized Search with Quick Reject Implementation (*_QR.py)

**Dataset**: Contains all benchmarks data and some datafiles for all the scripts to run without error in separate subdirectories

**DataPrep**: Contains separate scripts to prepare data for baseline approach and Randomized Search with Quick Reject

**Baseline**: Contains scripts for all baseline approaches in separate subdirectory -

          * simple_MTL: Contains scripts for simple MTL approach
          * random_search_exhaustive: Contains scripts for Quasi-Exhaustive Search (1000 random partitions)
          * clustering: Contains scripts for clustering based approaches
          * inter_task_affinity: Contains scripts for Inter-Task Affinity calculations
          * generate_data: Contains scripts to generate data for some of these baseline approaches

**Pairwise_Affinity**: Contains scripts needed for the Pairwise-Affinity study

**Groupwise_Affinity**: Contains the following scripts needed for the Groupwise-Affinity study

**Results**: Folder to contain Results from different studies that would further be used for our proposed approach

--------------------------------------------------------------------------------------------------

# Scripts Description

Our Proposed Approach:
--------------------------------------------------------------------------------------------------
***grouping_with_affinity_predictor***
----------------------------------------
Scripts-

* school_QR.py → Generates results for Randomized Search with Quick Reject for School Benchmark
* chemical_QR.py → Generates results for Randomized Search with Quick Reject for Chemical Benchmark
* landmine_QR.py → Generates results for Randomized Search with Quick Reject for Landmine Benchmark
* Parkinson_QR.py → Generates results for Randomized Search with Quick Reject for Parkinson Benchmark
* Quick_Reject_Class.py → Class contains attributes and methods for the Task-Affinity Predictor

Baseline Approaches:
--------------------------------------------------------------------------------------------------

**Simple_MTL**
----------------------------------------

* simple_MTL/all_task_MTL_School.py → Generates results for Simple MTL for School benchmarks, also gathers the weight vectors for all tasks
* simple_MTL/all_task_MTL_Chemical.py → Generates results for Simple MTL for Chemical benchmarks, also gathers the weight vectors for all tasks
* simple_MTL/all_task_MTL_Landmine.py → Generates results for Simple MTL for Landmine benchmarks, also gathers the weight vectors for all tasks
* simple_MTL/all_task_MTL_Parkinson.py → Generates results for Simple MTL for Parkinson benchmarks, also gathers the weight vectors for all tasks

**Random_Search_Exhaustive**
----------------------------------------
* random_search_exhaustive/random_search_School.py → Generates results for Randomized Search for 1000 different partitions (Quasi Exhaustive) for School benchmarks

* random_search_exhaustive/random_search_Chemical.py → Generates results for Randomized Search for 1000 different partitions (Quasi Exhaustive) for Chemical benchmarks

* random_search_exhaustive/random_search_Landmine.py → Generates results for Randomized Search for 1000 different partitions (Quasi Exhaustive) for Landmine benchmarks

* random_search_exhaustive/random_search_Parkinson.py → Generates results for Randomized Search for 1000 different partitions (Quasi Exhaustive) for Parkinson benchmarks


**Clustering**
----------------------------------------
* clustering/clustering_School.py → Generates results for different clusters generated by classic clustering algorithms for School benchmarks

* clustering/clustering_Chemical.py → Generates results for different clusters generated by classic clustering algorithms for Chemical benchmarks

* clustering/clustering_Landmine.py → Generates results for different clusters generated by classic clustering algorithms for Landmine benchmarks

* clustering/clustering_Parkinson.py → Generates results for different clusters generated by classic clustering algorithms for Parkinson benchmarks

----------------------------------------
* generate_data/clustering_hierarchical_WeightMatrix.py → Generates data for Hierarchical clustering all benchmarks
* generate_data/clustering_KMeans_WeightMatrix.py → Generates data for K-Means clustering all benchmarks
* generate_data/generate_hierarchical_MTL_affinities.py → Generates data for Hierarchical clustering using actual MTL Affinities for all benchmarks
* generate_data/generate_random_partitions.py → Generates 1000 different partitions from all possible partitions with uniform probability for all benchmarks


**Inter_Task_Affinity**
----------------------------------------
* inter_task_affinity/ITA_School.py → Generates inter-task affinities for each task pair (calculations are based on Fifty et al. 2021) for School benchmarks
* inter_task_affinity/ITA_Chemical.py → Generates inter-task affinities for each task pair (calculations are based on Fifty et al. 2021) for Chemical benchmarks
* inter_task_affinity/ITA_Landmine.py → Generates inter-task affinities for each task pair (calculations are based on Fifty et al. 2021) for Landmine benchmarks
* inter_task_affinity/ITA_Parkinson.py → Generates inter-task affinities for each task pair (calculations are based on Fifty et al. 2021) for Parkinson benchmarks



--------------------------------------------------------------------------------------------------
Study of Task Affinity:
--------------------------------------------------------------------------------------------------
** Pairwise Multi-Task Learning Affinity**
--------------------------------------------

***pairwise_affinity***

* school_STL.py → Generates results for Single Task Learning for School Benchmark

* chemical_STL.py → Generates results for Single Task Learning for Chemical Benchmark

* landmine_STL.py → Generates results for Single Task Learning for Landmine Benchmark

* Parkinson_STL.py → Generates results for Single Task Learning for Parkinson Benchmark

school_pairwise_MTL.py → Generates results for Pairwise Multi-Task Learning for School Benchmark

chemical_pairwise_MTL.py → Generates results for Pairwise Multi-Task Learning for Chemical Benchmark

landmine_pairwise_MTL.py → Generates results for Pairwise Multi-Task Learning for Landmine Benchmark

Parkinson_pairwise_MTL.py → Generates results for Pairwise Multi-Task Learning for Parkinson Benchmark

* pairwise_affinity.py → Generates pearson correlation, NN prediction performance on all benchmarks -> first in starts predictor_data_prep.py which generates Features (Task-Specific and Task-Relation) on all benchmarks

** Groupwise Multi-Task Learning Affinity**
--------------------------------------------

***groupwise_affinity***

* initial_training_school.py → Generates MTL results for the initial partitions for School Benchmark

* initial_training_chemical.py → Generates MTL results for the initial partitions for Chemical Benchmark

* initial_training_landmine.py → Generates MTL results for the initial partitions for Landmine Benchmark

* predicting_group_affinity.py → Generates pearson correlation, NN prediction performance for groups on all benchmarks


--------------------------------------------------------------------------------------------------