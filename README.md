# Repository for DR-FFIT (manuscript)

This repository contains the code used to implement DR-FFIT, the simulators, the search algorithms, the tests, and the analysis of the results.

The "lib" directory contains the DR-FFIT implementation, the simulators and other miscellaneous functions. The "Feature_extraction" directory contains the implementation of the black-box feature functions (AutoEncoders and PCA).

The "server_search_algorithms" directory contains the scripts used to produce the data from the first experiment (with initialization). The same scripts were modified to produce the "no initialization" data by changing the starting point, increasing the number of iterations, and changing the file names/data paths. 

The "figures" directory contains the figures for the manuscript.
