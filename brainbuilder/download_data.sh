#!/usr/bin/env bash
# This script pulls P56 Mouse data from Allen website
# required as base data for the brainbuilder
mkdir -p data
pushd data
wget -O atlasVolume.zip http://api.brain-map.org/api/v2/well_known_file_download/113567585
wget -O P56_Mouse_annotation.zip http://api.brain-map.org/api/v2/well_known_file_download/197642854
wget http://api.brain-map.org/api/v2/structure_graph_download/1.json
unzip atlasVolume.zip
mkdir -p P56_Mouse_annotation
pushd P56_Mouse_annotation
unzip ../P56_Mouse_annotation.zip
popd
mv 1.json P56_Mouse_annotation/annotation_hierarchy.json

kinit
mkdir -p bbp_recipe
scp bbpviz1.cscs.ch:/gpfs/bbp.cscs.ch/project/proj1/entities/bionames/SomatosensoryCxS1-v5.r0/bluerecipe_release_ChC_intervention_GSYNrescale/*.xml bbp_recipe/
scp bbpviz1.cscs.ch:/gpfs/bbp.cscs.ch/project/proj1/entities/morphologies/2012.07.23/v4_24.09.14Final_NoBadClonesMEGatedAntiStuck0_NoL6MCsubs_ih_exp_SomatosensoryCxS1-v5_r0_NeuronDB.dat bbp_recipe/neurondb.dat

mkdir -p bbp_circuits
pushd bbp_circuits
mkdir -p SomatosensoryCxS1-v5.r0_O1
pushd SomatosensoryCxS1-v5.r0_O1
mkdir -p 0 1 2 3 4 5 6 merged_circuit
scp bbpviz1.cscs.ch:/gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/O1/0/circuit.mvd2 0/
scp bbpviz1.cscs.ch:/gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/O1/1/circuit.mvd2 1/
scp bbpviz1.cscs.ch:/gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/O1/2/circuit.mvd2 2/
scp bbpviz1.cscs.ch:/gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/O1/3/circuit.mvd2 3/
scp bbpviz1.cscs.ch:/gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/O1/4/circuit.mvd2 4/
scp bbpviz1.cscs.ch:/gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/O1/5/circuit.mvd2 5/
scp bbpviz1.cscs.ch:/gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/O1/6/circuit.mvd2 6/
scp bbpviz1.cscs.ch:/gpfs/bbp.cscs.ch/project/proj1/circuits/SomatosensoryCxS1-v5.r0/O1/merged_circuit/circuit.mvd2 merged_circuit/
popd  # SomatosensoryCxS1-v5.r0_O1

popd  # bbp_circuits

mkdir -p hippo_recipe
pushd hippo_recipe
scp bbpviz1.cscs.ch:/gpfs/bbp.cscs.ch/project/proj42/circuits/CA1draftModel/morphologies/v4neuronDB.dat .
scp bbpviz1.cscs.ch:/gpfs/bbp.cscs.ch/project/proj42/circuits/CA1draftModel/bionames/builderRecipeAllPathways.xml .
popd  # hippo_recipe

popd  # data
