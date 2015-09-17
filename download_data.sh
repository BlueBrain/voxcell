# This script pulls P56 Mouse data from Allen website
# required as base data for the brainbuilder
mkdir data
pushd data
wget -O atlasVolume.zip http://api.brain-map.org/api/v2/well_known_file_download/113567585
wget -O P56_Mouse_annotation.zip http://api.brain-map.org/api/v2/well_known_file_download/197642854
wget http://api.brain-map.org/api/v2/structure_graph_download/1.json
unzip atlasVolume.zip
mkdir P56_Mouse_annotation
pushd P56_Mouse_annotation
unzip ../P56_Mouse_annotation.zip
popd
mv 1.json P56_Mouse_annotation/annotation_hierarchy.json

mkdir -p bbp_recipe
scp bbpviz1.cscs.ch:/gpfs/bbp.cscs.ch/project/proj1/entities/bionames/SomatosensoryCxS1-v5.r0/bluerecipe_release_ChC_intervention_GSYNrescale/*.xml bbp_recipe/
scp bbpviz1.cscs.ch:/gpfs/bbp.cscs.ch/project/proj1/entities/morphologies/2012.07.23/v4_24.09.14Final_NoBadClonesMEGatedAntiStuck0_NoL6MCsubs_ih_exp_SomatosensoryCxS1-v5_r0_NeuronDB.dat bbp_recipe/neurondb.dat
