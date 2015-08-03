# This script pulls P56 Mouse data from Allen website
# required as base data for the brainbuilder
mkdir data
pushd data
wget -O atlasVolume.zip http://api.brain-map.org/api/v2/well_known_file_download/113567585
wget -O P56_Mouse_annotation.zip http://api.brain-map.org/api/v2/well_known_file_download/197642854
wget http://api.brain-map.org/api/v2/structure_graph_download/1.jso
unzip atlasVolume.zip
mkdir P56_Mouse_annotation
pushd P56_Mouse_annotation
unzip ../P56_Mouse_annotation.zip
popd
mv 1.json P56_Mouse_annotation/annotation_hierarchy.json

