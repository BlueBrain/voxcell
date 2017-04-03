#modules that have tests
TEST_MODULES=voxcell/voxcell brainbuilder/brainbuilder
#modules that are installable (ie: ones with setup.py)
INSTALL_MODULES=voxcell voxcellview brainbuilder
#packages to cover
COVER_PACKAGES=voxcell,voxcellview,brainbuilder
IGNORE_LINT=voxcellview|notebooks
# documentation to build, separated by spaces
DOC_MODULES=voxcell/doc brainbuilder/doc
PYTHON_PIP_VERSION=pip==9.0.1
##### DO NOT MODIFY BELOW #####################

ifndef CI_DIR
CI_REPO?=ssh://bbpcode.epfl.ch/platform/ContinuousIntegration.git
CI_DIR?=ContinuousIntegration

FETCH_CI := $(shell \
		if [ ! -d $(CI_DIR) ]; then \
			git clone $(CI_REPO) $(CI_DIR) > /dev/null ;\
		fi;\
		echo $(CI_DIR) )
endif

include $(CI_DIR)/python/common_makefile
