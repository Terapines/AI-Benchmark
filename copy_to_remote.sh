#!/bin/bash

DIR=`dirname $0`

REMOTE=tptuser@192.168.2.68:/home/tptuser/Workspace/crux/

BUILD_DIR=${DIR}/build/

scp -r ${BUILD_DIR}/bin ${REMOTE}

scp -r ${DIR}/run.sh ${REMOTE}

# FIXME: One script for run and report
#!/bin/bash
# DIR=`dirname $0`
# REMOTE=tptuser@192.168.2.24:/home/tptuser/Workspace/draco/
# BUILD_DIR=${DIR}/build/
# ssh tptuser@192.168.2.24 "rm -rf /home/tptuser/Workspace/draco/*"
# scp -r ${BUILD_DIR}/bin ${REMOTE}
# scp -r ${DIR}/run.sh ${REMOTE}
# ssh tptuser@192.168.2.24 "bash -x /home/tptuser/Workspace/draco/run.sh"
# ./copy_remote_back.sh
# ./report.sh