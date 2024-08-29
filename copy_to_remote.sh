#!/bin/bash

DIR=`dirname $0`

REMOTE=tptuser@192.168.2.68:/home/tptuser/Workspace/crux/

BUILD_DIR=${DIR}/build/

scp -r ${BUILD_DIR}/bin ${REMOTE}

scp -r ${DIR}/run.sh ${REMOTE}