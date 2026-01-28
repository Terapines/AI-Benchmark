#!/bin/bash

DIR=`dirname $0`
# source configuration file
source ${DIR}/config.sh

REMOTE=${REMOTE}

BUILD_DIR=${DIR}/build/


scp -r ${REMOTE}/bin/ ${BUILD_DIR}/
