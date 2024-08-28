#!/bin/bash

DIR=`dirname $0`

REMOTE=tptuser@192.168.2.68:/home/tptuser/Workspace/crux/

BUILD_DIR=${DIR}/build/


scp -r ${REMOTE}/bin/ ${BUILD_DIR}/
