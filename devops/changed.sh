#!/bin/bash

# main script entrypoint 
function main {
    # get previous and current commit heads
    # jenkins offers this as environment variables within Jenkins GIT Plugin
    GIT_PREVIOUS_COMMIT=$(git rev-parse --short "HEAD^")
    GIT_COMMIT=$(git rev-parse --short HEAD)

    get_changed_microservices
    echo "$changed_microservices"
}

function get_changed_microservices {
    folders=`git diff --name-only $GIT_PREVIOUS_COMMIT $GIT_COMMIT | sort -u | awk 'BEGIN {FS="/"} {print $1}' | uniq`
    export changed_microservices=$folders
}
main
