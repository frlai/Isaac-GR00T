#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -x

export DOCKER_BUILDKIT=1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$DIR/.."

# Parse --profile and filter script-specific flags before passing to docker
profile="default"
docker_args=()
for arg in "$@"; do
    case $arg in
        --profile=*)
            profile="${arg#--profile=}"
            ;;
        --fix)
            # Skip --fix flag as it's not a valid docker build flag
            ;;
        *)
            docker_args+=("$arg")
            ;;
    esac
done

if [ "$profile" = "thor" ]; then
    image_name="gr00t-thor"
    docker build "${docker_args[@]}" \
        --network host \
        -f "$REPO_ROOT/scripts/deployment/thor/Dockerfile" \
        -t "$image_name" "$REPO_ROOT" \
        && echo "Image $image_name BUILT SUCCESSFULLY"
elif [ "$profile" = "spark" ]; then
    image_name="gr00t-spark"
    docker build "${docker_args[@]}" \
        --network host \
        -f "$REPO_ROOT/scripts/deployment/spark/Dockerfile" \
        -t "$image_name" "$REPO_ROOT" \
        && echo "Image $image_name BUILT SUCCESSFULLY"
elif [ "$profile" = "orin" ]; then
    image_name="gr00t-orin"
    docker build "${docker_args[@]}" \
        --network host \
        -f "$REPO_ROOT/scripts/deployment/orin/Dockerfile" \
        -t "$image_name" "$REPO_ROOT" \
        && echo "Image $image_name BUILT SUCCESSFULLY"
else
    image_name="gr00t"
    docker build "${docker_args[@]}" \
        --network host \
        -f "$DIR/Dockerfile" \
        -t "$image_name" "$REPO_ROOT" \
        && echo "Image $image_name BUILT SUCCESSFULLY"
fi
