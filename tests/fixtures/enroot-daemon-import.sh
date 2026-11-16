# Copyright (c) 2018-2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Unmodified daemon-import function from the installed Enroot helper.
# Tests replace only its external commands with CPU shell functions.

docker::daemon::import() (
    local -r uri="$1"
    local filename="$2" arch="$3"
    local -r extract_mode="$4"
    local image= tmpdir= engine=

    case "${uri}" in
    dockerd://*)
        engine="docker" ;;
    podman://*)
        engine="podman" ;;
    esac

    common::checkcmd jq "${engine}" mksquashfs tar

    # Parse the image reference of the form 'dockerd://<image>[:<tag>]'.
    local -r reg_image="[[:alnum:]/._:-]+"

    if [[ "${uri}" =~ ^[[:alpha:]]+://(${reg_image})$ ]]; then
        image="${BASH_REMATCH[1]}"
    else
        common::err "Invalid image reference: ${uri}"
    fi

    # Convert the architecture to the debian format.
    if [ -n "${arch}" ]; then
        arch=$(common::debarch "${arch}")
    fi

    # Generate an absolute filename if none was specified.
    if [ -z "${filename}" ]; then
        filename="${image//[:\/]/+}.sqsh"
    fi
    filename=$(common::realpath "${filename}")
    if [ -e "${filename}" ]; then
        common::err "File already exists: ${filename}"
    fi

    # Create a temporary directory and chdir to it.
    trap 'common::rmall "${tmpdir}" 2> /dev/null; docker rm -f -v "${tmpdir##*/}" > /dev/null 2>&1' EXIT
    tmpdir=$(common::mktmpdir enroot)
    common::chdir "${tmpdir}"

    # Download the image (if necessary) and create a container for extraction.
    common::log INFO "Fetching image" NL
    # TODO Use --platform once it comes out of experimental.
    "${engine}" create --name "${PWD##*/}" "${image}" >&2
    common::log

    # Extract and configure the rootfs.
    if [[ "${engine}" == "podman" ]] && [[ "${extract_mode}" == "mount" ]]; then
        common::log INFO "Mounting image from Podman container..."
        printf -v cmd "source %q/docker.sh && podman::_mount_squash %q %q %q %q" "${ENROOT_LIBRARY_PATH}" "${image}" "${arch}" "${PWD##*/}" "${filename}"
        podman unshare bash -c "$cmd"
    else
        common::log INFO "Extracting image content..."
        mkdir rootfs
        "${engine}" export "${PWD##*/}" | tar -C rootfs --warning=no-timestamp --anchored --exclude='dev/*' --exclude='.dockerenv' -px
        common::fixperms rootfs
        docker::daemon::_config_squash "${engine}" "${image}" "${arch}" rootfs "${filename}"
    fi

    # Cleanup the container created to extract the rootfs
    "${engine}" rm "${PWD##*/}"
)
