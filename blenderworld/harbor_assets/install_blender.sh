#!/bin/bash
set -euo pipefail
version=4.2.10
filename="blender-${version}-linux-x64.tar.xz"
base_url="https://download.blender.org/release/Blender4.2"
checksum=7b888b36cf5bd7e35d495eb9717f7a9b645ea5f2fb4fe9850991800b8f08d531
curl --fail --location --retry 3 "${base_url}/${filename}" -o "/tmp/${filename}" || \
    curl --fail --location --retry 3 "https://mirrors.ocf.berkeley.edu/blender/release/Blender4.2/${filename}" -o "/tmp/${filename}"
cd /tmp
printf '%s  %s\n' "$checksum" "$filename" | sha256sum --check -
mkdir -p /opt/blender
tar -xJf "$filename" --strip-components=1 -C /opt/blender
/opt/blender/4.2/python/bin/python3.11 -m ensurepip
/opt/blender/4.2/python/bin/python3.11 -m pip install --no-cache-dir requests==2.32.5
rm "/tmp/${filename}"
/opt/blender/blender --background --factory-startup --python-exit-code 2 --python-expr 'import bpy, requests; assert bpy.app.version == (4, 2, 10), bpy.app.version_string; print(bpy.app.version_string)'
