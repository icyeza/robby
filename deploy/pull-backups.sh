#!/bin/sh
# Pull Robby's encrypted backups to an off-site machine (run there, e.g. from cron or Task
# Scheduler). Only already-encrypted files and their checksums are copied, and each checksum is
# verified after the copy.
#
#   sh deploy/pull-backups.sh <ssh host> <local directory>
set -eu

[ $# -eq 2 ] || { echo "usage: $0 <ssh host> <local directory>" >&2; exit 2; }
host="$1"; dest="$2"
mkdir -p "$dest"
scp -q "$host:/srv/robby-backups/*.age" "$host:/srv/robby-backups/*.sha256" "$dest/"
cd "$dest" && sha256sum -c --quiet ./*.sha256 && echo "pulled and verified $(ls ./*.age | wc -l) backups"
