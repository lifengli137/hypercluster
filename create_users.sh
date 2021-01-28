#!/bin/bash

if [ "${UID}" -ne 0 ]; then
  echo "This script must be run as root (e.g. sudo $0)" >&2
  exit 2
fi

USERNAME="$1"
SSH_PUBKEY="$2"

if [ -z "${USERNAME}" ]; then
  echo "You must provide a username" >&2
  exit 1
fi

if [ -z "${SSH_PUBKEY}" ]; then
  echo "You must provide an SSH Public Key" >&2
  exit 1
fi

if grep -G "^%sudo" /etc/sudoers | grep -q 'NOPASSWD:'; then
  echo "WARN: The sudoers configuration does not appear to have NOPASSWD" >&2
  echo "The sudo group configuration is recommended to read" >&2
  echo >&2
  echo " %sudo ALL=(ALL:ALL) NOPASSWD: ALL" >&2
fi

adduser --disabled-password "${USERNAME}"
usermod -aG sudo "${USERNAME}"
USER_HOME=$(grep "${USERNAME}" /etc/passwd | cut -d':' -f6)
mkdir -p ${USER_HOME}/.ssh
echo "${SSH_PUBKEY}" >> ${USER_HOME}/.ssh/authorized_keys

echo "User ${USERNAME} created and public key populated"
