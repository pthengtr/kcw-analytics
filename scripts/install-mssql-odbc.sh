#!/usr/bin/env bash
# Install unixODBC + Microsoft ODBC Driver 18 and map KSS -> 192.168.1.99.
# Ubuntu 26.04 has no official msodbcsql18 repo yet; use the 24.04 amd64 package.
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Re-run with sudo: sudo bash $0"
  exit 1
fi

apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y unixodbc unixodbc-dev odbcinst curl

DEB_DIR="$(mktemp -d)"
trap 'rm -rf "${DEB_DIR}"' EXIT
curl -fsSL -o "${DEB_DIR}/msodbcsql18.deb" \
  https://packages.microsoft.com/ubuntu/24.04/prod/pool/main/m/msodbcsql18/msodbcsql18_18.6.2.1-1_amd64.deb

mkdir -p /opt/microsoft/msodbcsql18
touch /opt/microsoft/msodbcsql18/ACCEPT_EULA
echo 'msodbcsql18 msodbcsql/ACCEPT_EULA boolean true' | debconf-set-selections
ACCEPT_EULA=Y dpkg -i "${DEB_DIR}/msodbcsql18.deb"

if ! grep -qE '[[:space:]]KSS([[:space:]]|$)' /etc/hosts; then
  echo '192.168.1.99 KSS kss KSS.local' >> /etc/hosts
fi

echo
odbcinst -q -d
echo
grep -E 'KSS' /etc/hosts
echo "OK: ODBC Driver 18 installed; KSS -> 192.168.1.99"
