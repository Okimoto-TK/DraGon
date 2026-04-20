#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-11111}"
CONTAINER_NAME="${CONTAINER_NAME:-wandb-local}"
VOLUME_NAME="${VOLUME_NAME:-wandb}"

echo "[1/5] checking docker"
if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found, installing via official script"
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  sudo sh /tmp/get-docker.sh
fi

echo "[2/5] enabling docker"
sudo systemctl enable --now docker

echo "[3/5] verifying docker"
sudo docker --version
sudo docker run --rm hello-world

echo "[4/5] installing wandb cli"
if command -v pip3 >/dev/null 2>&1; then
  pip3 install --user wandb
elif command -v pip >/dev/null 2>&1; then
  pip install --user wandb
else
  echo "pip not found, install python3-pip first"
fi

echo "[5/5] starting local wandb server"
sudo docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
sudo docker volume create "${VOLUME_NAME}" >/dev/null
sudo docker run -d \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:8080" \
  -v "${VOLUME_NAME}:/vol" \
  wandb/local

echo
echo "W&B local server should be starting."
echo "Open: http://<your-server-ip>:${PORT}"
echo "If you are on the same machine, open: http://127.0.0.1:${PORT}"
echo
echo "Useful commands:"
echo "  sudo docker logs -f ${CONTAINER_NAME}"
echo "  sudo docker stop ${CONTAINER_NAME}"
echo "  sudo docker start ${CONTAINER_NAME}"
