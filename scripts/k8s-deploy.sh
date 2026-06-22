#!/usr/bin/env sh
set -eu

kubectl apply -k k8s/
kubectl get pods -n genai-assistant
