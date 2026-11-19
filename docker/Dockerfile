# Administrator builds a CPU contract-smoke image from a verified pinned base.
# Example build argument: python:3.11-slim@sha256:<verified-real-digest>
# No default mutable tag, candidate dependencies or credentials are downloaded.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN mkdir -p /workspace /data /model /output && chown 65532:65532 /output
USER 65532:65532
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/python", "-E", "-s", "-B"]
