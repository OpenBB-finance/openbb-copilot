#!/bin/bash
set -e

if [ -z "${RATE_LIMIT_ENABLED}" ]; then
    echo "WARNING: Rate limit is disabled"
fi

if [ -z "${AUTH_ENABLED}" ]; then
    echo "WARNING: Auth is disabled"
fi

if [ -z "${JINA_AI_BASE_URL}" ] || [ -z "${JINA_AI_API_KEY}" ]; then
    echo "WARNING: URL Retrieval is disabled"
fi

# The CloudWatch agent injects OTEL_TRACES_SAMPLER_ARG as an X-Ray endpoint URL,
# which logfire would otherwise read as its sample-rate fallback and fail to
# parse as a float. LOGFIRE_TRACE_SAMPLE_RATE takes precedence, so logfire
# never reads the OTel var. 1.0 is logfire's own default (sample everything).
export LOGFIRE_TRACE_SAMPLE_RATE="${LOGFIRE_TRACE_SAMPLE_RATE:-1.0}"

echo "Starting OpenBB AI service..."
uvicorn openbb_ada.main:app --no-access-log --loop uvloop --proxy-headers --host 0.0.0.0 --port ${PORT:-7778} --workers 4 $1
