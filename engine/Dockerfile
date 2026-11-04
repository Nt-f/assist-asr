# asr-engine on llama.cpp's own image: the same libraries (CPU variants, Vulkan or CUDA by BASE), our engine
# compiled against the headers of exactly that build. LLAMA_BUILD and the image tag must move together.
#   docker build -f engine/Dockerfile --build-arg VARIANT=server-vulkan -t assist-asr-engine:vulkan .
ARG LLAMA_BUILD=b12098
ARG VARIANT=server
FROM ghcr.io/ggml-org/llama.cpp:${VARIANT}-${LLAMA_BUILD} AS build
ARG LLAMA_BUILD
RUN apt-get update && apt-get install -y --no-install-recommends g++ curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /src && curl -fsSL "https://github.com/ggml-org/llama.cpp/archive/refs/tags/${LLAMA_BUILD}.tar.gz" \
    | tar -xz -C /src --strip-components=1
COPY engine/engine.cpp /src/engine.cpp
RUN g++ -O2 -std=c++17 -I/src/include -I/src/ggml/include -I/src/tools/mtmd -I/src/vendor/cpp-httplib \
      -I/src/vendor/nlohmann /src/engine.cpp /src/vendor/cpp-httplib/httplib.cpp \
      -L/app -lllama -lmtmd -lggml -lggml-base -lpthread -Wl,-rpath,/app -o /app/asr-engine

FROM ghcr.io/ggml-org/llama.cpp:${VARIANT}-${LLAMA_BUILD}
COPY --from=build /app/asr-engine /app/asr-engine
COPY engine/entrypoint.sh /app/asr-engine.sh
ENTRYPOINT ["/app/asr-engine.sh"]
