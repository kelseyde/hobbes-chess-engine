EXE ?= hobbes-chess-engine

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    DEFAULT_NET := $(file < network.txt)
    EXE := $(EXE).exe
    JOBS := $(NUMBER_OF_PROCESSORS)
    RUSTFLAGS := -C target-cpu=native -C link-arg=/OPT:REF -C link-arg=/OPT:ICF
else
    DETECTED_OS := $(shell uname -s)
    DEFAULT_NET := $(shell cat network.txt)
    ifeq ($(DETECTED_OS),Linux)
        JOBS := $(shell nproc)
        RUSTFLAGS := -C target-cpu=native -C link-arg=-Wl,--as-needed -C link-arg=-Wl,--gc-sections
    else ifeq ($(DETECTED_OS),Darwin)
        JOBS := $(shell sysctl -n hw.ncpu)
        RUSTFLAGS := -C target-cpu=native -C link-arg=-Wl,-dead_strip
    else
        JOBS := 4
        RUSTFLAGS := -C target-cpu=native
    endif
endif

export RUSTFLAGS
export CARGO_BUILD_JOBS := $(JOBS)
export CARGO_INCREMENTAL := 1

openbench: download-net
	cargo rustc --release -p hobbes-chess-engine --jobs $(JOBS) -- $(RUSTFLAGS) --emit link=$(EXE)

download-net:
	$(info Downloading network $(DEFAULT_NET).nnue)
	curl -L -o hobbes.nnue https://github.com/kelseyde/hobbes-networks/releases/download/$(DEFAULT_NET)/$(DEFAULT_NET).nnue