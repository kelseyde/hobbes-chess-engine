EXE ?= hobbes-chess-engine

ifeq ($(OS),Windows_NT)
    EXE := $(EXE).exe
    JOBS := $(NUMBER_OF_PROCESSORS)
    RUSTFLAGS := -C target-cpu=native -C link-arg=/OPT:REF -C link-arg=/OPT:ICF
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        JOBS := $(shell nproc)
        RUSTFLAGS := -C target-cpu=native -C link-arg=-Wl,--as-needed -C link-arg=-Wl,--gc-sections
    else ifeq ($(UNAME_S),Darwin)
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

openbench:
	cargo rustc --release -p hobbes-chess-engine --jobs $(JOBS) -- $(RUSTFLAGS) --emit link=$(EXE)
