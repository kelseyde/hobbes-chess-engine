EXE ?= hobbes-chess-engine

ifeq ($(OS),Windows_NT)
	EXE := $(EXE).exe
endif

openbench:
	cargo rustc --release -p hobbes-chess-engine -- -C target-cpu=native --emit link=$(EXE)