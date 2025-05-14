EXE ?= hobbes-chess-engine

openbench:
	cargo rustc --release -p hobbes-chess-engine -- -C target-cpu=native --emit link=$(EXE)