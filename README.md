<p align="center"><img src="resources/hobbes.png" width="180"></p>

<div align="center">

<h3>Hobbes</h3>
A chess engine written in Rust.

<br>
<br>
  <a href="https://github.com/kelseyde/hobbes-chess-engine/releases"><strong>Download Hobbes</strong></a> | <a href="https://github.com/kelseyde/hobbes-chess-engine/issues"><strong>Report a Bug</strong></a>
<br>
<br>

[![release][release-badge]][release-link]
[![License][license-badge]][license-link]

</div>

## Overview

A strong chess engine written in Rust, with NNUE evaluation trained from zero knowledge, using self-generated training data. 

Hobbes started off as a rewrite of my Java chess engine, [Calvin](https://github.com/kelseyde/calvin-chess-engine), although it has now surpassed Calvin by quite some distance. It is also my first project in the Rust programming language. 

## Strength 

|                                   Version                                   | Release Date |  🎯 Elo  | CCRL Blitz | CCRL Rapid | CEGT Rapid |
|:---------------------------------------------------------------------------:|:------------:|:--------:|:----------:|:----------:|:----------:|
| [2.1](https://github.com/kelseyde/hobbes-chess-engine/releases/tag/2.1) |  2026-05-26  | **3725** |     -      | - |     -      |
| [2.0](https://github.com/kelseyde/hobbes-chess-engine/releases/tag/2.0) |  2026-05-25  | **3725** |     -      | - |     -      |
| [1.0](https://github.com/kelseyde/hobbes-chess-engine/releases/tag/1.0) |  2026-03-05  | **3715** |     3716 (#19)      | 3572 (#20) |     -      |

## Search

Hobbes features a classical alpha-beta negamax search with iterative deepening. Other features include aspiration windows, quiescence search, principle variation search, reverse futility pruning, null move pruning, singular extensions, and many more. 

## Evaluation

Hobbes uses a multilayer, efficiently updated neural network (NNUE) for its evaluation function. The architecture of the network is `(704x16hm->1536pw)x2->(16x2->32->1)x8`. Other NNUE optimisations that Hobbes employs are sparse matrix multiplication, fused refreshes, Finny tables, and lazy updates.

The network is trained entirely on data generated from self-play. The training data is a (roughly) 80/20 split of standard data and DFRC data. The network was initialised from random values and trained up over many iterations; the full history of past nets is documented [here](https://github.com/kelseyde/hobbes-chess-engine/blob/main/network_history.txt). All of hobbes' networks have been trained using [bullet](https://github.com/jw1912/bullet).

## Building Hobbes

Official Hobbes binaries can be downloaded from the [Releases](https://github.com/kelseyde/hobbes-chess-engine/releases) page. 

If you would like to build Hobbes from source, it is first necessary to [install Rust](https://www.rust-lang.org/tools/install).

Then, you need to download Hobbes' latest neural network. The latest network name is recorded in [network.txt](https://github.com/kelseyde/hobbes-chess-engine/blob/main/network.txt). Then, execute this command in the root directory (substituting {lastest_net} for the net name):

`curl -L -o hobbes.nnue https://github.com/kelseyde/hobbes-networks/releases/download/{latest_net}/{latest_net}.nnue`

Then, call `cargo build -r`. The executable will be created in the `target/release` directory.

Please note, building Hobbes from source using these steps will create a basic executable with SIMD disabled. If you desire to build an optimised, tournament-ready executable from source, please refer to the Makefile, or else contact me directly.

## Acknowledgements

- Mattia and Jonathan, the authors of [Heimdall](https://github.com/nocturn9x/heimdall) and [Pawnocchio](https://github.com/JonathanHallstrom/pawnocchio) respectively, who both contributed major improvements in the early days of Hobbes.
- The members of the [MattBench](https://chess.n9x.co/index/) OpenBench instance who have contributed hardware towards running Hobbes tests.
- Jamie, the author of [bullet](https://github.com/jw1912/bullet), an incredible tool for training NNUEs among other things, with which I have trained all of Hobbes's neural networks.
- Other engines, including (in no particular order): [Stockfish](https://github.com/official-stockfish/Stockfish), [Reckless](https://github.com/codedeliveryservice/Reckless), [Integral](https://github.com/aronpetko/integral), [Sirius](https://github.com/mcthouacbb/Sirius), [Stormphrax](https://github.com/Ciekce/Stormphrax), [Pawnocchio](https://github.com/JonathanHallstrom/pawnocchio) and [Simbelmyne](https://github.com/sroelants/simbelmyne).
- The Stockfish discord community for endless amounts of useful information.
- Bill Watterson, the author of the [comic book](https://en.wikipedia.org/wiki/Calvin_and_Hobbes) which gave this engine its name.

[release-badge]: https://img.shields.io/github/v/release/kelseyde/hobbes-chess-engine?style=for-the-badge&color=A8DEFF
[release-link]: https://github.com/kelseyde/hobbes-chess-engine/releases/latest

[license-badge]: https://img.shields.io/github/license/kelseyde/hobbes-chess-engine?style=for-the-badge&color=fab157
[license-link]: https://github.com/kelseyde/hobbes-chess-engine/blob/main/LICENSE
