<div align="center">

<p align="center"><img src="resources/hobbes.png" width="180"></p>

<h3>Hobbes</h3>
<b>A chess engine written in Rust.</b>

<br>
<br>

[![Beta][beta-badge]][beta-link]
[![License][license-badge]][license-link]

<br>

</div>

A strong chess engine written in Rust, with NNUE evaluation trained from zero knowledge, using self-generated training data.

Hobbes is a rewrite of my Java chess engine, [Calvin](https://github.com/kelseyde/calvin-chess-engine). Hobbes is also a collaborative effort from the members of [mattbench](https://chess.n9x.co/index/), who have each contributed hardware (and patches) towards turning Hobbes into a strong engine. 

## Strength

Hobbes is not yet officially released. The current strength is approximately 3640 elo CCRL blitz.

## Search

Hobbes features a classical alpha-beta negamax search with iterative deepening. Other features include aspiration windows, quiescence search, principle variation search, reverse futility pruning, null move pruning, singular extensions, and many more. 

## Evaluation

Hobbes uses an efficiently updated neural network (NNUE) for its evaluation function. The architecture of the network is `(768x16hm->1280)x2->1`. Other NNUE optimisations that Hobbes employs are fused refreshes, Finny tables, and lazy updates.

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

[beta-badge]: https://img.shields.io/badge/version-BETA-A8DEFF?style=for-the-badge
[beta-link]: https://github.com/kelseyde/hobbes-chess-engine

[license-badge]: https://img.shields.io/github/license/kelseyde/hobbes-chess-engine?style=for-the-badge&color=fab157
[license-link]: https://github.com/kelseyde/hobbes-chess-engine/blob/main/LICENSE
