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

## Evaluation

## Building Hobbes

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
