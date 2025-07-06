## Search

#### Basics
- [x] Iterative Deepening
- [x] MVV-LVA
- [x] Quiescence Search
- [x] Transposition Table 
- [x] Quiet history
- [x] PVS
- [x] Aspiration windows

#### RFP
- [x] RFP
- [x] RFP improving
- [ ] RFP opponent worsening
- [ ] RFP fail firm

#### NMP
- [x] NMP
- [x] NMP depth-based reduction
- [ ] NMP eval-based reduction
- [ ] NMP verification search

#### LMR
- [x] LMR
- [ ] LMR improving
- [ ] LMR PV node
- [ ] LMR Cutnode
- [ ] LMR History

#### Move-loop pruning
- [ ] Late move pruning
- [x] Futility pruning
- [X] QS SEE pruning
- [x] PVS SEE quiet pruning 
- [x] PVS SEE noisy pruning
- [ ] QS futility pruning
- [ ] QS delta pruning
- [ ] History pruning
- [ ] Bad noisy pruning

### Correction History
- [x] Pawn correction history
- [x] Non-pawn correction history
- [ ] Minor correction history
- [ ] Major correction history
- [ ] Continuation correction history

### Extensions
- [x] Check extensions
- [ ] Singular extensions
- [ ] Double extensions
- [ ] Triple extensions
- [ ] Negative extensions
- [ ] Double negative extensions
- [ ] Multicut

### Misc search
- [ ] IIR
- [ ] Cutnode IIR
- [ ] IIR TT depth condition
- [ ] Prior countermove bonus
- [ ] Dynamic policy updates
- [ ] Hindsight reductions
- [ ] Hindsight extensions

## Move Ordering
- [x] MVV-LVA
- [x] Quiet history
- [x] Continuation history 1
- [ ] Continuation history 2
- [ ] Capture history 
- [X] Killer moves
- [x] Maluses
- [ ] Basic movepicker
- [ ] TT move before movegen
- [ ] Staged movegen
- [ ] Killer stage
- [ ] Split good/bad noisies
- [ ] Use capthist in SEE margin
- [ ] Split good/bad quiets

## Evaluation
- [x] NN
- [ ] UE
- [ ] SIMD
- [ ] Horizontal mirroring
- [ ] Output buckets
- [ ] Lazy updates
- [ ] Finny tables

## Time Management
- [x] Hard bound (applies to the entire search)
- [x] Soft bound (checked on each new depth in the ID loop)
- [ ] Node-based scaling
- [ ] Best move stability
- [ ] Eval stability
