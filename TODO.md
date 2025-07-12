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
- [x] RFP fail firm

#### NMP
- [x] NMP
- [x] NMP depth-based reduction
- [x] NMP eval-based reduction
- [ ] NMP verification search

#### LMR
- [x] LMR
- [x] LMR improving
- [ ] LMR PV node
- [x] LMR Cutnode
- [x] LMR History

#### Move-loop pruning
- [x] Late move pruning
- [x] Futility pruning
- [X] QS SEE pruning
- [x] PVS SEE quiet pruning 
- [x] PVS SEE noisy pruning
- [x] QS futility pruning
- [ ] QS delta pruning
- [x] History pruning
- [x] Bad noisy pruning
- [x] Skip quiets

#### Transposition Table
- [ ] Static eval to TT
- [ ] Early static eval write
- [ ] No TT cut in PV nodes
- [ ] TT buckets
- [ ] TT aging
- [ ] TT low depth extension
- [ ] TT cut PV node depth reduction

### Correction History
- [x] Pawn correction history
- [x] Non-pawn correction history
- [x] Minor correction history
- [x] Major correction history
- [x] Countermove correction history
- [x] Follow-up move correction history

### Extensions
- [x] Check extensions
- [x] Singular extensions
- [x] Double extensions
- [ ] Triple extensions
- [x] Negative extensions
- [ ] Double negative extensions
- [ ] Multicut

### Misc search
- [x] IIR
- [x] Cutnode IIR
- [ ] IIR TT depth condition
- [ ] Prior countermove bonus
- [ ] Dynamic policy updates
- [ ] Hindsight reductions
- [ ] Hindsight extensions
- [x] Razoring

## Move Ordering
- [x] MVV-LVA
- [x] Quiet history
- [x] Continuation history 1
- [x] Continuation history 2
- [x] Capture history 
- [X] Killer moves
- [x] Maluses
- [x] Basic movepicker
- [x] TT move before movegen
- [x] Incremental selection sort
- [x] Staged movegen
- [x] Quiet threat history
- [ ] Killer stage
- [x] Split good/bad noisies
- [ ] Use capthist in SEE margin
- [ ] Split good/bad quiets

## Evaluation
- [x] NN
- [x] UE
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

## UCI
- [ ] Configurable Hash size
- [ ] PV printing
- [ ] Nodes/NPS printing
- [ ] Seldepth
- [ ] Hashfull
- [ ] Pretty print