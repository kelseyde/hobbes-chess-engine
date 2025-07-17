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
- [ ] RFP constant offset
- [ ] RFP prev-move history
- [ ] RFP parent PV

#### NMP
- [x] NMP
- [x] NMP depth-based reduction
- [x] NMP eval-based reduction
- [x] NMP TT capture
- [ ] Increment NMP base reduction
- [ ] NMP verification search
- [ ] NMP TT condition

#### LMR
- [x] LMR
- [x] LMR improving
- [x] LMR PV node
- [ ] LMR TT-PV
- [x] LMR Cutnode
- [x] LMR History
- [ ] Noisy LMR
- [x] Fractional LMR
- [ ] Factorised LMR
- [ ] PV-distance LMR
- [ ] Corrplexity LMR
- [ ] Futility LMR
- [ ] LMR if no TT-PV
- [ ] Cutoff-count LMR

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
- [ ] FP history
- [x] FP movecount
- [ ] PVS SEE quiet history
- [ ] PVS SEE noisy history
- [ ] Use LMR depth in more places
- [x] Qs evasion pruning
- [ ] Qs guard recaptures

#### Transposition Table
- [ ] Static eval to TT
- [ ] Early static eval write (Qs)
- [ ] Early static eval write (PVS)
- [x] No TT cut in PV nodes
- [ ] TT buckets
- [ ] TT aging
- [ ] TT low depth extension
- [ ] TT cut PV node depth reduction
- [ ] Better replacement scheme
- [ ] SF TT aging
- [ ] Qs standpat TT store
- [ ] "Would TT prune" PV reduction

### Correction History
- [x] Pawn correction history
- [x] Non-pawn correction history
- [x] Minor correction history
- [x] Major correction history
- [x] Countermove correction history
- [x] Follow-up move correction history
- [ ] Gravity corrhist

### Extensions
- [x] Check extensions
- [x] Singular extensions
- [x] Double extensions
- [ ] Triple extensions
- [x] Negative extensions
- [ ] Double negative extensions
- [x] Multicut

### Misc search
- [x] IIR
- [x] Cutnode IIR
- [ ] IIR TT depth condition
- [x] Hindsight reductions
- [x] Hindsight extensions
- [x] Razoring
- [x] Alpha raise reductions
- [ ] Probcut
- [ ] SF small probcut idea
- [ ] Deeper/shallower

## Move Ordering / History
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
- [x] Prior countermove bonus
- [ ] Dynamic policy updates
- [ ] Threats capthist
- [ ] History factoriser
- [x] Post-LMR update conthist
- [ ] History depth alpha bonus
- [ ] History depth beta bonus

## Evaluation
- [x] NN
- [x] UE
- [x] SIMD
- [x] Horizontal mirroring
- [x] Output buckets
- [ ] Lazy updates
- [x] Finny tables
- [ ] Remove unnecessary copy

## Time Management
- [x] Hard bound (applies to the entire search)
- [x] Soft bound (checked on each new depth in the ID loop)
- [x] Node-based scaling
- [ ] Best move stability
- [ ] Eval stability

## UCI
- [x] Configurable Hash size
- [ ] PV printing
- [x] Nodes/NPS printing
- [ ] Seldepth
- [x] Hashfull
- [ ] Pretty print
