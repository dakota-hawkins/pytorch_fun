# µFormer sun 2023.

Three levels of scores
1. single residue
2. motif level
3. sequence semantics

## µFormer Architecture
(1) PLM (pre-trained)
   - 30 mil UniRef50 sequences
   - masked language modeling strategy
   - Output for fine tuning:
     (i) sequence embedding
     (ii) residue probabilities

(2) Scoring Modules
   (a) sequence scoring (semantics → function)
      - predict mutational effects
   (b) motif scoring (local patterns)
      - extract consequential sequence patterns
   (c) residue level (grammar validity)
      - calculate fitness scores of given sequence

→ combined for final prediction

Additional:
PLM: uses pairwise masked language model (PMLM) to consider dependencies among masked tokens

## Assessing Epistasis
$$S_{epi} = Effect_{p_1,p_2 \ldots p_n} - \sum_i^n Effect_{p_i}$$

Protein Design via Reinforcement Learning
- search variants with 1 < n < 5 mutations
- µFormer acts as reward function
- Noise injected with Dirichlet PPO Algorithm

Unseen Residues

Ablation Studies
- take apart model & reconstruct with different component configurations
  - dropping LLM part produces largest drop
  - scoring modules have target specific effects
  - exact LLM less important
