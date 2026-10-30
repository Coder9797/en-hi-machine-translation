# Dataset

This project uses a subset of the IIT Bombay English–Hindi Parallel Corpus.

- Creators: Anoop Kunchukuttan, Pratik Mehta, and Pushpak Bhattacharyya.
- Paper: *The IIT Bombay English-Hindi Parallel Corpus*, LREC 2018.
- Corpus page: https://www.cfilt.iitb.ac.in/iitb_parallel/
- Public mirror used for download: https://huggingface.co/datasets/cfilt/iitb-english-hindi
- License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Noncommercial use only. The processed subset is an excerpt, not a new mixed database.

Training and validation sentences are sampled from the corpus training split. Test sentences are sampled from the corpus held-out test split. The sample sizes and the random seed are in `configs/experiment.yaml`. The exact pairs used for a run are written to `data/processed/` by `python -m src.pipeline --stages data` and are not redistributed as the full corpus.
