"""
ChatHBA REPL — 対話型チャット推論
===================================

使い方:
  python chat_repl.py
  python chat_repl.py --checkpoint=chat_hba_best.pt --temperature=0.8

チェックポイント: chat_hba_best.pt (chat_hba.py の学習で生成)
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# chat_hba.py と同じディレクトリ前提
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from chat_hba import (  # noqa: E402
  ChatHBA,
  CHECKPOINT_PATH,
  _END_TOKEN,
  _PAD_TOKEN,
  format_qa,
  load_corpus,
  build_vocab,
  generate_answer,
  CORPUS_PATH,
)

def load_model_and_vocab(
  checkpoint_path: Path,
  context_len: int = 64,
  hidden_dim: int = 48,
  n_layers: int = 2,
  n_heads: int = 4,
) -> tuple:
  """チェックポイントからモデルと vocab を復元。"""

  print(f"[repl] loading corpus for vocab rebuild...")
  records = load_corpus(CORPUS_PATH)
  stoi, itos = build_vocab(records)
  vocab_size = len(stoi)
  print(f"[repl] vocab_size={vocab_size}")

  model = ChatHBA(
  vocab_size=vocab_size,
  context_len=context_len,
  hidden_dim=hidden_dim,
  n_layers=n_layers,
  n_heads=n_heads,
  )

  state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
  model.load_state_dict(state)
  model.eval()
  print(f"[repl] checkpoint loaded: {checkpoint_path}")
  print(f"[repl] params: {model.param_count():,}")
  return model, stoi, itos

def repl(
  model: ChatHBA,
  stoi: dict,
  itos: dict,
  context_len: int = 64,
  temperature: float = 0.8,
  tau: float = 0.1,
  max_new_chars: int = 80,
):
  """対話ループ。"""
  print()
  print("=" * 50)
  print("  ChatHBA REPL  (type 'bye' or Ctrl-C to quit)")
  print("=" * 50)
  print()

  while True:
  try:
  q = input(">>> ").strip()
  except (EOFError, KeyboardInterrupt):
  print("\n[repl] bye.")
  break

  if not q:
  continue
  if q.lower() in {"bye", "quit", "exit"}:
  print("[repl] bye.")
  break

  ans = generate_answer(
  model, q, stoi, itos,
  context_len=context_len,
  max_new_chars=max_new_chars,
  temperature=temperature,
  tau=tau,
  )
  print(f"HBA: {ans}")
  print()

def main():
  checkpoint = CHECKPOINT_PATH
  context_len = 64
  hidden_dim  = 48
  n_layers  = 2
  n_heads  = 4
  temperature = 0.8
  tau  = 0.1

  for arg in sys.argv[1:]:
  if  arg.startswith("--checkpoint="):  checkpoint  = Path(arg.split("=",1)[1])
  elif arg.startswith("--context-len="): context_len = int(arg.split("=",1)[1])
  elif arg.startswith("--hidden-dim="):  hidden_dim  = int(arg.split("=",1)[1])
  elif arg.startswith("--n-layers="):  n_layers  = int(arg.split("=",1)[1])
  elif arg.startswith("--n-heads="):  n_heads  = int(arg.split("=",1)[1])
  elif arg.startswith("--temperature="): temperature = float(arg.split("=",1)[1])
  elif arg.startswith("--tau="):  tau  = float(arg.split("=",1)[1])

  if not Path(checkpoint).exists():
  print(f"[error] checkpoint not found: {checkpoint}")
  print("[error] まず chat_hba.py で学習してください。")
  sys.exit(1)

  model, stoi, itos = load_model_and_vocab(
  checkpoint, context_len, hidden_dim, n_layers, n_heads
  )
  repl(model, stoi, itos, context_len, temperature, tau)

if __name__ == "__main__":
  main()
