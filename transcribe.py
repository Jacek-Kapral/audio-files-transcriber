"""
Transkrybuje pliki .ogg (i inne audio) do tekstu za pomocą Whisper.

Użycie:
  python transcribe.py                    # wszystkie .ogg z bieżącego katalogu
  python transcribe.py sciezka/do/plikow
  python transcribe.py plik1.ogg plik2.ogg
"""

import argparse
import sys
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description="Transkrypcja audio (Whisper)")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Katalog lub lista plików .ogg",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Model Whisper (base dobry kompromis)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        help="Zapisz transkrypcje do pliku .txt",
    )
    parser.add_argument(
        "--lang",
        default="pl",
        help="Kod języka (pl, en, ...). Puste = auto.",
    )
    return parser.parse_args()


def _collect_ogg(paths):
    ogg_files = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"Ignoruję (nie istnieje): {path}", file=sys.stderr)
            continue
        if path.is_file():
            if path.suffix.lower() in (".ogg", ".oga", ".mp3", ".wav", ".m4a", ".flac"):
                ogg_files.append(path.resolve())
            else:
                print(f"Ignoruję (nie audio): {path}", file=sys.stderr)
        else:
            for ext in ("*.ogg", "*.oga", "*.mp3", "*.wav", "*.m4a", "*.flac"):
                ogg_files.extend(path.resolve().glob(ext))
    return sorted(set(ogg_files))


def main():
    args = _parse_args()
    try:
        import whisper
    except ImportError:
        print("Zainstaluj: pip install openai-whisper", file=sys.stderr)
        sys.exit(1)

    files = _collect_ogg(args.paths)
    if not files:
        print("Brak plików audio do transkrypcji.", file=sys.stderr)
        sys.exit(1)

    model = whisper.load_model(args.model)
    lang = args.lang if args.lang else None
    results = []

    for f in files:
        print(f"Transkrypcja: {f.name} ...")
        result = model.transcribe(str(f), language=lang, fp16=False)
        text = result["text"].strip()
        results.append((f.name, text))
        print(f"--- {f.name}\n{text}\n")

    if args.out:
        args.out = Path(args.out)
        if args.out.suffix.lower() != ".txt":
            args.out = args.out.with_suffix(args.out.suffix + ".txt")
        with open(args.out, "w", encoding="utf-8") as out:
            for name, text in results:
                out.write(f"--- {name}\n{text}\n\n")
        print(f"Zapisano: {args.out}")


if __name__ == "__main__":
    main()
