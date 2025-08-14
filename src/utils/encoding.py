from typing import Optional

def read_text_safely(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    try:
        from charset_normalizer import from_path
        result = from_path(path).best()
        if result is not None:
            return str(result)
    except Exception:
        pass

    with open(path, "rb") as f:
        data = f.read()
    return data.decode("utf-8", errors="replace")
