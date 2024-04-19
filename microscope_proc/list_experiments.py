import glob
import pathlib
import sys

def list_experiments(path0: str) -> list[tuple[str, list[str]]]:
    d = dict()

    for sp in glob.glob(path0 + "/*"):
        p = pathlib.Path(sp)
        
        if p.suffix == ".jpg":
            ri = p.stem.rindex("_")
            base_name = p.stem[:ri]
            d[base_name] = d.get(base_name, []) + [sp]

    for di in d:
        d[di].sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # print(d)
    return sorted(((k, v) for k, v in d.items()), key=lambda x: int("".join(x[0].split("_")[1:])))

if __name__ == "__main__":
    
    print(list_experiments(sys.argv[1]))