"""Raw Hessian data and parser for water."""

import numpy as np

HESSIAN_RAW = """
      1  2  3  4  5
      1  0.558922D+00
      2 -0.864213D-01  0.497715D+00
      3  0.000000D+00  0.000000D+00 -0.375567D-03
      4 -0.479241D+00  0.675216D-02  0.000000D+00  0.495540D+00
      5 -0.618227D-01 -0.490774D-01  0.000000D+00  0.792307D-02  0.514860D-01
      6  0.000000D+00  0.000000D+00  0.187783D-03  0.000000D+00  0.000000D+00
      7 -0.796808D-01  0.796691D-01  0.000000D+00 -0.162987D-01  0.538996D-01
      8  0.148244D+00 -0.448638D+00  0.000000D+00 -0.146752D-01 -0.240857D-02
      9  0.000000D+00  0.000000D+00  0.187783D-03  0.000000D+00  0.000000D+00
                6             7             8             9
      6 -0.197503D-03
      7  0.000000D+00  0.959794D-01
      8  0.000000D+00 -0.133569D+00  0.451046D+00
      9  0.971976D-05  0.000000D+00  0.000000D+00 -0.197503D-03
"""


def parse_hessian(raw_str, dim=9):
    """Parse lower triangular Hessian string into symmetric matrix."""
    matrix = np.zeros((dim, dim))
    lines = raw_str.strip().split("\n")
    col_indices = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if all(part.isdigit() for part in parts):
            col_indices = [int(x) - 1 for x in parts]
            continue

        row_idx = int(parts[0]) - 1
        values = parts[1:]
        for i, val_str in enumerate(values):
            val = float(val_str.replace("D", "E"))
            col_idx = col_indices[i]
            matrix[row_idx, col_idx] = val
            matrix[col_idx, row_idx] = val

    return matrix


if __name__ == "__main__":
    hessian = parse_hessian(HESSIAN_RAW)
    print("Hessian parsed successfully.")
    print(f"Shape: {hessian.shape}")
    print(f"Sample (0,0): {hessian[0,0]}")
